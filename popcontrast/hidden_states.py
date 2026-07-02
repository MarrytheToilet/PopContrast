"""Decoder residual-stream capture (for probe/CAA) and PopSteer injection.

A forward hook on `model.model.decoder.block[l]` sees the block's output tuple,
whose element [0] is the post-block hidden state (B, seq, d_model). We:
  - CAPTURE  output[0] during teacher forcing -> probe/CAA training data, tagged
             by the target item's popularity bucket and decode step.
  - INJECT   a norm-preserving pop-debiasing edit by returning a modified tuple
             during generation (fires each decode step automatically).

Injection (matches IDEA.md §3.1 Step 3-4):
    h~ = ||h|| * normalize( h/||h|| - alpha_eff * v_pop )
    alpha_eff = alpha * sigmoid(<h, v_pop> - tau)   [optional gating; tau=None -> static]
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from popcontrast.data_utils import SemIdTables, Popularity, offset_encode, MAX_HISTORY_TOKENS
from popcontrast.model_utils import decoder_blocks


# --------------------------------------------------------------------------- #
# Minibatch builder (keeps target item_id, unlike genrec's collate)
# --------------------------------------------------------------------------- #

def build_batch(
    samples: Sequence[dict],
    tables: SemIdTables,
    max_history_tokens: int = MAX_HISTORY_TOKENS,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(input_ids, attention_mask, labels, target_item_ids) from ds.samples slices.

    History items -> flattened offset tokens (left-padded). Target item -> its
    offset token sequence as labels. We keep target_item_ids so we can look up the
    popularity bucket (genrec's collate discards the item id).
    """
    B = len(samples)
    input_ids = torch.zeros(B, max_history_tokens, dtype=torch.long)
    attention_mask = torch.zeros(B, max_history_tokens, dtype=torch.long)
    labels = torch.empty(B, tables.sem_id_len, dtype=torch.long)
    target_ids = torch.empty(B, dtype=torch.long)

    for i, s in enumerate(samples):
        toks: List[int] = []
        for it in s["history"]:
            toks.extend(tables.item_tokens[it])
        if len(toks) > max_history_tokens:
            toks = toks[-max_history_tokens:]
        start = max_history_tokens - len(toks)
        input_ids[i, start:] = torch.tensor(toks, dtype=torch.long)
        attention_mask[i, start:] = 1
        labels[i] = torch.tensor(tables.item_tokens[s["target"]], dtype=torch.long)
        target_ids[i] = s["target"]

    return (
        input_ids.to(device),
        attention_mask.to(device),
        labels.to(device),
        target_ids.to(device),
    )


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

@contextmanager
def capture_hooks(model, layers: Sequence[int]):
    """Register capture hooks on the given decoder blocks; yield a store dict.

    store[l] holds the most recent block-output hidden state (B, seq, d_model).
    """
    store: Dict[int, torch.Tensor] = {}
    blocks = decoder_blocks(model)
    handles = []

    def make_hook(l):
        def hook(_module, _inp, output):
            store[l] = output[0].detach()
        return hook

    for l in layers:
        handles.append(blocks[l].register_forward_hook(make_hook(l)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@dataclass
class HiddenCollection:
    """Captured decoder hidden states for probe/CAA.

    H[(layer, step)] -> np.ndarray (N, d_model)
    y[(layer, step)] -> np.ndarray (N,) with 1=head-target, 0=tail-target
    """
    H: Dict[Tuple[int, int], np.ndarray]
    y: Dict[Tuple[int, int], np.ndarray]
    layers: List[int]
    steps: int


@torch.no_grad()
def collect_hidden(
    model,
    samples: Sequence[dict],
    tables: SemIdTables,
    pop: Popularity,
    layers: Sequence[int],
    max_history_tokens: int = MAX_HISTORY_TOKENS,
    batch_size: int = 128,
    max_n: Optional[int] = None,
    device: str = "cuda",
) -> HiddenCollection:
    """Teacher-forcing forward; capture per-(layer, step) hidden states.

    At decode step j, the hidden state that predicts target token j is captured
    and labelled by the target item's popularity bucket. This is the data for the
    A1 probe (head/tail linear separability) and the CAA/RepE direction extraction.
    """
    if max_n is not None:
        samples = samples[:max_n]
    layers = list(layers)
    steps = tables.sem_id_len
    bucket = pop.bucket  # np array of "head"/"tail"

    buf_H: Dict[Tuple[int, int], List[np.ndarray]] = {(l, j): [] for l in layers for j in range(steps)}
    buf_y: Dict[Tuple[int, int], List[np.ndarray]] = {(l, j): [] for l in layers for j in range(steps)}

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        input_ids, attn, labels, tgt = build_batch(chunk, tables, max_history_tokens, device)
        with capture_hooks(model, layers) as store:
            # Tiger.forward(input_ids, attention_mask, labels) -> teacher forcing.
            model(input_ids=input_ids, attention_mask=attn, labels=labels)
        y = (bucket[tgt.cpu().numpy()] == "head").astype(np.int64)  # (B,)
        for l in layers:
            h = store[l]  # (B, steps, d_model): decoder positions predicting each target token
            for j in range(steps):
                buf_H[(l, j)].append(h[:, j, :].float().cpu().numpy())
                buf_y[(l, j)].append(y)

    H = {k: np.concatenate(v, axis=0) for k, v in buf_H.items()}
    yc = {k: np.concatenate(v, axis=0) for k, v in buf_y.items()}
    return HiddenCollection(H=H, y=yc, layers=layers, steps=steps)


# --------------------------------------------------------------------------- #
# Injection (PopSteer)
# --------------------------------------------------------------------------- #

def _norm_preserving_debias(
    h: torch.Tensor, v_pop: torch.Tensor, alpha: float, tau: Optional[float]
) -> torch.Tensor:
    """h~ = ||h|| * normalize(h/||h|| - alpha_eff * v_pop).  v_pop assumed unit-norm."""
    norm = h.norm(dim=-1, keepdim=True)                      # (..., 1)
    unit = h / norm.clamp_min(1e-8)
    if tau is None:
        alpha_eff = alpha
    else:
        proj = (h * v_pop).sum(dim=-1, keepdim=True)         # <h, v_pop>
        alpha_eff = alpha * torch.sigmoid(proj - tau)
    steered = unit - alpha_eff * v_pop
    steered = steered / steered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return norm * steered


def _directional_ablation(
    h: torch.Tensor, v_pop: torch.Tensor, alpha: float, tau: Optional[float],
    clamp: bool = False,
) -> torch.Tensor:
    """Project OUT the popularity component: h~ = h - alpha_eff * (h . v_pop) v_pop.

    Unlike fixed-vector subtraction (which pushes every token the same distance and
    tends to collapse outputs), ablation removes each token's *own* projection onto
    v_pop — the standard 'remove a concept direction' operator (Arditi et al. 2024,
    refusal direction; cf. LEACE). alpha=1 fully ablates; tau gates it.

    clamp=True -> one-sided: only remove the POSITIVE popularity component
    (h - max(0, h.v) v), so tail-leaning states (projection<=0) are untouched and we
    never inject anti-popularity. Strongest coverage protection (Arditi clamped variant).
    """
    proj = (h * v_pop).sum(dim=-1, keepdim=True)             # <h, v_pop>
    if clamp:
        proj = proj.clamp_min(0.0)
    if tau is None:
        a = alpha
    else:
        a = alpha * torch.sigmoid(proj - tau)
    return h - a * proj * v_pop


@contextmanager
def pop_intervention_hooks(
    model,
    layer: int,
    v_pop: torch.Tensor,
    alpha: float,
    tau: Optional[float] = None,
    mode: str = "steer",   # "steer" = norm-preserving subtraction; "ablate" = projection-out
):
    """Generic residual-stream intervention at decoder block `layer` output.

    mode='steer'  -> h~ = ||h|| normalize(h/||h|| - alpha_eff v_pop)  (original CAA-style)
    mode='ablate' -> h~ = h - alpha_eff (h . v_pop) v_pop             (directional ablation)
    """
    v = v_pop.to(next(model.parameters()).device)
    v = v / v.norm().clamp_min(1e-8)
    block = decoder_blocks(model)[layer]

    def hook(_module, _inp, output):
        h = output[0]
        if mode == "ablate":
            h_new = _directional_ablation(h, v, alpha, tau, clamp=False)
        elif mode == "ablate_clamp":
            h_new = _directional_ablation(h, v, alpha, tau, clamp=True)
        elif mode == "add":  # inject +v (for the Arditi symmetric causal test)
            h_new = h + alpha * v
        else:
            h_new = _norm_preserving_debias(h, v, alpha, tau)
        return (h_new,) + tuple(output[1:])

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def pop_steer_hooks(
    model,
    layer: int,
    v_pop: torch.Tensor,
    alpha: float,
    tau: Optional[float] = None,
):
    """Inject norm-preserving pop-debiasing at the output of decoder block `layer`.

    Fires on every decode step during generation (and every position in teacher
    forcing). v_pop is a (d_model,) unit vector on the model's device.
    """
    v = v_pop.to(next(model.parameters()).device)
    v = v / v.norm().clamp_min(1e-8)
    block = decoder_blocks(model)[layer]

    def hook(_module, _inp, output):
        h = output[0]
        h_new = _norm_preserving_debias(h, v, alpha, tau)
        return (h_new,) + tuple(output[1:])

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
