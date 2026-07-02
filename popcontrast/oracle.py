"""Exact full-catalog scoring ("global search" oracle) + segmented recall.

Because SIDs are only `sem_id_len` tokens and the whole catalog fits in memory,
we compute the *exact* p(item | history) for every item by teacher-forcing its
token sequence, reusing the encoded history across item chunks. This is a harder,
cheaper oracle than beam=100 (IDEA.md §4.3, decision locked 2026-07-01).

The same machinery, wrapped in `pop_steer_hooks`, measures whether PopSteer
injection lifts tail recall monotonically in alpha without collapsing head recall
— the §7 Step-3 minimal-validation signal. Ranking by exact scores decouples the
signal from beam-search noise; trie-beam efficiency is reported separately.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from popcontrast.data_utils import SemIdTables, Popularity, MAX_HISTORY_TOKENS
from popcontrast.hidden_states import pop_intervention_hooks


@torch.no_grad()
def encode_history(
    model, history_items: Sequence[int], tables: SemIdTables,
    max_history_tokens: int = MAX_HISTORY_TOKENS, device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode one user's history once -> (enc_hidden (1,H,d), attn (1,H))."""
    toks: List[int] = []
    for it in history_items:
        toks.extend(tables.item_tokens[it])
    if len(toks) > max_history_tokens:
        toks = toks[-max_history_tokens:]
    input_ids = torch.zeros(1, max_history_tokens, dtype=torch.long, device=device)
    attn = torch.zeros(1, max_history_tokens, dtype=torch.long, device=device)
    start = max_history_tokens - len(toks)
    input_ids[0, start:] = torch.tensor(toks, dtype=torch.long, device=device)
    attn[0, start:] = 1
    enc = model.model.encoder(input_ids=input_ids, attention_mask=attn)[0]
    return enc, attn


@torch.no_grad()
def score_all_items(
    model, enc: torch.Tensor, attn: torch.Tensor,
    item_tokens: torch.Tensor, chunk_size: int = 16384,
) -> torch.Tensor:
    """Exact log p(item | history) for every item. item_tokens: (num_items, L) long.

    Returns (num_items,) log-prob scores. Reuses `enc` across chunks. Default
    chunk_size scores the whole catalog in one forward. Any active intervention
    (pop_intervention_hooks) fires inside model.model here.
    """
    num_items, L = item_tokens.shape
    scores = torch.empty(num_items, device=enc.device)
    for s in range(0, num_items, chunk_size):
        labels = item_tokens[s:s + chunk_size]          # (C, L)
        C = labels.shape[0]
        enc_c = enc.expand(C, -1, -1)
        attn_c = attn.expand(C, -1)
        out = model.model(encoder_outputs=(enc_c,), attention_mask=attn_c, labels=labels)
        logp = torch.log_softmax(out.logits.float(), dim=-1)   # (C, L, vocab)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (C, L)
        scores[s:s + chunk_size] = tok_logp.sum(dim=-1)  # joint log-prob
    return scores


@torch.no_grad()
def score_all_items_perstep(
    model, enc: torch.Tensor, attn: torch.Tensor,
    item_tokens: torch.Tensor, chunk_size: int = 16384,
) -> torch.Tensor:
    """Per-step token log-probs for every item. Returns (num_items, L).

    Same forward as score_all_items but keeps the per-SID-step log p(t_j | h, t_<j)
    instead of summing — needed for per-step adaptive contrastive decoding.
    """
    num_items, L = item_tokens.shape
    out = torch.empty(num_items, L, device=enc.device)
    for s in range(0, num_items, chunk_size):
        labels = item_tokens[s:s + chunk_size]
        C = labels.shape[0]
        o = model.model(encoder_outputs=(enc.expand(C, -1, -1),),
                        attention_mask=attn.expand(C, -1), labels=labels)
        logp = torch.log_softmax(o.logits.float(), dim=-1)
        out[s:s + chunk_size] = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return out


def _item_tokens_tensor(tables: SemIdTables, device: str) -> torch.Tensor:
    return torch.tensor(tables.item_tokens, dtype=torch.long, device=device)


@dataclass
class RecallResult:
    """Segmented ranking metrics at several K."""
    overall: Dict[int, Dict[str, float]]   # k -> {recall, ndcg}
    head: Dict[int, Dict[str, float]]
    tail: Dict[int, Dict[str, float]]
    coverage: Dict[int, float]             # k -> fraction of catalog ever recommended
    n_users: int
    n_head_users: int
    n_tail_users: int

    def line(self, k: int) -> str:
        o, h, t = self.overall[k], self.head[k], self.tail[k]
        return (f"@{k}: R={o['recall']:.4f} N={o['ndcg']:.4f} | "
                f"head R={h['recall']:.4f} tail R={t['recall']:.4f} | "
                f"cov={self.coverage[k]:.4f}")


@torch.no_grad()
def evaluate_exact(
    model,
    samples: Sequence[dict],
    tables: SemIdTables,
    pop: Popularity,
    k_list: Sequence[int] = (5, 10),
    inject: Optional[dict] = None,
    max_history_tokens: int = MAX_HISTORY_TOKENS,
    max_n: Optional[int] = None,
    chunk_size: int = 4096,
    device: str = "cuda",
) -> RecallResult:
    """Exact top-K ranking metrics over `samples`, segmented by target bucket.

    inject: None for the unmodified oracle, or dict(layer=, v_pop=, alpha=, tau=)
            to rank *under* PopSteer injection (the §7 Step-3 test).
    """
    if max_n is not None:
        samples = samples[:max_n]
    item_tok = _item_tokens_tensor(tables, device)
    max_k = max(k_list)
    bucket = pop.bucket

    # accumulators
    rec = {seg: {k: [] for k in k_list} for seg in ("overall", "head", "tail")}
    ndcg = {seg: {k: [] for k in k_list} for seg in ("overall", "head", "tail")}
    recommended = {k: set() for k in k_list}

    def _run(enc, attn):
        if inject is None:
            return score_all_items(model, enc, attn, item_tok, chunk_size)
        with pop_intervention_hooks(model, inject["layer"], inject["v_pop"],
                                    inject["alpha"], inject.get("tau"),
                                    mode=inject.get("mode", "steer")):
            return score_all_items(model, enc, attn, item_tok, chunk_size)

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())
    for s in samples:
        tgt = s["target"]
        seg = "head" if bucket[tgt] == "head" else "tail"
        with autocast:
            enc, attn = encode_history(model, s["history"], tables, max_history_tokens, device)
            scores = _run(enc, attn)
        topk = torch.topk(scores, max_k).indices.cpu().numpy()  # item ids, best first

        # Because SIDs can collide, several items share a token seq; a hit is when
        # the target item id appears in the ranked list. (Exact-item recall.)
        hit_pos = np.where(topk == tgt)[0]
        rank = hit_pos[0] if len(hit_pos) else None
        for k in k_list:
            hit = rank is not None and rank < k
            r = 1.0 if hit else 0.0
            n = (1.0 / np.log2(rank + 2)) if hit else 0.0
            rec["overall"][k].append(r); ndcg["overall"][k].append(n)
            rec[seg][k].append(r); ndcg[seg][k].append(n)
            recommended[k].update(topk[:k].tolist())

    def _agg(d):
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in d.items()}

    n_items = len(tables.item_tokens)
    return RecallResult(
        overall={k: {"recall": _agg(rec["overall"])[k], "ndcg": _agg(ndcg["overall"])[k]} for k in k_list},
        head={k: {"recall": _agg(rec["head"])[k], "ndcg": _agg(ndcg["head"])[k]} for k in k_list},
        tail={k: {"recall": _agg(rec["tail"])[k], "ndcg": _agg(ndcg["tail"])[k]} for k in k_list},
        coverage={k: len(recommended[k]) / n_items for k in k_list},
        n_users=len(samples),
        n_head_users=sum(1 for s in samples if bucket[s["target"]] == "head"),
        n_tail_users=sum(1 for s in samples if bucket[s["target"]] == "tail"),
    )
