"""Trie-constrained beam search — the real GR decoding path.

genrec's vendored beam search (t5.py) is *unconstrained*: it can emit token
combinations that aren't any real item. TIGER's whole efficiency argument rests
on trie-constrained decoding within the catalog. We implement it standalone here
(reusing the model's cached forward) so we can:
  - decode only *valid* items (proper GR baseline),
  - apply PopSteer injection during real beam decoding (wrap in pop_steer_hooks),
  - compare small-beam vs exact-oracle coverage (the tail-recovery diagnostic),
  - report latency / #forward-passes to defend "we did not degrade to full-catalog
    scoring" (IDEA.md §4.3 efficiency).

Mask: at decode step j, a beam whose already-decoded offset-token prefix is
`p` may only emit tokens in `tables.allowed_next(p)`; everything else -> -inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from popcontrast.data_utils import SemIdTables


NEG_INF = -1e9


@dataclass
class BeamOutput:
    item_ids: List[List[int]]      # per user: ranked valid item ids (best first)
    scores: List[List[float]]      # per user: matching log-prob scores
    n_forward: int                 # decoder forward passes used (efficiency metric)


def _build_prefix_mask(
    prefixes: List[Tuple[int, ...]], tables: SemIdTables, vocab_size: int, device
) -> torch.Tensor:
    """(num_beams, vocab) additive mask: 0 for allowed next tokens, -inf otherwise."""
    mask = torch.full((len(prefixes), vocab_size), NEG_INF, device=device)
    for b, pref in enumerate(prefixes):
        allowed = tables.allowed_next(pref)
        if allowed:
            mask[b, allowed] = 0.0
        else:
            # dead prefix (shouldn't happen mid-item); leave fully masked -> beam dies
            mask[b, 0] = 0.0
    return mask


@torch.no_grad()
def trie_beam_search(
    model,
    enc: torch.Tensor,          # (1, H, d) encoded history (reuse encode_history)
    attn: torch.Tensor,         # (1, H)
    tables: SemIdTables,
    num_beams: int = 20,
    device: str = "cuda",
) -> Tuple[List[int], List[float], int]:
    """Constrained beam search for ONE user. Returns (item_ids, scores, n_forward).

    Ranks *items*; when several items share a token sequence (SID collision), all
    are emitted at that sequence's score (stable order).
    """
    vocab = model.model.config.vocab_size
    L = tables.sem_id_len
    start_id = model.model.config.decoder_start_token_id

    enc_b = enc.expand(num_beams, -1, -1)
    attn_b = attn.expand(num_beams, -1)

    dec = torch.full((num_beams, 1), start_id, dtype=torch.long, device=device)
    beam_scores = torch.full((num_beams,), NEG_INF, device=device)
    beam_scores[0] = 0.0
    past = None
    n_forward = 0

    for step in range(L):
        step_in = dec if past is None else dec[:, -1:]
        out = model.model(
            encoder_outputs=(enc_b,), attention_mask=attn_b,
            decoder_input_ids=step_in, use_cache=True, past_key_values=past,
        )
        past = out.past_key_values
        n_forward += 1

        logits = out.logits[:, -1, :]                       # (num_beams, vocab)
        logp = F.log_softmax(logits, dim=-1)

        # trie mask by each beam's decoded prefix (offset tokens after start)
        prefixes = [tuple(dec[b, 1:].tolist()) for b in range(dec.shape[0])]
        logp = logp + _build_prefix_mask(prefixes, tables, vocab, device)

        cand = beam_scores.unsqueeze(1) + logp              # (num_beams, vocab)
        flat = cand.view(-1)
        k = min(num_beams, flat.numel())
        top_scores, top_idx = flat.topk(k, largest=True, sorted=True)
        beam_idx = top_idx // vocab
        tok_idx = top_idx % vocab

        dec = torch.cat([dec[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        beam_scores = top_scores
        past = _reorder_past(past, beam_idx)

    # decode -> items (handle SID collisions via tokens_to_items)
    item_ids: List[int] = []
    scores: List[float] = []
    seen = set()
    order = torch.argsort(beam_scores, descending=True)
    for b in order.tolist():
        toks = tuple(dec[b, 1:].tolist())
        for it in tables.tokens_to_items.get(toks, []):
            if it not in seen:
                seen.add(it)
                item_ids.append(it)
                scores.append(float(beam_scores[b]))
    return item_ids, scores, n_forward


def _reorder_past(past, beam_idx):
    if past is None:
        return None
    reordered = []
    for layer_cache in past:
        new_layer = []
        for kv in layer_cache:
            if kv is None:
                new_layer.append(None)
            else:
                k, v = kv
                new_layer.append((k[beam_idx], v[beam_idx]))
        reordered.append(tuple(new_layer))
    return tuple(reordered)
