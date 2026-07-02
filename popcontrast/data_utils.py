"""Foundational data utilities shared by Plan A (PopSteer) and Plan B (PopContrast).

Everything here reuses genrec's `AmazonSeqDataset` so we stay byte-for-byte aligned
with whatever the trained TIGER checkpoint saw. Three things live here:

  1. build_dataset()      -- instantiate the genrec sequence dataset for a split.
  2. compute_popularity() -- per-item interaction counts -> head/tail buckets.
  3. SemIdTables          -- item<->SID<->offset-token maps + the decoding trie,
                             which the exact oracle and the trie-beam constraint use.

IMPORTANT alignment note: the TIGER gin config sets `add_disambiguation=False`
(3-code SIDs, collisions allowed). The dataset class *defaults* to True. Always
pass add_disambiguation matching training, or the SIDs won't match the checkpoint.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# genrec dataset. Import is cheap; instantiation loads the RQ-VAE + builds SIDs.
from genrec.data.amazon import AmazonSeqDataset


# Defaults that match config/tiger/amazon/tiger.gin + rqvae.gin
DEFAULT_ENCODER = "/home/hanyu/models/sentence-t5-xl"
DEFAULT_RQVAE_PATH = "out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt"
CODEBOOK_SIZE = 256
SEM_ID_DIM = 3            # number of RQ-VAE codebooks (before disambiguation)
ADD_DISAMBIGUATION = False  # must match training
# tiger.gin: train.max_seq_len=50 -> max_history_tokens = 50 * sem_id_dim = 150.
# MUST match training or the model is starved of history (baseline recall tanks).
MAX_HISTORY_TOKENS = 150


def build_dataset(
    split: str = "beauty",
    train_test_split: str = "train",
    *,
    root: str = "dataset/amazon",
    rqvae_path: str = DEFAULT_RQVAE_PATH,
    encoder_model_name: str = DEFAULT_ENCODER,
    add_disambiguation: bool = ADD_DISAMBIGUATION,
    max_seq_len: int = 20,
    codebook_size: int = CODEBOOK_SIZE,
    n_layers: int = SEM_ID_DIM,
) -> AmazonSeqDataset:
    """Instantiate the genrec sequence dataset, aligned with the TIGER config."""
    return AmazonSeqDataset(
        root=root,
        split=split,
        train_test_split=train_test_split,
        max_seq_len=max_seq_len,
        add_disambiguation=add_disambiguation,
        pretrained_rqvae_path=rqvae_path,
        encoder_model_name=encoder_model_name,
        rqvae_input_dim=768,
        rqvae_embed_dim=32,
        rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=codebook_size,
        rqvae_n_layers=n_layers,
    )


# --------------------------------------------------------------------------- #
# Popularity  ->  head / tail buckets
# --------------------------------------------------------------------------- #

@dataclass
class Popularity:
    """Per-item interaction counts and head/tail assignment.

    counts[item_id]  -> number of training interactions
    bucket[item_id]  -> "head" or "tail"
    head_frac        -> fraction of *items* placed in head (top by count)
    """
    counts: np.ndarray            # shape (num_items,), int
    bucket: np.ndarray            # shape (num_items,), dtype=object/str
    head_ids: np.ndarray          # item ids in head
    tail_ids: np.ndarray          # item ids in tail
    head_frac: float

    @property
    def num_items(self) -> int:
        return len(self.counts)


def compute_popularity(
    ds: AmazonSeqDataset,
    head_frac: float = 0.20,
    exclude_heldout: bool = True,
) -> Popularity:
    """Count per-item interactions over the *training* portion of each sequence.

    head = top `head_frac` of items by interaction count (ties broken by id).
    tail = the rest. `head_frac` is scannable for the A4 threshold-sensitivity
    ablation. Set exclude_heldout=False to count over full sequences.
    """
    # num_items = number of distinct items = len of the SID list built by the dataset
    num_items = len(ds.sem_ids_list)
    counts = np.zeros(num_items, dtype=np.int64)

    for seq in ds.sequences:
        items = seq[:-2] if exclude_heldout else seq  # leave-one-out: last 2 held out
        for it in items:
            if 0 <= it < num_items:
                counts[it] += 1

    # Rank items by count (descending); top head_frac -> head.
    order = np.argsort(-counts, kind="stable")
    n_head = max(1, int(round(head_frac * num_items)))
    head_ids = np.sort(order[:n_head])
    tail_ids = np.sort(order[n_head:])

    bucket = np.empty(num_items, dtype=object)
    bucket[head_ids] = "head"
    bucket[tail_ids] = "tail"

    return Popularity(
        counts=counts,
        bucket=bucket,
        head_ids=head_ids,
        tail_ids=tail_ids,
        head_frac=head_frac,
    )


# --------------------------------------------------------------------------- #
# SID <-> offset-token maps + decoding trie
# --------------------------------------------------------------------------- #

def offset_encode(codes: Sequence[int], codebook_size: int = CODEBOOK_SIZE) -> List[int]:
    """genrec's flat offset encoding: token = code + pos*codebook_size + 1 (0=pad)."""
    return [c + j * codebook_size + 1 for j, c in enumerate(codes)]


@dataclass
class SemIdTables:
    """Item <-> SID <-> offset-token tables and the prefix trie used for decoding.

    - item_codes[item_id]      -> raw SID code list (len = sem_id_dim [+1 if disambig])
    - item_tokens[item_id]     -> offset-encoded token list (the decoder targets)
    - trie[prefix_tuple]       -> sorted list of allowed next offset-tokens
                                  (prefix_tuple is a tuple of offset tokens; () = root)
    - tokens_to_items[tuple]   -> list of item ids sharing that full token sequence
                                  (>1 only when add_disambiguation=False and codes collide)
    """
    item_codes: List[List[int]]
    item_tokens: List[List[int]]
    trie: Dict[Tuple[int, ...], List[int]]
    tokens_to_items: Dict[Tuple[int, ...], List[int]]
    codebook_size: int
    sem_id_len: int  # length of each token sequence (3, or 4 with disambiguation)

    def allowed_next(self, prefix_tokens: Tuple[int, ...]) -> List[int]:
        """Valid next tokens given the offset-token prefix already decoded."""
        return self.trie.get(tuple(prefix_tokens), [])


def build_sem_id_tables(
    ds: AmazonSeqDataset,
    codebook_size: int = CODEBOOK_SIZE,
) -> SemIdTables:
    """Build offset-token tables + prefix trie from the dataset's sem_ids_list."""
    item_codes = [list(c) for c in ds.sem_ids_list]
    item_tokens = [offset_encode(c, codebook_size) for c in item_codes]

    lengths = {len(t) for t in item_tokens}
    assert len(lengths) == 1, f"non-uniform SID length: {lengths}"
    sem_id_len = lengths.pop()

    trie: Dict[Tuple[int, ...], set] = defaultdict(set)
    tokens_to_items: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for item_id, toks in enumerate(item_tokens):
        for depth in range(len(toks)):
            prefix = tuple(toks[:depth])
            trie[prefix].add(toks[depth])
        tokens_to_items[tuple(toks)].append(item_id)

    trie_sorted = {prefix: sorted(nxt) for prefix, nxt in trie.items()}
    return SemIdTables(
        item_codes=item_codes,
        item_tokens=item_tokens,
        trie=trie_sorted,
        tokens_to_items=dict(tokens_to_items),
        codebook_size=codebook_size,
        sem_id_len=sem_id_len,
    )


def summarize(ds: AmazonSeqDataset, pop: Popularity, tables: SemIdTables) -> str:
    """Human-readable sanity summary (printed by the CLI below)."""
    n = pop.num_items
    collisions = sum(1 for v in tables.tokens_to_items.values() if len(v) > 1)
    colliding_items = sum(len(v) for v in tables.tokens_to_items.values() if len(v) > 1)
    head_cnt = pop.counts[pop.head_ids]
    tail_cnt = pop.counts[pop.tail_ids]
    lines = [
        f"items={n}  users={len(ds.sequences)}  samples={len(ds.samples)}",
        f"SID token length={tables.sem_id_len}  codebook_size={tables.codebook_size}",
        f"SID collisions: {collisions} groups, {colliding_items} items collide "
        f"(add_disambiguation={ds.add_disambiguation})",
        f"head items={len(pop.head_ids)} (frac={pop.head_frac})  "
        f"tail items={len(pop.tail_ids)}",
        f"head interactions: total={head_cnt.sum()} mean={head_cnt.mean():.1f}  "
        f"tail interactions: total={tail_cnt.sum()} mean={tail_cnt.mean():.2f}",
        f"popularity concentration: head holds "
        f"{100*head_cnt.sum()/max(1,pop.counts.sum()):.1f}% of all interactions",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check popularity + SID trie for a split.")
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--head-frac", type=float, default=0.20)
    ap.add_argument("--rqvae-path", default=DEFAULT_RQVAE_PATH)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = ap.parse_args()

    ds = build_dataset(
        split=args.split,
        train_test_split="train",
        rqvae_path=args.rqvae_path,
        encoder_model_name=args.encoder,
    )
    pop = compute_popularity(ds, head_frac=args.head_frac)
    tables = build_sem_id_tables(ds)
    print(summarize(ds, pop, tables))
