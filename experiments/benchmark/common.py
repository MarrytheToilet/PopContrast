"""Shared, deterministic scoring and metadata for matched recommendation benchmarks."""
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed):
    # Length grouping varies sequence shapes; expandable CUDA segments avoid
    # retaining disjoint large blocks for every padding length on shared GPUs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_model(checkpoint=None, sid_length=3, device="cuda"):
    # Load the exact original implementation without genrec's optional imports.
    module_name = "popcontrast_t5_backend"
    if module_name not in sys.modules:
        path = ROOT / "genrec/genrec/modules/t5.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[module_name]
    cfg = mod.T5Config(num_layers=4, num_decoder_layers=4, d_model=128,
                       d_ff=1024, num_heads=6, d_kv=64, dropout_rate=0.1,
                       vocab_size=256 * sid_length + 1, pad_token_id=0,
                       eos_token_id=0, decoder_start_token_id=0,
                       feed_forward_proj="relu")
    model = mod.T5ForConditionalGeneration(cfg)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        if all(key.startswith("model.") for key in state):
            state = {key[6:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(device)


def load_bundle(split, directory=None, sid_length=3):
    path = Path(directory or ROOT / "results/benchmark/data") / f"{split}.npz"
    with np.load(path) as packed:
        data = {key: packed[key] for key in packed.files}
    if sid_length in [4, 5]:
        # A genuine extra disambiguation position, not padding an existing ID.
        seen = {}
        suffixes = []
        for row in data["item_tokens"]:
            key = tuple(row)
            suffix = seen.get(key, 0)
            if suffix >= 256 ** (sid_length - 3):
                raise ValueError("Disambiguation exceeds the added codebooks")
            suffixes.append([769 + suffix] if sid_length == 4 else [769 + suffix // 256, 1025 + suffix % 256])
            seen[key] = suffix + 1
        data["item_tokens"] = np.column_stack([data["item_tokens"], np.asarray(suffixes)])
    elif sid_length == 6:
        # Same three 8-bit RQ codes, encoded as six 4-bit digits. No new item
        # semantics or padded fake positions; longer autoregressive factorization.
        codes = np.asarray(data["item_codes"])
        digits = np.stack([codes // 16, codes % 16], axis=2).reshape(len(codes), 6)
        data["item_tokens"] = digits + np.arange(6) * 256 + 1
    elif sid_length != 3:
        raise ValueError("Supported IDs: 3 RQ, 4/5 disambiguated, 6-digit reencoding")
    return data


def tokens_for_histories(histories, item_tokens, device="cpu"):
    histories = torch.as_tensor(histories, dtype=torch.long, device=device)
    lookup = torch.as_tensor(item_tokens, dtype=torch.long, device=device)
    values = lookup[histories.clamp_min(0)]
    values = values.masked_fill((histories < 0).unsqueeze(-1), 0)
    values = values.flatten(1)
    return values, (values != 0).long()


def stable_topk(scores, k=10, ids=None):
    """Descending score, then ascending exact item ID; identical for every path."""
    scores = np.asarray(scores)
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids)
    if len(scores) > max(1024, 4 * k):
        # Partition first, then resolve every boundary tie by exact item ID.
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > threshold)
        equal = np.flatnonzero(scores == threshold)
        equal = equal[np.argsort(ids[equal], kind="stable")[:k - len(above)]]
        candidates = np.concatenate([above, equal])
        return ids[candidates[np.lexsort((ids[candidates], -scores[candidates]))[:k]]]
    return ids[np.lexsort((ids, -scores))[:k]]


def metrics(top, targets, head_mask, n_items):
    top, targets = np.asarray(top), np.asarray(targets)
    matches = top == targets[:, None]
    hit = matches.any(axis=1)
    ranks = matches.argmax(axis=1)
    ndcg = hit / np.log2(ranks + 2)
    head = np.asarray(head_mask)[targets]
    counts = np.bincount(top.ravel(), minlength=n_items)
    counts = counts[:n_items]
    positive = counts[counts > 0] / counts.sum()
    sorted_counts = np.sort(counts)
    return {"n_users": len(targets), "hits": int(hit.sum()), "R10": float(hit.mean()),
            "N10": float(ndcg.mean()),
            "headR10": float(hit[head].mean()) if head.any() else None,
            "tailR10": float(hit[~head].mean()) if (~head).any() else None,
            "head_users": int(head.sum()), "tail_users": int((~head).sum()),
            "cov10": float(np.count_nonzero(counts) / n_items),
            "entropy": float(-(positive * np.log(positive)).sum() / np.log(n_items)),
            "gini": float(np.dot(2 * np.arange(1, n_items + 1) - n_items - 1, sorted_counts) / (n_items * counts.sum()))}, hit


def paired_ci(base_hit, new_hit, seed=20260927, repetitions=2000):
    delta = np.asarray(new_hit, dtype=float) - np.asarray(base_hit, dtype=float)
    if len(delta) == 0:
        return {"delta": None, "ci95": [None, None], "n": 0}
    # Resampling three outcome categories is exactly the paired-user bootstrap
    # for a mean of {-1, 0, 1}, without allocating repetitions x users indices.
    prob = np.array([(delta == x).mean() for x in [-1, 0, 1]])
    draws = np.random.default_rng(seed).multinomial(len(delta), prob, size=repetitions)
    values = (draws[:, 2] - draws[:, 0]) / len(delta)
    return {"delta": float(delta.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "gained_hits": int((delta == 1).sum()), "lost_hits": int((delta == -1).sum()),
            "n": len(delta), "resamples": repetitions}
