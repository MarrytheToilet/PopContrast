"""Additional decode-time baselines (reviewer request), all post-hoc on cached scores.
Run from genrec/ root (CPU).

  rank-disc    : subtract z(popularity-rank percentile)          [nonparametric prior]
  eps-random   : replace ceil(eps*10) of top-10 with random items [exploration]
  calib-quota  : Steck-inspired calibrated rerank — per-user tail quota equal to the
                 tail share of the user's own history, filled from top-200 candidates
Alignment: eval samples rebuilt with the same seed as the cache; targets asserted equal.
Output: results/extra_baselines.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity

from popcontrast import RESULTS_DIR as RES
out = {}

for SPLIT in ["beauty", "sports", "toys"]:
    ds_tr = build_dataset(split=SPLIT, train_test_split="train")
    ds_te = build_dataset(split=SPLIT, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
    S = cache["scores"].float(); T = cache["targets"].numpy(); H = cache["seg_head"].numpy()
    I = S.shape[1]; N = S.shape[0]
    rng = np.random.default_rng(0)
    _ = rng.permutation(len(ds_tr.samples))       # match cache builder's rng call order
    idx = rng.permutation(len(ds_te.samples))[:N]
    samples = [ds_te.samples[i] for i in idx]
    assert np.array_equal(np.array([s["target"] for s in samples]), T), "cache alignment failed"

    counts = pop.counts[:I]
    tail_item = (pop.bucket[:I] == "tail")
    tail_user = ~H

    def metrics(top):
        hit = (top == T[:, None]).any(1)
        return {"R10": float(hit.mean()), "tailR10": float(hit[tail_user].mean()),
                "cov10": float(len(np.unique(top)) / I)}

    res = {}
    # 1) rank-percentile discount
    pct = np.argsort(np.argsort(counts)) / (I - 1)
    zr = torch.tensor((pct - pct.mean()) / pct.std(), dtype=torch.float32)
    for b in [0.5, 1.0]:
        top = (S - b * zr[None, :]).topk(10, dim=1).indices.numpy()
        res[f"rank-disc b={b}"] = metrics(top)

    # 2) epsilon-random exploration (3 of 10 slots random)
    g = np.random.default_rng(1)
    top7 = S.topk(7, dim=1).indices.numpy()
    rand3 = g.integers(0, I, size=(N, 3))
    res["eps-random 0.3"] = metrics(np.concatenate([top7, rand3], axis=1))

    # 3) calibrated quota rerank (Steck-inspired)
    top200 = S.topk(200, dim=1).indices.numpy()
    quota_top = np.empty((N, 10), dtype=np.int64)
    for u in range(N):
        hist = samples[u]["history"]
        tshare = np.mean([tail_item[i] for i in hist]) if hist else 0.8
        k_tail = int(round(10 * tshare))
        cands = top200[u]; is_t = tail_item[cands]
        tail_pick = cands[is_t][:k_tail]
        head_pick = cands[~is_t][:10 - len(tail_pick)]
        merged = np.concatenate([head_pick, tail_pick])
        if len(merged) < 10:   # top-200 short on one side: back-fill by score
            extra = [c for c in cands if c not in merged][:10 - len(merged)]
            merged = np.concatenate([merged, np.array(extra, dtype=np.int64)])
        quota_top[u] = merged[:10]
    res["calib-quota"] = metrics(quota_top)

    out[SPLIT] = res
    print(f"== {SPLIT} ==")
    for k, v in res.items():
        print(f"  {k:16s} R10={v['R10']:.4f} tail={v['tailR10']:.4f} cov={v['cov10']:.4f}")

with open(os.path.join(RES, "extra_baselines.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/extra_baselines.json")
