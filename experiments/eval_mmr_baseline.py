"""MMR (maximal marginal relevance) re-ranking baseline — the standard intra-list
diversification method. Run from genrec/ root (CPU-friendly; uses cached scores).

For each user: take top-200 candidates by cached score, greedily build top-10 by
    argmax_i  lam * z(score_i) - (1-lam) * max_{j in S} cos(e_i, e_j)
with sentence-t5 item embeddings e. Answers: "is PopContrast just diversifying?"
(expectation: MMR raises within-list diversity/coverage a bit but does not target
popularity, so tail recall stays ~flat).
Output: results/mmr_baseline.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity

from popcontrast import RESULTS_DIR as RES
LAMS = [0.9, 0.7]
TOPC = 200

out = {}
for SPLIT in ["beauty", "sports", "toys"]:
    ds = build_dataset(split=SPLIT, train_test_split="train")
    pop = compute_popularity(ds)
    emb = torch.nn.functional.normalize(ds.item_embeddings.float(), dim=1)  # (I,768)
    cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
    S = cache["scores"].float(); T = cache["targets"].numpy(); H = cache["seg_head"].numpy()
    I = S.shape[1]; N = S.shape[0]
    tail_user = ~H

    res = {}
    top_c = S.topk(TOPC, dim=1)
    cand_idx = top_c.indices                        # (N,200)
    cand_sc = top_c.values
    cand_sc = (cand_sc - cand_sc.mean(dim=1, keepdim=True)) / cand_sc.std(dim=1, keepdim=True).clamp_min(1e-6)

    for lam in LAMS:
        tops = np.empty((N, 10), dtype=np.int64)
        for u in range(N):
            idx = cand_idx[u]; sc = cand_sc[u]
            E = emb[idx]                            # (200,768)
            sim = E @ E.T                           # (200,200)
            chosen = []
            avail = torch.ones(TOPC, dtype=torch.bool)
            for _ in range(10):
                if chosen:
                    pen = sim[:, chosen].max(dim=1).values
                else:
                    pen = torch.zeros(TOPC)
                mmr = lam * sc - (1 - lam) * pen
                mmr[~avail] = -1e9
                j = int(mmr.argmax())
                chosen.append(j); avail[j] = False
            tops[u] = idx[chosen].numpy()
        hit = (tops == T[:, None]).any(1)
        res[f"lam={lam}"] = {"R10": float(hit.mean()),
                             "tailR10": float(hit[tail_user].mean()),
                             "cov10": float(len(np.unique(tops)) / I)}
        print(f"[{SPLIT}] MMR lam={lam}: {res[f'lam={lam}']}", flush=True)
    out[SPLIT] = res

with open(os.path.join(RES, "mmr_baseline.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/mmr_baseline.json", flush=True)
