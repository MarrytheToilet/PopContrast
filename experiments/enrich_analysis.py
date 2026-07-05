"""Cache-based enrichment analyses (CPU; run from genrec/ root).

1) Popularity-QUINTILE recall breakdown under beta sweep (finer than head/tail).
2) GROUP-wise decode-time baselines (reviewer request): coarsened corrections that
   subtract the *group mean* prior (model-marginal or log-count) per popularity
   quintile -> tests whether per-item granularity of the marginal matters.
3) RANK-SHIFT mechanism data: per-item mean rank change (beta=0.75 vs baseline)
   vs popularity -> novel diagnostic figure.
4) Tokenizer: per-code popularity entropy + popularity-mass concentration per position.

Outputs: results/enrich_analysis.json, results/rankshift_beauty.npz
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables

from popcontrast import RESULTS_DIR as RES
out = {}

for SPLIT in ["beauty", "sports", "toys"]:
    ds = build_dataset(split=SPLIT, train_test_split="train")
    pop = compute_popularity(ds)
    cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
    S = cache["scores"].float()          # (N,I) cpu
    T = cache["targets"].numpy()
    M = cache["marginal"].float()
    I = S.shape[1]
    counts = pop.counts[:I]
    logpop = np.log1p(counts)
    z = (M - M.mean()) / M.std()

    # quintiles by popularity rank (equal item counts; q4 = most popular)
    order = np.argsort(counts, kind="stable")
    qid = np.empty(I, dtype=int)
    for q in range(5):
        qid[order[q * I // 5:(q + 1) * I // 5]] = q
    tq = qid[T]

    def recall_by_q(prior, beta):
        sc = S if beta == 0 else S - beta * prior[None, :]
        top = sc.topk(10, dim=1).indices.numpy()
        hit = (top == T[:, None]).any(1)
        return [float(hit[tq == q].mean()) if (tq == q).any() else None for q in range(5)], float(hit.mean())

    res_q = {}
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        byq, overall = recall_by_q(z, b)
        res_q[b] = {"overall": overall, "by_quintile": byq}
    out.setdefault(SPLIT, {})["quintile_recall"] = res_q
    out[SPLIT]["quintile_item_counts"] = [int((qid == q).sum()) for q in range(5)]
    out[SPLIT]["quintile_target_counts"] = [int((tq == q).sum()) for q in range(5)]

    # group-wise coarsened baselines
    gm_marg = torch.tensor([float(M[qid == q].mean()) for q in range(5)])
    gm_cnt = torch.tensor([float(np.log1p(counts)[qid == q].mean()) for q in range(5)])
    prior_gm = gm_marg[qid]; prior_gm = (prior_gm - prior_gm.mean()) / prior_gm.std()
    prior_gc = gm_cnt[qid];  prior_gc = (prior_gc - prior_gc.mean()) / prior_gc.std()
    res_g = {}
    for name, pr in [("group_marginal", prior_gm), ("group_count", prior_gc)]:
        res_g[name] = {}
        for b in [0.25, 0.5, 0.75, 1.0]:
            sc = S - b * pr[None, :]
            top = sc.topk(10, dim=1).indices.numpy()
            hit = (top == T[:, None]).any(1)
            tail_mask = (tq <= 3)  # bottom 80% = tail (matches head=top-20%)
            cov = len(np.unique(top)) / I
            res_g[name][b] = {"R10": float(hit.mean()),
                              "tailR10": float(hit[tail_mask].mean()),
                              "cov10": float(cov)}
    out[SPLIT]["group_baselines"] = res_g

    # tokenizer: per-code popularity entropy + mass concentration (positions)
    codes = np.array(ds.sem_ids_list)[:I]
    tok = {}
    for j in range(codes.shape[1]):
        ents = []
        mass_share_top10pct_codes = None
        code_mass = {}
        for c in np.unique(codes[:, j]):
            m_ = counts[codes[:, j] == c].astype(float)
            code_mass[c] = m_.sum()
            if len(m_) > 1 and m_.sum() > 0:
                p = m_ / m_.sum()
                p = p[p > 0]
                ents.append(float(-(p * np.log(p)).sum() / np.log(len(m_))))
        cm = np.sort(np.array(list(code_mass.values())))[::-1]
        k = max(1, len(cm) // 10)
        tok[f"pos{j}"] = {"mean_within_code_pop_entropy": float(np.mean(ents)),
                          "top10pct_codes_mass_share": float(cm[:k].sum() / cm.sum())}
    out[SPLIT]["tokenizer_entropy"] = tok

    # rank-shift (beauty only, for the mechanism figure)
    if SPLIT == "beauty":
        r0 = torch.argsort(torch.argsort(-S, dim=1), dim=1).float()          # baseline rank per user
        r1 = torch.argsort(torch.argsort(-(S - 0.75 * z[None, :]), dim=1), dim=1).float()
        dmean = (r1 - r0).mean(0).numpy()                                     # per-item mean rank change
        np.savez(os.path.join(RES, "rankshift_beauty.npz"),
                 logpop=logpop, drank=dmean, marginal=M.numpy())
        out[SPLIT]["rankshift_saved"] = True
    print(f"[{SPLIT}] done", flush=True)

with open(os.path.join(RES, "enrich_analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/enrich_analysis.json", flush=True)
