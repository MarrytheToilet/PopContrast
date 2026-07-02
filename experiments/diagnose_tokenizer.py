"""Tokenizer root-cause diagnostic (LETTER direction): does the RQ-VAE codebook
bake in popularity BEFORE the decoder ever runs? Run from genrec/ root.

We test a causal chain:
    item popularity  --(tokenizer)-->  code "density"  --(decoder)-->  marginal prob
Already known: popularity <-> marginal (Spearman ~0.71). Here we add the two upstream
links to locate where the bias originates.

Metrics per split (saved to results/tokenizer_diag_<split>.json):
  - code density d(item) = mean over its L codes of log(#items sharing that code at that pos)
  - Spearman(popularity, code-density)      : do popular items get denser codes?
  - Spearman(code-density, marginal)         : does density drive the decoder prior?
  - partial: Spearman(pop, marginal | density) via residuals (is the bias fully mediated?)
  - collision bias: are SID-colliding items more popular than non-colliding?
  - codebook usage Gini per position (utilization skew)
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables

RES = "/home/hanyu/research/PopSteer/results"
SPLITS = os.environ.get("TOK_SPLITS", "beauty,sports,toys").split(",")


def gini(x):
    x = np.sort(x.astype(float)); n = len(x)
    return float((2*np.arange(1, n+1)-n-1).dot(x)/(n*x.sum())) if x.sum() > 0 else 0.0


def resid(y, x):
    # residual of y after linear-regressing on x (for partial correlation via ranks)
    ry = np.argsort(np.argsort(y)).astype(float); rx = np.argsort(np.argsort(x)).astype(float)
    b = np.polyfit(rx, ry, 1)
    return ry - np.polyval(b, rx)


def run(split):
    ds = build_dataset(split=split, train_test_split="train")
    pop = compute_popularity(ds); tables = build_sem_id_tables(ds)
    codes = np.array(ds.sem_ids_list)              # (I, L) raw codes
    I, L = codes.shape
    logpop = np.log1p(pop.counts[:I])

    # per-(position, code) usage frequency
    dens = np.zeros(I)
    per_pos_gini = []
    for j in range(L):
        vals, cnts = np.unique(codes[:, j], return_counts=True)
        freq = dict(zip(vals.tolist(), cnts.tolist()))
        dens += np.array([np.log(freq[c]) for c in codes[:, j]])
        per_pos_gini.append(gini(cnts))
    dens /= L                                       # mean log code-frequency = "density"

    rho_pop_dens, _ = spearmanr(logpop, dens)
    marg = None
    cache = os.path.join(RES, f"cache_scores_{split}.pt")
    out = {"items": I, "L": L,
           "spearman_pop_density": float(rho_pop_dens),
           "codebook_gini_per_pos": [float(g) for g in per_pos_gini]}
    if os.path.exists(cache):
        marg = torch.load(cache)["marginal"].numpy()[:I]
        rho_dens_marg, _ = spearmanr(dens, marg)
        rho_pop_marg, _ = spearmanr(logpop, marg)
        # partial Spearman(pop, marginal | density): correlate residuals after removing density
        pr, _ = spearmanr(resid(logpop, dens), resid(marg, dens))
        out.update({"spearman_density_marginal": float(rho_dens_marg),
                    "spearman_pop_marginal": float(rho_pop_marg),
                    "partial_pop_marginal_given_density": float(pr)})

    # collision bias: colliding items vs non-colliding, mean log-pop
    tok = [tuple(t) for t in tables.item_tokens]
    from collections import Counter
    cnt = Counter(tok)
    colliding = np.array([cnt[t] > 1 for t in tok])
    out["n_colliding_items"] = int(colliding.sum())
    if colliding.any() and (~colliding).any():
        out["logpop_colliding_mean"] = float(logpop[colliding].mean())
        out["logpop_noncolliding_mean"] = float(logpop[~colliding].mean())
        # rank-biserial: are collisions concentrated among popular items?
        rho_col, _ = spearmanr(logpop, colliding.astype(float))
        out["spearman_pop_iscolliding"] = float(rho_col)

    # save per-item arrays for figures
    np.savez(os.path.join(RES, f"tokdata_{split}.npz"),
             logpop=logpop, density=dens, marginal=(marg if marg is not None else np.zeros(I)))
    return out


if __name__ == "__main__":
    allres = {}
    for sp in SPLITS:
        print(f"=== {sp} ===", flush=True)
        r = run(sp); allres[sp] = r
        for k, v in r.items():
            print(f"  {k}: {v}", flush=True)
    with open(os.path.join(RES, "tokenizer_diag.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print("saved results/tokenizer_diag.json", flush=True)
