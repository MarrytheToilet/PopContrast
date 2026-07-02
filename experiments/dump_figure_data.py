"""Dump per-item quantities needed for data-driven figures (run from genrec/ root).

For each split with a cache: saves results/figdata_<split>.npz with
  log_pop   : (I,) log(1+interaction count)
  marginal  : (I,) model marginal score (from cache)
  cnt_base  : (I,) how many eval users got item in top-10 at baseline
  cnt_beta  : (I,) same, under model-PMI with beta=BETA
  tail_mask : (I,) bool, item is tail
"""
import glob, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables

RES = "/home/hanyu/research/PopSteer/results"
BETA = 1.0


def zscore(t):
    return (t - t.mean()) / t.std().clamp_min(1e-6)


for cache_path in sorted(glob.glob(os.path.join(RES, "cache_scores_*.pt"))):
    split = os.path.basename(cache_path)[len("cache_scores_"):-len(".pt")]
    print(f"[{split}] loading cache + popularity", flush=True)
    cache = torch.load(cache_path)
    scores = cache["scores"]                 # (N, I) cpu
    marginal = cache["marginal"]             # (I,)
    ds = build_dataset(split=split, train_test_split="train")
    pop = compute_popularity(ds)
    tables = build_sem_id_tables(ds)
    I = scores.shape[1]
    log_pop = np.log1p(pop.counts[:I])
    tail_mask = (pop.bucket[:I] == "tail")

    prior = zscore(marginal)
    base_top = scores.topk(10, dim=1).indices.reshape(-1)
    beta_top = (scores - BETA * prior[None, :]).topk(10, dim=1).indices.reshape(-1)
    cnt_base = np.bincount(base_top.numpy(), minlength=I)
    cnt_beta = np.bincount(beta_top.numpy(), minlength=I)

    np.savez(os.path.join(RES, f"figdata_{split}.npz"),
             log_pop=log_pop, marginal=marginal.numpy(),
             cnt_base=cnt_base, cnt_beta=cnt_beta, tail_mask=tail_mask, beta=BETA)
    print(f"  saved figdata_{split}.npz (I={I})", flush=True)
