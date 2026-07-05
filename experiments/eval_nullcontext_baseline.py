"""Null-context prior baseline (CAD-style): estimate the popularity prior from a
single EMPTY/minimal context instead of averaging over sampled histories.
    prior_null(i) = log p_theta(i | null)
Then rank by  s = log p(i|u) - beta * z(prior_null)  using cached user scores.
This isolates the value of *averaging over real histories* (our m-hat) vs. the
one-shot null-context prior used by context-aware decoding. Minutes on GPU.
Output: results/nullcontext_baseline.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables, MAX_HISTORY_TOKENS
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import score_all_items, _item_tokens_tensor

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
BETAS = [0.25, 0.5, 0.75, 1.0]

out = {}
for SPLIT in ["beauty", "sports", "toys"]:
    ds_tr = build_dataset(split=SPLIT, train_test_split="train")
    pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
    model = load_tiger(f"out/tiger/amazon/{SPLIT}/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]

    # minimal null context: a single attended PAD position (fully-empty masks are degenerate)
    ids = torch.zeros(1, MAX_HISTORY_TOKENS, dtype=torch.long, device=DEVICE)
    attn = torch.zeros_like(ids); attn[0, -1] = 1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = model.model.encoder(input_ids=ids, attention_mask=attn)[0]
        prior = score_all_items(model, enc, attn, item_tok).float().cpu()

    logpop = np.log1p(pop.counts[:I])
    rho, _ = spearmanr(prior.numpy(), logpop)
    cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
    S = cache["scores"].float(); T = cache["targets"].numpy(); Hh = cache["seg_head"].numpy()
    m = cache["marginal"].float()
    rho_m, _ = spearmanr(prior.numpy(), m.numpy())
    z = (prior - prior.mean()) / prior.std()
    tail = ~Hh
    res = {"spearman_pop": float(rho), "spearman_vs_avg_marginal": float(rho_m)}
    for b in BETAS:
        top = (S - b * z[None, :]).topk(10, dim=1).indices.numpy()
        hit = (top == T[:, None]).any(1)
        res[f"b={b}"] = {"R10": float(hit.mean()), "tailR10": float(hit[tail].mean()),
                         "cov10": float(len(np.unique(top)) / I)}
    out[SPLIT] = res
    print(f"[{SPLIT}] rho(pop)={rho:.3f} rho(vs m-hat)={rho_m:.3f}", flush=True)
    for b in BETAS:
        print(f"   b={b}: {res[f'b={b}']}", flush=True)

with open(os.path.join(RES, "nullcontext_baseline.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved", flush=True)
