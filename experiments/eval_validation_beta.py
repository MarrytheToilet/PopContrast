"""Validation-selected beta (reviewer: avoid test-set operating-point selection).
Run from genrec/ root. For each dataset: score n=2000 VALIDATION users exactly,
sweep beta on validation, apply the fixed rule (largest beta with overall R@10 >=
baseline), report the selected beta. Test-set one-shot numbers at that beta already
exist in results/main_panel_<split>.json. Output: results/validation_beta.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
N = int(os.environ.get("VB_N", 2000))
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

out = {}
for SPLIT in ["beauty", "sports", "toys"]:
    ds_tr = build_dataset(split=SPLIT, train_test_split="train")
    ds_va = build_dataset(split=SPLIT, train_test_split="valid", max_seq_len=50)
    pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
    model = load_tiger(f"out/tiger/amazon/{SPLIT}/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]
    m = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))["marginal"].float().to(DEVICE)
    z = (m - m.mean()) / m.std()

    rng = np.random.default_rng(7)
    samples = [ds_va.samples[i] for i in rng.permutation(len(ds_va.samples))[:N]]
    T = np.array([s["target"] for s in samples])
    tail = np.array([pop.bucket[s["target"]] == "tail" for s in samples])

    S = torch.empty(len(samples), I)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, s in enumerate(samples):
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            S[i] = score_all_items(model, enc, attn, item_tok).cpu()
            if (i + 1) % 500 == 0: print(f"  [{SPLIT}] {i+1}/{len(samples)}", flush=True)

    res = {}
    zc = z.cpu()
    for b in BETAS:
        top = (S - b * zc[None, :]).topk(10, dim=1).indices.numpy()
        hit = (top == T[:, None]).any(1)
        res[b] = {"R10": float(hit.mean()), "tailR10": float(hit[tail].mean()),
                  "cov10": float(len(np.unique(top)) / I)}
    base = res[0.0]["R10"]
    sel = max([b for b in BETAS if res[b]["R10"] >= base])
    out[SPLIT] = {"validation_sweep": {str(k): v for k, v in res.items()},
                  "selected_beta": sel}
    print(f"[{SPLIT}] validation-selected beta = {sel}  (sweep: "
          + ", ".join(f"b={b}:R10={res[b]['R10']:.4f}" for b in BETAS) + ")", flush=True)

with open(os.path.join(RES, "validation_beta.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/validation_beta.json", flush=True)
