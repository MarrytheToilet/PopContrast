"""THE reviewer-critical experiment: does PopContrast work inside REAL trie-beam
decoding, or are promoted tail items pruned before the leaf? Run from genrec/ root.

For each user: run trie-beam (width B) under RAW log p; apply the marginal penalty
to the B complete candidates (leaf-level, as deployed); take top-10. Compare against
the EXACT corrected top-10 (full-catalog scores from cache). Report per (B, beta):
  overlap@10 (beam-corrected vs exact-corrected), R@10, tail R@10, Cov@10,
  and needed-candidate coverage: fraction of exact-corrected top-10 present in the
  raw beam's candidate set (the quantity Prop 5.2 conditions on).
Output: results/beam_corrected_<split>.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history
from popcontrast.decoding import trie_beam_search

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
SPLIT = os.environ.get("BC_SPLIT", "beauty")
N = int(os.environ.get("BC_N", 1000))
BEAMS = [10, 20, 50, 100]
BETAS = [0.75, 1.5]

ds_tr = build_dataset(split=SPLIT, train_test_split="train")
ds_te = build_dataset(split=SPLIT, train_test_split="test", max_seq_len=50)
pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
model = load_tiger(f"out/tiger/amazon/{SPLIT}/best_model.pt", device=DEVICE)

cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
S = cache["scores"].float(); T = cache["targets"].numpy(); H = cache["seg_head"].numpy()
M = cache["marginal"].float(); I = S.shape[1]
z = (M - M.mean()) / M.std()
rng = np.random.default_rng(0)
_ = rng.permutation(len(ds_tr.samples))
idx = rng.permutation(len(ds_te.samples))[:S.shape[0]]
samples = [ds_te.samples[i] for i in idx][:N]
T = T[:N]; tail_user = ~H[:N]
zt = z.to(DEVICE)

# exact corrected top-10 per beta (from cached full-catalog scores)
exact_top = {b: (S[:N] - b * z[None, :]).topk(10, dim=1).indices.numpy() for b in BETAS}

out = {"split": SPLIT, "n": N, "beams": BEAMS, "betas": BETAS, "results": {}}
agg = {(B, b): {"ov": [], "hit": [], "cov": set(), "need": []} for B in BEAMS for b in BETAS}

with torch.no_grad():
    for u, s in enumerate(samples):
        enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
        for B in BEAMS:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                items, scores, _ = trie_beam_search(model, enc, attn, tables, num_beams=B, device=DEVICE)
            items = np.array(items); raw = torch.tensor(scores, device=DEVICE, dtype=torch.float32)
            for b in BETAS:
                corr = raw - b * zt[torch.tensor(items, device=DEVICE)]
                order = torch.argsort(corr, descending=True).cpu().numpy()
                top10 = items[order[:10]]
                et = exact_top[b][u]
                a = agg[(B, b)]
                a["ov"].append(len(set(top10.tolist()) & set(et.tolist())) / 10.0)
                a["need"].append(len(set(et.tolist()) & set(items.tolist())) / 10.0)
                a["hit"].append(int(T[u] in top10))
                a["cov"].update(top10.tolist())
        if (u + 1) % 200 == 0:
            print(f"  {u+1}/{N}", flush=True)

for (B, b), a in agg.items():
    hit = np.array(a["hit"])
    out["results"][f"B{B}_beta{b}"] = {
        "overlap10_vs_exact": float(np.mean(a["ov"])),
        "needed_candidates_in_beam": float(np.mean(a["need"])),
        "R10": float(hit.mean()),
        "tailR10": float(hit[tail_user].mean()),
        "cov10": float(len(a["cov"]) / I),
    }
# exact-corrected references on the same users
for b in BETAS:
    et = exact_top[b]
    hit = (et == T[:, None]).any(1)
    out["results"][f"exact_beta{b}"] = {
        "R10": float(hit.mean()), "tailR10": float(hit[tail_user].mean()),
        "cov10": float(len(np.unique(et)) / I)}

with open(os.path.join(RES, f"beam_corrected_{SPLIT}.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["results"], indent=2), flush=True)
