"""Reviewer-response experiments (run from genrec/ root):
1) LEACE closed-form linear concept erasure at a decoder layer (cited-but-untested gap):
   fit eraser x' = x − Σ^{1/2} P Σ^{-1/2}(x−μ) with P the projector onto the whitened
   popularity cross-covariance; hook it; measure tail/overall R@10 vs baseline.
2) M-sensitivity of the marginal: marginals from M=32/128 histories; Spearman with
   popularity + β=0.75 eval via cached user scores (evaluation itself is free).
Outputs: results/rebuttal_checks.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from scipy.stats import spearmanr
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger, decoder_blocks
from popcontrast.hidden_states import collect_hidden
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor

RES = "/home/hanyu/research/PopSteer/results"
DEVICE = "cuda"
SPLIT = "beauty"
LAYER = 3
N_EVAL = 2000

ds_tr = build_dataset(split=SPLIT, train_test_split="train")
ds_te = build_dataset(split=SPLIT, train_test_split="test", max_seq_len=50)
pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
model = load_tiger(f"out/tiger/amazon/{SPLIT}/best_model.pt", device=DEVICE)
item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]
logpop = np.log1p(pop.counts[:I])
rng = np.random.default_rng(0)
out = {}

# ---------- 1) LEACE ----------
print("[leace] collecting hidden states", flush=True)
coll = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:15000]]
hc = collect_hidden(model, coll, tables, pop, [LAYER], batch_size=256, device=DEVICE)
X = np.concatenate([hc.H[(LAYER, j)] for j in range(tables.sem_id_len)], 0)
y = np.concatenate([hc.y[(LAYER, j)] for j in range(tables.sem_id_len)], 0).astype(np.float64)
mu = X.mean(0); Xc = X - mu; zc = y - y.mean()
Sig = (Xc.T @ Xc) / len(X) + 1e-4 * np.eye(X.shape[1])
evals, evecs = np.linalg.eigh(Sig)
W = evecs @ np.diag(evals**-0.5) @ evecs.T          # Σ^{-1/2}
Wp = evecs @ np.diag(evals**0.5) @ evecs.T          # Σ^{1/2}
sxz = Xc.T @ zc / len(X)                            # (d,)
u = W @ sxz; u = u / np.linalg.norm(u)
A = Wp @ np.outer(u, u) @ W                         # eraser matrix
A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)

block = decoder_blocks(model)[LAYER]
def leace_hook(_m, _i, output):
    h = output[0]
    h2 = h - (h - mu_t) @ A_t.T
    return (h2,) + tuple(output[1:])

eval_s = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:N_EVAL]]
tails = np.array([pop.bucket[s["target"]] == "tail" for s in eval_s])

def eval_recall(hook=None):
    handle = block.register_forward_hook(hook) if hook else None
    hits = []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for s in eval_s:
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            sc = score_all_items(model, enc, attn, item_tok)
            hits.append(int(s["target"] in sc.topk(10).indices.cpu().tolist()))
    if handle: handle.remove()
    h = np.array(hits)
    return float(h.mean()), float(h[tails].mean())

print("[leace] eval baseline vs erased", flush=True)
b_all, b_tail = eval_recall(None)
l_all, l_tail = eval_recall(leace_hook)
out["leace"] = {"layer": LAYER, "n": N_EVAL,
                "baseline": {"R10": b_all, "tailR10": b_tail},
                "leace":    {"R10": l_all, "tailR10": l_tail}}
print(out["leace"], flush=True)

# ---------- 2) M-sensitivity ----------
print("[msens] computing marginals at M=32,128", flush=True)
marg_hist = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:512]]
cache = torch.load(os.path.join(RES, f"cache_scores_{SPLIT}.pt"))
S = cache["scores"]; T = cache["targets"].numpy(); H = cache["seg_head"].numpy()
m512 = cache["marginal"]
tail_c = ~H
res_m = {}
for M in [32, 128, 512]:
    if M == 512:
        m = m512
    else:
        acc = torch.zeros(I, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            for s in marg_hist[:M]:
                enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
                acc += score_all_items(model, enc, attn, item_tok)
        m = (acc / M).cpu()
    rho, _ = spearmanr(m.numpy(), logpop)
    z = (m - m.mean()) / m.std()
    top = (S - 0.75 * z[None, :]).topk(10, dim=1).indices.numpy()
    hit = (top == T[:, None]).any(1)
    cov = len(np.unique(top)) / I
    res_m[M] = {"spearman_pop": float(rho), "R10": float(hit.mean()),
                "tailR10": float(hit[tail_c].mean()), "cov10": float(cov)}
    print(f"  M={M}: {res_m[M]}", flush=True)
out["m_sensitivity"] = res_m

with open(os.path.join(RES, "rebuttal_checks.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved results/rebuttal_checks.json", flush=True)
