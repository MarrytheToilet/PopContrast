"""Per-step ADAPTIVE contrastive decoding (novelty: structured over the semantic-ID
hierarchy). Run from genrec/ root.

Score item by summing per-SID-step:  s = Σ_j [ c_j − β·g_j·m_j ]
  c_j = log p(t_j | h, t_<j)   (per-step conditional, this user)
  m_j = mean_h log p(t_j | h, t_<j)   (per-step marginal / popularity prior)
  g_j = σ(κ·(m_j − c_j))       gate: ~1 when the user's context did NOT make this
                                token more likely than average (a popularity default)
                                → subtract; ~0 when the user specifically predicts it
                                → protect the personalized signal.

Static global PMI is the β·Σ m_j special case (g_j≡1). Hypothesis: adaptive lifts
tail/coverage while holding overall (reproduces the strict-Pareto point robustly).

Compares baseline / static-PMI / adaptive on each split → results/adaptive_contrast.json.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items_perstep, _item_tokens_tensor

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
SPLITS = os.environ.get("ADA_SPLITS", "beauty,sports,toys").split(",")
N_EVAL = int(os.environ.get("ADA_N", 3000))
MARG_M = int(os.environ.get("ADA_M", 256))


def metrics(topk, targets, seg_head, n_items):
    hit = (topk == targets[:, None])
    hit10 = hit.any(1).float(); rank = torch.where(hit.any(1), hit.float().argmax(1), torch.tensor(-1, device=hit.device))
    ndcg = torch.where(hit.any(1), 1/torch.log2(rank.float()+2), torch.zeros_like(hit10))
    tail = ~seg_head
    cov = torch.unique(topk[:, :10]).numel() / n_items
    cnt = torch.bincount(topk[:, :10].reshape(-1), minlength=n_items).float().cpu().numpy()
    nz = cnt[cnt > 0]; p = nz/nz.sum(); ent = float(-(p*np.log(p)).sum()/np.log(n_items))
    seg = lambda v, m: float(v[m].mean()) if m.any() else 0.0
    return {"R10": float(hit10.mean()), "N10": float(ndcg.mean()),
            "headR10": seg(hit10, seg_head), "tailR10": seg(hit10, tail),
            "cov10": cov, "ent": ent}


def run_split(split):
    ds_tr = build_dataset(split=split, train_test_split="train")
    ds_te = build_dataset(split=split, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
    model = load_tiger(f"out/tiger/amazon/{split}/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]
    rng = np.random.default_rng(0)
    marg_s = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:MARG_M]]
    eval_s = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:N_EVAL]]

    # per-step marginal (I,L)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m = torch.zeros(I, tables.sem_id_len, device=DEVICE)
        for s in marg_s:
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            m += score_all_items_perstep(model, enc, attn, item_tok)
        m /= MARG_M

    targets = torch.tensor([s["target"] for s in eval_s], device=DEVICE)
    seg_head = torch.tensor([pop.bucket[s["target"]] == "head" for s in eval_s], device=DEVICE)

    # cache per-user per-step conditionals (N,I,L) is big; instead score per user and
    # evaluate all methods on the fly, storing only topk per method.
    configs = [("baseline", None), ("static-b0.5", ("static", 0.5, None)),
               ("static-b1.0", ("static", 1.0, None)),
               ("adapt-b1.0-k3", ("adapt", 1.0, 3.0)), ("adapt-b1.5-k3", ("adapt", 1.5, 3.0)),
               ("adapt-b2.0-k5", ("adapt", 2.0, 5.0))]
    tops = {name: torch.empty(len(eval_s), 10, dtype=torch.long) for name, _ in configs}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, s in enumerate(eval_s):
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            c = score_all_items_perstep(model, enc, attn, item_tok)   # (I,L)
            for name, cfg in configs:
                if cfg is None:
                    sc = c.sum(1)
                else:
                    kind, beta, kappa = cfg
                    if kind == "static":
                        sc = (c - beta * m).sum(1)
                    else:
                        g = torch.sigmoid(kappa * (m - c))
                        sc = (c - beta * g * m).sum(1)
                tops[name][i] = torch.topk(sc, 10).indices.cpu()
    out = {}
    for name, _ in configs:
        out[name] = metrics(tops[name].to(DEVICE), targets, seg_head, I)
    out["_meta"] = {"n": len(eval_s), "items": I,
                    "head": int(seg_head.sum()), "tail": int((~seg_head).sum())}
    return out


if __name__ == "__main__":
    allres = {}
    for sp in SPLITS:
        print(f"=== {sp} ===", flush=True)
        allres[sp] = run_split(sp)
        for name, mtr in allres[sp].items():
            if name == "_meta": continue
            print(f"  {name:16s} R10={mtr['R10']:.4f} head={mtr['headR10']:.4f} "
                  f"tail={mtr['tailR10']:.4f} cov={mtr['cov10']:.4f} ent={mtr['ent']:.3f}", flush=True)
    with open(os.path.join(RES, "adaptive_contrast.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print("saved results/adaptive_contrast.json", flush=True)
