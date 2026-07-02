"""Fast steering-method sweep: many (direction x operator x gate x alpha) configs
in one run, sharing model / hidden-state collection / eval set.

Directions supported (per layer, step):
  caa   : mean(H_head) - mean(H_tail)                    (contrastive activation addition)
  probe : logistic-regression weight vector (raw feats)  (max-separating linear axis)
  repe  : top PC of class-mean-centered activations      (representation reading)
Operators:
  steer : h~ = ||h|| normalize(h/||h|| - alpha_eff v)    (fixed-vector subtraction)
  ablate: h~ = h - alpha_eff (h.v) v                      (project OUT the component)
Gate: tau=None -> static alpha;  else alpha_eff = alpha * sigmoid(<h,v> - tau).

Edit CONFIGS at the bottom (or pass --preset). Prints head/tail/coverage per config.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger, num_decoder_layers
from popcontrast.hidden_states import collect_hidden
from popcontrast.oracle import evaluate_exact


def compute_directions(hc, kinds=("caa", "probe", "repe"), seed=0):
    """dirs[kind][(layer, step)] -> unit np.ndarray (d_model,) in activation space."""
    rng = np.random.default_rng(seed)
    dirs = {k: {} for k in kinds}
    for (l, j), X in hc.H.items():
        y = hc.y[(l, j)]
        H_head, H_tail = X[y == 1], X[y == 0]
        if "caa" in kinds:
            v = H_head.mean(0) - H_tail.mean(0)
            dirs["caa"][(l, j)] = _unit(v)
        if "probe" in kinds:
            # balance classes, fit on RAW features so the weight lives in activation space
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            m = min(len(pos), len(neg))
            idx = np.concatenate([rng.choice(pos, m, False), rng.choice(neg, m, False)])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[idx], y[idx])
            dirs["probe"][(l, j)] = _unit(clf.coef_[0])
        if "repe" in kinds:
            # RepE-style: PCA on class-mean-centered activations; top PC.
            Xc = np.concatenate([H_head - H_head.mean(0), H_tail - H_tail.mean(0)], 0)
            # top principal component via SVD
            _, _, Vt = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
            pc = Vt[0]
            # orient PC toward head-vs-tail mean difference
            if np.dot(pc, H_head.mean(0) - H_tail.mean(0)) < 0:
                pc = -pc
            dirs["repe"][(l, j)] = _unit(pc)
    return dirs


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--checkpoint", default="out/tiger/amazon/beauty/best_model.pt")
    ap.add_argument("--collect-n", type=int, default=20000)
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.collect_n, args.eval_n = 1000, 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_tr = build_dataset(split=args.split, train_test_split="train")
    ds_te = build_dataset(split=args.split, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    tables = build_sem_id_tables(ds_tr)
    model = load_tiger(args.checkpoint, device=device)
    L = num_decoder_layers(model)
    layers = list(range(L))

    rng = np.random.default_rng(args.seed)
    tr_idx = rng.permutation(len(ds_tr.samples))[:args.collect_n]
    te_idx = rng.permutation(len(ds_te.samples))[:args.eval_n]
    collect_samples = [ds_tr.samples[i] for i in tr_idx]
    eval_samples = [ds_te.samples[i] for i in te_idx]

    print(f"[collect] {args.collect_n} samples, layers {layers}")
    hc = collect_hidden(model, collect_samples, tables, pop, layers, batch_size=256, device=device)
    dirs = compute_directions(hc)

    def vpop(kind, layer, step):
        return torch.tensor(dirs[kind][(layer, step)], dtype=torch.float32, device=device)

    # ---- configs to compare (name, layer, step, dir_kind, mode, alpha, tau) ----
    CONFIGS = [
        ("baseline",             None, None, None,    None,          0.0, None),
        ("ablate-caa-L3",        3, 0, "caa",   "ablate",       1.0, None),
        ("ablate-probe-L3",      3, 0, "probe", "ablate",       1.0, None),
        ("ablate-repe-L3",       3, 0, "repe",  "ablate",       1.0, None),
        ("ablateClamp-probe-L3", 3, 0, "probe", "ablate_clamp", 1.0, None),
        ("ablate-probe-a2.0",    3, 0, "probe", "ablate",       2.0, None),
    ]
    if args.smoke:
        CONFIGS = CONFIGS[:3]

    print(f"\n[eval] {len(eval_samples)} users "
          f"(head={sum(1 for s in eval_samples if pop.bucket[s['target']]=='head')}, "
          f"tail={sum(1 for s in eval_samples if pop.bucket[s['target']]=='tail')})")
    print(f"{'config':22s} {'overall':>8} {'head':>8} {'tail':>8} {'cov':>8}")
    for name, layer, step, kind, mode, alpha, tau in CONFIGS:
        inj = None if kind is None else dict(
            layer=layer, v_pop=vpop(kind, layer, step), alpha=alpha, tau=tau, mode=mode)
        res = evaluate_exact(model, eval_samples, tables, pop, k_list=(10,),
                             inject=inj, device=device)
        r = res.overall[10]["recall"]; h = res.head[10]["recall"]
        t = res.tail[10]["recall"]; c = res.coverage[10]
        print(f"{name:22s} {r:8.4f} {h:8.4f} {t:8.4f} {c:8.4f}")


if __name__ == "__main__":
    main()
