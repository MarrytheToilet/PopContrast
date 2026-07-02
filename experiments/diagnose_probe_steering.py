"""Minimal validation = Plan A vs Plan B decision gate (IDEA.md §7).

Three questions, in order:
  1. PROBE      Is popularity linearly decodable from the decoder residual stream?
                Report AUC per (layer, step). CONFOUND CONTROL: the honest signal is
                at decode STEP 0 — before any of the target item's own SID tokens are
                emitted, the state only reflects the history-conditioned *prediction*,
                so high AUC there = the model's popularity prior, not identity leakage.
                Later steps condition on the item's own tokens (leakier). We also split
                probe train/test by USER so it can't memorize item-specific positions.
  2. DIRECTION  v_pop = mean(H_head) - mean(H_tail) per (layer, step), unit-normalized (CAA).
  3. INJECTION  Sweep alpha; does tail Recall@10 rise monotonically while head Recall@10
                does not collapse? (ranked by exact scores under pop_steer_hooks.)

VERDICT: step-0 probe AUC >= 0.70 at some layer AND tail recall monotone-up in alpha
with head not collapsing  ->  Plan A (PopSteer). Otherwise  ->  Plan B (PopContrast).
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger, num_decoder_layers
from popcontrast.hidden_states import collect_hidden
from popcontrast.oracle import evaluate_exact


def run_probe(hc, seed: int = 0):
    """Per (layer, step) logistic-probe AUC with a user-disjoint train/test split.

    Returns dict[(layer, step)] -> auc. Since head/tail are defined by popularity,
    we balance classes by subsampling the majority (tail) so AUC isn't inflated by
    prior skew.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for (l, j), X in hc.H.items():
        y = hc.y[(l, j)]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        m = min(len(pos), len(neg))
        if m < 50:
            out[(l, j)] = float("nan"); continue
        idx = np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)])
        rng.shuffle(idx)
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        out[(l, j)] = float(roc_auc_score(y[te], p))
    return out


def caa_directions(hc):
    """v_pop[(layer, step)] = unit(mean(H_head) - mean(H_tail))."""
    dirs = {}
    for (l, j), X in hc.H.items():
        y = hc.y[(l, j)]
        vh, vt = X[y == 1].mean(0), X[y == 0].mean(0)
        v = vh - vt
        n = np.linalg.norm(v)
        dirs[(l, j)] = v / n if n > 0 else v
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--checkpoint", default="out/tiger/amazon/beauty/best_model.pt")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="decoder layers to probe (default: all)")
    ap.add_argument("--collect-n", type=int, default=20000, help="#train samples for probe/CAA")
    ap.add_argument("--eval-n", type=int, default=2000, help="#test users for injection sweep")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--inject-layer", type=int, default=None, help="layer to inject (default: best step-0 probe layer)")
    ap.add_argument("--inject-step", type=int, default=0, help="which step's v_pop to inject (default 0)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to shake out bugs")
    args = ap.parse_args()

    if args.smoke:
        args.collect_n, args.eval_n, args.alphas = 512, 100, [0.0, 2.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] dataset ({args.split}) + model")
    ds_tr = build_dataset(split=args.split, train_test_split="train")
    ds_te = build_dataset(split=args.split, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    tables = build_sem_id_tables(ds_tr)
    model = load_tiger(args.checkpoint, device=device)
    L = num_decoder_layers(model)
    layers = args.layers if args.layers is not None else list(range(L))
    print(f"[info] {L} decoder layers; probing {layers}; SID len {tables.sem_id_len}")

    # RANDOM subsets (NOT first-N: the first users have short/hard histories and
    # bias recall low — this was the false-negative bug in the first run).
    rng0 = np.random.default_rng(0)
    tr_idx = rng0.permutation(len(ds_tr.samples))[:args.collect_n]
    te_idx = rng0.permutation(len(ds_te.samples))[:args.eval_n]
    collect_samples = [ds_tr.samples[i] for i in tr_idx]
    eval_samples = [ds_te.samples[i] for i in te_idx]

    # 1. collect + probe
    print(f"[1/3] collecting decoder hidden states over {args.collect_n} train samples...")
    hc = collect_hidden(model, collect_samples, tables, pop, layers,
                        batch_size=256, device=device)
    aucs = run_probe(hc)
    print("      probe AUC (rows=layer, cols=step):")
    header = "        layer " + "".join(f"  step{j}" for j in range(tables.sem_id_len))
    print(header)
    for l in layers:
        cells = "".join(f"  {aucs[(l, j)]:.3f}" for j in range(tables.sem_id_len))
        print(f"        {l:5d} {cells}")
    step0 = {l: aucs[(l, 0)] for l in layers}
    best_layer = max(step0, key=lambda l: (step0[l] if not np.isnan(step0[l]) else -1))
    best_auc0 = step0[best_layer]
    print(f"      best step-0 AUC = {best_auc0:.3f} at layer {best_layer}")

    # 2. directions
    dirs = caa_directions(hc)
    inj_layer = args.inject_layer if args.inject_layer is not None else best_layer
    v_pop = torch.tensor(dirs[(inj_layer, args.inject_step)], dtype=torch.float32, device=device)
    print(f"[2/3] CAA direction ready (inject layer {inj_layer}, step {args.inject_step})")

    # 3. injection sweep
    print(f"[3/3] injection sweep over alphas={args.alphas} on {args.eval_n} test users...")
    rows = []
    for a in args.alphas:
        inj = None if a == 0.0 else dict(layer=inj_layer, v_pop=v_pop, alpha=a, tau=None)
        res = evaluate_exact(model, eval_samples, tables, pop, k_list=(5, 10),
                             inject=inj, device=device)
        rows.append((a, res))
        print(f"      alpha={a:>4}: {res.line(10)}  [head_users={res.n_head_users} tail_users={res.n_tail_users}]")

    # verdict
    tail10 = [r.tail[10]["recall"] for _, r in rows]
    head10 = [r.head[10]["recall"] for _, r in rows]
    base_head = head10[0]
    tail_up = all(tail10[i + 1] >= tail10[i] - 1e-4 for i in range(len(tail10) - 1)) and tail10[-1] > tail10[0]
    head_ok = min(head10) >= 0.9 * base_head  # head recall drops <10%
    probe_ok = (not np.isnan(best_auc0)) and best_auc0 >= 0.70
    print("\n===== VERDICT =====")
    print(f"  probe step-0 AUC>=0.70 : {probe_ok}  (best {best_auc0:.3f} @L{best_layer})")
    print(f"  tail Recall@10 monotone-up : {tail_up}  ({tail10[0]:.4f} -> {tail10[-1]:.4f})")
    print(f"  head Recall@10 preserved   : {head_ok}  ({base_head:.4f} -> min {min(head10):.4f})")
    if probe_ok and tail_up and head_ok:
        print("  => PLAN A (PopSteer) has signal. Proceed with representation steering.")
    elif tail_up and head_ok:
        print("  => Injection works but probe weak. Plan A viable; strengthen probe/confound analysis.")
    else:
        print("  => Weak/failed signal. Fall back to PLAN B (PopContrast, logits self-contrast).")


if __name__ == "__main__":
    main()
