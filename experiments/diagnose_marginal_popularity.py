"""Decisive diagnostic: is popularity in the MARGINAL p(item) (not a residual dir)?

Compute a marginal/popularity-prior score per item = mean over M sampled histories
of exact log p(item|h). Then:
  (1) Spearman(popularity, marginal) — does the model's prior encode popularity?
  (2) rank by PMI = score(item|h) - beta * marginal[item]; sweep beta; measure
      head/tail/coverage. If tail rises as beta grows, popularity WAS in the
      marginal and this self-contained (no external model) correction fixes it.

This is the research-recommended #7 (CAD/PMI) — the mechanism most likely to move
tail — and simultaneously confirms WHY representation ablation failed (#8).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor


@torch.no_grad()
def compute_marginal(model, samples, tables, item_tok, M=256, device="cuda"):
    """marginal[item] = mean_h log p(item | h) over M sampled histories (model's prior)."""
    acc = torch.zeros(item_tok.shape[0], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in samples[:M]:
            enc, attn = encode_history(model, s["history"], tables, device=device)
            acc += score_all_items(model, enc, attn, item_tok)
    return acc / M


@torch.no_grad()
def eval_pmi(model, samples, tables, pop, item_tok, marginal, beta, device="cuda"):
    K = 10
    bucket = pop.bucket
    rec = {"overall": [], "head": [], "tail": []}
    recommended = set()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in samples:
            tgt = s["target"]; seg = "head" if bucket[tgt] == "head" else "tail"
            enc, attn = encode_history(model, s["history"], tables, device=device)
            scores = score_all_items(model, enc, attn, item_tok) - beta * marginal
            topk = torch.topk(scores, K).indices.cpu().numpy()
            hit = 1.0 if tgt in topk else 0.0
            rec["overall"].append(hit); rec[seg].append(hit)
            recommended.update(topk.tolist())
    return (np.mean(rec["overall"]), np.mean(rec["head"]), np.mean(rec["tail"]),
            len(recommended) / item_tok.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--checkpoint", default="out/tiger/amazon/beauty/best_model.pt")
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--marginal-m", type=int, default=256)
    ap.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()
    device = "cuda"

    ds_tr = build_dataset(split=args.split, train_test_split="train")
    ds_te = build_dataset(split=args.split, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    tables = build_sem_id_tables(ds_tr)
    model = load_tiger(args.checkpoint, device=device)
    item_tok = _item_tokens_tensor(tables, device)

    rng = np.random.default_rng(0)
    marg_samples = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:args.marginal_m]]
    eval_samples = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:args.eval_n]]

    print(f"[marginal] averaging over {args.marginal_m} histories...")
    marginal = compute_marginal(model, marg_samples, tables, item_tok, M=args.marginal_m, device=device)

    # (1) does the marginal encode popularity?
    logpop = np.log1p(pop.counts)
    rho, _ = spearmanr(marginal.float().cpu().numpy(), logpop)
    print(f"[diagnostic] Spearman(model marginal score, log popularity) = {rho:.3f}")
    print(f"             (high positive => popularity lives in the marginal p(item))")

    # (2) PMI sweep
    print(f"\n[eval] {len(eval_samples)} users "
          f"(head={sum(1 for s in eval_samples if pop.bucket[s['target']]=='head')}, "
          f"tail={sum(1 for s in eval_samples if pop.bucket[s['target']]=='tail')})")
    print(f"{'beta':>6} {'overall':>8} {'head':>8} {'tail':>8} {'cov':>8}")
    for b in args.betas:
        o, h, t, c = eval_pmi(model, eval_samples, tables, pop, item_tok, marginal, b, device)
        print(f"{b:6.2f} {o:8.4f} {h:8.4f} {t:8.4f} {c:8.4f}")


if __name__ == "__main__":
    main()
