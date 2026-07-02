"""PopContrast (main method): decode-time popularity debiasing by subtracting a
self-contained popularity prior from the item log-prob.

rank items by:  s(item|h) = logp(item|h) - beta * prior(item)

Two priors compared:
  model : model's OWN marginal, mean over M histories of logp(item|h)   [PMI/CAD-style, self-contained]
  naive : beta * log(1 + interaction_count)                              [proposal baseline #6, uses external counts]

Full metric panel vs beta: Recall/NDCG@10 (overall/head/tail), Coverage@10,
Gini + normalized entropy of recommendation counts (degenerate-collapse detectors).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def norm_entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


@torch.no_grad()
def compute_model_marginal(model, samples, tables, item_tok, M, device="cuda"):
    acc = torch.zeros(item_tok.shape[0], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in samples[:M]:
            enc, attn = encode_history(model, s["history"], tables, device=device)
            acc += score_all_items(model, enc, attn, item_tok)
    return acc / M


@torch.no_grad()
def evaluate(model, samples, tables, pop, item_tok, prior, beta, device="cuda"):
    """prior: (N,) tensor to subtract (beta*prior); K=5,10 panel + coverage/gini/entropy."""
    Ks = (5, 10)
    bucket = pop.bucket
    n_items = item_tok.shape[0]
    rec = {seg: {k: [] for k in Ks} for seg in ("overall", "head", "tail")}
    ndcg = {seg: {k: [] for k in Ks} for seg in ("overall", "head", "tail")}
    counts = np.zeros(n_items, dtype=np.int64)  # rec counts @10 for gini/entropy
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in samples:
            tgt = s["target"]; seg = "head" if bucket[tgt] == "head" else "tail"
            enc, attn = encode_history(model, s["history"], tables, device=device)
            scores = score_all_items(model, enc, attn, item_tok)
            if beta != 0.0:
                scores = scores - beta * prior
            topk = torch.topk(scores, max(Ks)).indices.cpu().numpy()
            counts[topk[:10]] += 1
            pos = np.where(topk == tgt)[0]
            rank = int(pos[0]) if len(pos) else None
            for k in Ks:
                hit = rank is not None and rank < k
                r = 1.0 if hit else 0.0
                n = (1.0 / np.log2(rank + 2)) if hit else 0.0
                rec["overall"][k].append(r); ndcg["overall"][k].append(n)
                rec[seg][k].append(r); ndcg[seg][k].append(n)
    agg = lambda d, k: float(np.mean(d[k])) if d[k] else 0.0
    return {
        "R10": agg(rec["overall"], 10), "N10": agg(ndcg["overall"], 10),
        "R5": agg(rec["overall"], 5),
        "headR10": agg(rec["head"], 10), "tailR10": agg(rec["tail"], 10),
        "tailN10": agg(ndcg["tail"], 10),
        "cov10": float((counts > 0).sum() / n_items),
        "gini": gini(counts), "ent": norm_entropy(counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--checkpoint", default="out/tiger/amazon/beauty/best_model.pt")
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--marginal-m", type=int, default=256)
    ap.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5, 0.75])
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

    print(f"[priors] model marginal over {args.marginal_m} histories + naive log-pop")
    model_marg = compute_model_marginal(model, marg_samples, tables, item_tok, args.marginal_m, device)
    # normalize priors to comparable scale (zero-mean, unit-std) so beta is comparable
    def z(t):
        return (t - t.mean()) / t.std().clamp_min(1e-6)
    priors = {
        "model": z(model_marg),
        "naive": z(torch.tensor(np.log1p(pop.counts), dtype=torch.float32, device=device)),
    }
    print(f"[eval] {len(eval_samples)} users "
          f"(head={sum(1 for s in eval_samples if pop.bucket[s['target']]=='head')}, "
          f"tail={sum(1 for s in eval_samples if pop.bucket[s['target']]=='tail')})")

    hdr = f"{'prior':6} {'beta':>5} {'R@10':>7} {'N@10':>7} {'headR':>7} {'tailR':>7} {'tailN':>7} {'cov':>7} {'gini':>6} {'ent':>6}"
    print(hdr)
    # baseline once
    b0 = evaluate(model, eval_samples, tables, pop, item_tok, None, 0.0, device)
    print(f"{'base':6} {0.0:5.2f} {b0['R10']:7.4f} {b0['N10']:7.4f} {b0['headR10']:7.4f} "
          f"{b0['tailR10']:7.4f} {b0['tailN10']:7.4f} {b0['cov10']:7.4f} {b0['gini']:6.3f} {b0['ent']:6.3f}")
    for name, prior in priors.items():
        for b in args.betas:
            if b == 0.0:
                continue
            m = evaluate(model, eval_samples, tables, pop, item_tok, prior, b, device)
            print(f"{name:6} {b:5.2f} {m['R10']:7.4f} {m['N10']:7.4f} {m['headR10']:7.4f} "
                  f"{m['tailR10']:7.4f} {m['tailN10']:7.4f} {m['cov10']:7.4f} {m['gini']:6.3f} {m['ent']:6.3f}")


if __name__ == "__main__":
    main()
