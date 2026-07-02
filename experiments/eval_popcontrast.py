"""Overnight experiment driver for PopContrast (run from genrec/ root).

Strategy: the expensive step is scoring the full catalog per user. Since every
prior/beta/floor/adaptive variant is just POST-HOC arithmetic on those baseline
log-probs, we score each eval user ONCE, cache (N_users, N_items) on GPU, then
sweep ALL method configs instantly via vectorized topk.

Outputs (absolute paths under PopSteer/results and PopSteer/reports):
  results/cache_scores.pt        cached baseline scores + targets + segs (reusable)
  results/main_panel.json        baseline / model-PMI / naive / floored / adaptive panels
  results/oracle_recovery.json   exact-oracle vs small-beam tail-recovery diagnostic
  reports/NIGHT_SUMMARY.md       written separately after review

Robust: each block in try/except, results saved incrementally.
"""
from __future__ import annotations
import json, math, os, traceback
import numpy as np
import torch

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor
from popcontrast.decoding import trie_beam_search

RES = "/home/hanyu/research/PopSteer/results"
os.makedirs(RES, exist_ok=True)
DEVICE = "cuda"
SPLIT = os.environ.get("NIGHT_SPLIT", "beauty")
CKPT = f"out/tiger/amazon/{SPLIT}/best_model.pt"
N_EVAL = int(os.environ.get("NIGHT_N_EVAL", 5000))
MARGINAL_M = int(os.environ.get("NIGHT_MARGINAL_M", 512))
SEED = 0


def log(msg):
    print(msg, flush=True)


def save_json(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)
    log(f"  [saved] {name}")


# ---------- vectorized metric panel on cached scores ----------
def panel(scores, targets, seg_head, prior=None, beta=0.0, floor=None, beta_vec=None, n_items=None):
    """scores (N,I) GPU; prior (I,) or None; beta scalar or beta_vec (N,) per-user.
    floor: keep only items with logp >= rowmax + log(floor) (prob-ratio); else -inf."""
    adj = scores
    if prior is not None and (beta != 0.0 or beta_vec is not None):
        b = beta_vec[:, None] if beta_vec is not None else beta
        adj = scores - b * prior[None, :]
    if floor is not None:
        rowmax = scores.max(dim=1, keepdim=True).values
        mask = scores < (rowmax + math.log(floor))
        adj = adj.masked_fill(mask, float("-inf"))
    top = adj.topk(10, dim=1).indices                      # (N,10)
    tgt = targets[:, None]
    match = (top == tgt)                                   # (N,10)
    hit10 = match.any(dim=1).float()
    hit5 = match[:, :5].any(dim=1).float()
    # ndcg@10: 1/log2(rank+2) at first match
    rankpos = torch.where(match.any(1), match.float().argmax(1), torch.full_like(hit10, -1).long())
    ndcg10 = torch.where(match.any(1), 1.0 / torch.log2(rankpos.float() + 2), torch.zeros_like(hit10))
    def seg(v, m):
        return float(v[m].mean().item()) if m.any() else 0.0
    tail = ~seg_head
    counts = torch.bincount(top[:, :10].reshape(-1), minlength=n_items).float().cpu().numpy()
    nz = counts[counts > 0]
    p = nz / nz.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(n_items)) if len(nz) else 0.0
    xs = np.sort(counts); nnn = len(xs)
    gini = float((2*np.arange(1,nnn+1)-nnn-1).dot(xs)/(nnn*xs.sum())) if xs.sum()>0 else 0.0
    return {
        "R5": float(hit5.mean()), "R10": float(hit10.mean()), "N10": float(ndcg10.mean()),
        "headR10": seg(hit10, seg_head), "tailR10": seg(hit10, tail),
        "tailN10": seg(ndcg10, tail),
        "cov10": float((counts > 0).sum() / n_items), "gini": gini, "ent": ent,
    }


def main():
    torch.manual_seed(SEED)
    log(f"[setup] loading dataset+model ({SPLIT})")
    ds_tr = build_dataset(split=SPLIT, train_test_split="train")
    ds_te = build_dataset(split=SPLIT, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    tables = build_sem_id_tables(ds_tr)
    model = load_tiger(CKPT, device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE)
    n_items = item_tok.shape[0]
    rng = np.random.default_rng(SEED)
    marg_samples = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:MARGINAL_M]]
    eval_samples = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:N_EVAL]]

    # ---- model marginal ----
    log(f"[marginal] averaging over {MARGINAL_M} histories")
    marg = torch.zeros(n_items, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in marg_samples:
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            marg += score_all_items(model, enc, attn, item_tok)
    marg /= MARGINAL_M

    # ---- cache baseline scores for eval users ----
    cache_path = os.path.join(RES, f"cache_scores_{SPLIT}.pt")
    log(f"[cache] scoring {N_EVAL} eval users (one full-catalog pass each)")
    scores = torch.empty(len(eval_samples), n_items, device=DEVICE)
    targets = torch.empty(len(eval_samples), dtype=torch.long, device=DEVICE)
    seg_head = torch.zeros(len(eval_samples), dtype=torch.bool, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, s in enumerate(eval_samples):
            enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
            scores[i] = score_all_items(model, enc, attn, item_tok)
            targets[i] = s["target"]
            seg_head[i] = (pop.bucket[s["target"]] == "head")
            if (i + 1) % 1000 == 0:
                log(f"    scored {i+1}/{len(eval_samples)}")
    torch.save({"scores": scores.cpu(), "targets": targets.cpu(),
                "seg_head": seg_head.cpu(), "marginal": marg.cpu()}, cache_path)
    log(f"  [saved] cache_scores.pt  (head={int(seg_head.sum())} tail={int((~seg_head).sum())})")

    # priors (z-normalized so beta comparable)
    def z(t): return (t - t.mean()) / t.std().clamp_min(1e-6)
    prior_model = z(marg)
    prior_naive = z(torch.tensor(np.log1p(pop.counts), dtype=torch.float32, device=DEVICE))

    results = {"meta": {"n_eval": len(eval_samples), "n_items": n_items,
                        "head_users": int(seg_head.sum()), "tail_users": int((~seg_head).sum()),
                        "marginal_m": MARGINAL_M}}

    # ---- Block 1: baseline ----
    results["baseline"] = panel(scores, targets, seg_head, n_items=n_items)
    log(f"[baseline] {results['baseline']}")

    # ---- Block 2: model-PMI beta sweep ----
    betas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    results["model_pmi"] = {}
    for b in betas:
        results["model_pmi"][b] = panel(scores, targets, seg_head, prior_model, beta=b, n_items=n_items)
    log("[model_pmi] done")

    # ---- Block 3: naive log-pop beta sweep ----
    results["naive"] = {}
    for b in betas:
        results["naive"][b] = panel(scores, targets, seg_head, prior_naive, beta=b, n_items=n_items)
    log("[naive] done")

    # ---- Block 4: floored PMI (plausibility floor x beta) ----
    results["floored"] = {}
    for floor in [1e-1, 1e-2, 1e-3]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0]:
            results["floored"][f"floor{floor}_b{b}"] = panel(
                scores, targets, seg_head, prior_model, beta=b, floor=floor, n_items=n_items)
    log("[floored] done")

    # ---- Block 5: adaptive per-user beta (gate by baseline popularity tilt) ----
    # tilt_u = mean model-marginal over user's baseline top-10 items (how popularity-defaulting)
    base_top = scores.topk(10, dim=1).indices
    tilt = prior_model[base_top].mean(dim=1)               # (N,)
    tilt_z = (tilt - tilt.median()) / tilt.std().clamp_min(1e-6)
    results["adaptive"] = {}
    for b in [0.5, 1.0, 1.5]:
        bvec = b * torch.sigmoid(tilt_z)                   # stronger when baseline is popularity-tilted
        results["adaptive"][f"b{b}"] = panel(scores, targets, seg_head, prior_model,
                                             beta_vec=bvec, n_items=n_items)
    log("[adaptive] done")

    save_json(f"main_panel_{SPLIT}.json", results)
    return scores, targets, seg_head, prior_model, marg, model, tables, pop, item_tok, eval_samples


if __name__ == "__main__":
    try:
        main()
        log("[DONE] main_panel complete")
    except Exception:
        log("[ERROR] " + traceback.format_exc())
