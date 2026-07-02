"""Signature diagnostic: is long-tail loss due to BEAM SEARCH PRUNING (SimGR's
claim) or the DECODE DISTRIBUTION / marginal (our claim)?

Compares, on the same users:
  - exact  : full-catalog exact top-K (from cached baseline scores = the oracle)
  - beam@b : trie-constrained beam search top-K (b in {10,20})
Metrics: R@10 (overall/head/tail), coverage@10, and top-10 set overlap(beam,exact).

Interpretation:
  beam ≈ exact  -> search is NOT the bottleneck; the tail loss is distributional
                   (popularity in the marginal) -> supports PopContrast's premise.
  beam << exact on tail -> search pruning drops tail (SimGR-style) -> different story.

Run from genrec/ root AFTER eval_popcontrast.py has produced results/cache_scores.pt.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch

from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, _item_tokens_tensor
from popcontrast.decoding import trie_beam_search

RES = "/home/hanyu/research/PopSteer/results"
DEVICE = "cuda"
N_DIAG = int(os.environ.get("DIAG_N", 2000))
SEED = 0


def main():
    ds_tr = build_dataset(split="beauty", train_test_split="train")
    ds_te = build_dataset(split="beauty", train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr)
    tables = build_sem_id_tables(ds_tr)
    model = load_tiger("out/tiger/amazon/beauty/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE)
    n_items = item_tok.shape[0]

    # SELF-CONTAINED: compute exact AND beam from the SAME freshly-sampled users
    # (no reliance on the night_run cache — that caused an rng-alignment bug).
    from popcontrast.oracle import score_all_items
    rng = np.random.default_rng(SEED)
    eval_idx = rng.permutation(len(ds_te.samples))[:N_DIAG]
    eval_samples = [ds_te.samples[i] for i in eval_idx]
    N = len(eval_samples)
    is_head = np.array([pop.bucket[s["target"]] == "head" for s in eval_samples])

    beams = [10, 20]
    agg = {"exact": {"hit": [], "cov": set()}}
    for b in beams:
        agg[f"beam{b}"] = {"hit": [], "cov": set(), "overlap": []}

    for i in range(N):
        s = eval_samples[i]; tgt = int(s["target"])
        enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc = score_all_items(model, enc, attn, item_tok)
        et = sc.topk(10).indices.cpu().numpy()
        agg["exact"]["hit"].append(int(tgt in et)); agg["exact"]["cov"].update(et.tolist())
        for b in beams:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                items, _, _ = trie_beam_search(model, enc, attn, tables, num_beams=b, device=DEVICE)
            bt = items[:10]
            agg[f"beam{b}"]["hit"].append(int(tgt in bt))
            agg[f"beam{b}"]["cov"].update(bt)
            agg[f"beam{b}"]["overlap"].append(len(set(bt) & set(et.tolist())) / 10.0)

    def r(hits, mask=None):
        h = np.array(hits)
        if mask is not None:
            h = h[mask]
        return float(h.mean()) if len(h) else 0.0

    out = {"n": N, "n_items": n_items,
           "head_users": int(is_head.sum()), "tail_users": int((~is_head).sum())}
    out["exact"] = {"R10": r(agg["exact"]["hit"]),
                    "headR10": r(agg["exact"]["hit"], is_head),
                    "tailR10": r(agg["exact"]["hit"], ~is_head),
                    "cov10": len(agg["exact"]["cov"]) / n_items}
    for b in beams:
        out[f"beam{b}"] = {"R10": r(agg[f"beam{b}"]["hit"]),
                           "headR10": r(agg[f"beam{b}"]["hit"], is_head),
                           "tailR10": r(agg[f"beam{b}"]["hit"], ~is_head),
                           "cov10": len(agg[f"beam{b}"]["cov"]) / n_items,
                           "overlap_with_exact": float(np.mean(agg[f"beam{b}"]["overlap"]))}
    with open(os.path.join(RES, "oracle_recovery.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
