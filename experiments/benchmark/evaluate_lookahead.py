"""Development experiment: terminal calibration and prior-mass lookahead.

Every completed item uses the SAME PopContrast score. Only prefix selection
changes. The prior-mass potential is an approximate continuation model; it is
not a proof of optimal finite-beam search, bias removal, or a new A* principle.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_bundle, load_model, metrics, paired_ci, save_json, stable_topk, tokens_for_histories, sha256
from .evaluate import standardize, select_validation, group_analysis
from .search import beam_search_batch, encode, make_trie, marginal_potential


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prior-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-valid", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--sid-length", type=int, default=3)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--betas", nargs="+", type=float, default=[.25, .5, .75, 1., 1.5])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    save_json(out / "config.json", {**vars(args), "checkpoint_sha256": sha256(args.checkpoint),
              "prior_sha256": sha256(args.prior_file), "quality_only": True,
              "hypothesis": "offline prior-mass lookahead approximates continuation tilt; terminal score unchanged",
              "development_status": "Beauty has been inspected during method development; do not treat it as untouched confirmation"})
    torch.set_num_threads(2)
    data = load_bundle(args.split, sid_length=args.sid_length)
    model = load_model(args.checkpoint, sid_length=args.sid_length).eval()
    with np.load(args.prior_file) as packed:
        marginal = packed["geometric"]
    prior = standardize(marginal)
    trie = make_trie(data["item_tokens"], prior)
    potentials = {beta: marginal_potential(data["item_tokens"], marginal, prior, beta) for beta in args.betas}
    save_json(out / "offline_structure.json", {"prefix_count": len(trie.min_prior),
              "potential_values_per_strength": len(trie.min_prior), "marginal_items": len(prior)})
    selection = {}
    for partition, n in [("valid", args.n_valid), ("test", args.n_test)]:
        if not n:
            continue
        indices = (np.random.default_rng(7).permutation(len(data["valid_target"]))[:n]
                   if partition == "valid" else data["legacy_test_idx"][:n])
        targets = data[partition + "_target"][indices]
        histories = data[partition + "_history"][indices]
        tops = {"raw": []}
        tops.update({f"{mode}:{beta}": [] for mode in ["rerank", "terminal", "potential"] for beta in args.betas})
        for start in range(0,len(histories),args.eval_batch_size):
            batch=histories[start:start+args.eval_batch_size]
            ids, mask = tokens_for_histories(batch, data["item_tokens"], "cuda")
            enc = encode(model, ids, mask)
            raw = beam_search_batch(model, enc, mask, trie, prior, args.beam)
            tops["raw"].extend(result["top"] for result in raw)
            for beta in args.betas:
                tops[f"rerank:{beta}"].extend(stable_topk(result["raw_scores"] - beta * prior[result["items"]], ids=result["items"])
                                               for result in raw)
                trie.potential = potentials[beta]
                for mode in ["terminal", "potential"]:
                    results = beam_search_batch(model, enc, mask, trie, prior, args.beam, beta, mode)
                    tops[f"{mode}:{beta}"].extend(result["top"] for result in results)
            if start//200 != (start+len(batch))//200:
                print(partition, start+len(batch), len(indices), flush=True)
        rows, hits = {}, {}
        for method, top in tops.items():
            rows[method], hits[method] = metrics(np.asarray(top), targets, data["head_mask"], len(prior))
            if method != "raw":
                rows[method]["paired_overall"] = paired_ci(hits["raw"], hits[method])
                tail = ~data["head_mask"][targets]
                rows[method]["paired_tail"] = paired_ci(hits["raw"][tail], hits[method][tail])
        save_json(out / f"{partition}_metrics.json", rows)
        np.savez_compressed(out / f"{partition}_top10.npz", indices=indices, targets=targets, **tops)
        save_json(out / f"{partition}_segments.json", {method: group_analysis(data, histories, targets, hits["raw"], hit)
                  for method, hit in hits.items() if method != "raw"})
        if partition == "valid":
            selection = select_validation(rows, hits, targets, data["head_mask"])
            save_json(out / "validation_selection.json", selection)
        else:
            save_json(out / "selected_test.json", {family: {rule: {"method": choices[rule], "test": rows[choices[rule]]}
                      for rule in ["strict", "budget_5pct", "simultaneous_ci_budget_5pct"]}
                      for family, choices in selection.items()})
        print(json.dumps({"partition": partition, "raw": rows["raw"], "potential:0.75": rows.get("potential:0.75")}), flush=True)
    save_json(out / "complete.json", {"finished_unix": time.time(), "n_valid": args.n_valid, "n_test": args.n_test})


if __name__ == "__main__":
    main()
