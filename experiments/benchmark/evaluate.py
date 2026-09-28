"""Matched beam accuracy, validation selection, and prefix-bound experiments.

Every method sees identical users, histories, checkpoint, and exact item labels.
Quality runs may share a GPU with training; they make NO online latency claim.
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import (load_bundle, load_model, metrics, paired_ci, save_json,
                     seed_all, sha256, stable_topk, tokens_for_histories)
from .search import beam_search, beam_search_batch, encode, make_trie, score_items, score_items_cached

BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def catalog_scores(model, enc, mask, tokens, args):
    scorer = score_items_cached if getattr(args,"cached_catalog_scoring",False) else score_items
    return scorer(model,enc,mask,tokens,args.chunk_size)


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(ddof=1), 1e-8)


def select_validation(rows, hits, targets, head_mask):
    """Same recall budgets and selection rule for each estimator family."""
    base = hits["raw"]
    tail = ~head_mask[targets]
    budget = .05 * base.mean()
    out = {}
    families = sorted({key.split(":")[0] for key in rows if key != "raw"})
    for family in families:
        names = ["raw"] + [key for key in rows if key.startswith(family + ":")]
        strict = [name for name in names if hits[name].mean() >= base.mean()]
        budgeted = [name for name in names if hits[name].mean() - base.mean() >= -budget]
        # Bonferroni one-sided paired bootstrap bounds over this finite grid.
        feasible = ["raw"]
        bounds = {}
        for name in names[1:]:
            delta = hits[name].astype(float) - base
            prob = [(delta == x).mean() for x in [-1, 0, 1]]
            draws = np.random.default_rng(41).multinomial(len(base), prob, size=10000)
            lower = float(np.quantile((draws[:, 2] - draws[:, 0]) / len(base), .05 / max(1, len(names) - 1)))
            bounds[name] = lower
            if lower >= -budget:
                feasible.append(name)
        def best(candidates):
            # Higher tail recall, then overall recall, then prefer no correction.
            return max(candidates, key=lambda key: (float(hits[key][tail].mean()),
                        float(hits[key].mean()), key == "raw", -names.index(key)))
        out[family] = {"strict": best(strict), "budget_5pct": best(budgeted),
                       "simultaneous_ci_budget_5pct": best(feasible),
                       "absolute_recall_budget": float(budget), "one_sided_lower_bounds": bounds}
    return out


@torch.no_grad()
def priors(model, data, args, output):
    path = output / "priors.npz"
    if path.exists():
        with np.load(path) as source:
            return {key: source[key] for key in source.files}
    tokens = torch.as_tensor(data["item_tokens"], device="cuda", dtype=torch.long)
    start = time.perf_counter()
    if args.legacy_prior:
        if args.sid_length != 3:
            raise ValueError("Legacy marginal is only valid for original 3-token model")
        # Hash verification prevents silently using an old model's marginal on a new model.
        meta_path = Path(args.data_dir) / f"{args.split}.json" if args.data_dir else Path(__file__).resolve().parents[2] / f"results/benchmark/data/{args.split}.json"
        expected = json.loads(meta_path.read_text())["checkpoint_sha256"]
        if sha256(args.checkpoint) != expected:
            raise ValueError("Legacy marginal/checkpoint mismatch")
        values = {"geometric": data["legacy_marginal"].astype(np.float64)}
    else:
        all_scores = []
        indices = data["legacy_marginal_idx"][:args.marginal_m]
        if len(indices) != args.marginal_m:
            raise ValueError("Marginal M exceeds exported sampling budget")
        for count, index in enumerate(indices):
            ids, mask = tokens_for_histories(data["train_history"][[index]], data["item_tokens"], "cuda")
            with autocast(args):
                enc = encode(model, ids, mask)
                scores = catalog_scores(model, enc, mask, tokens, args)
                if count == 0 and getattr(args,"cached_catalog_scoring",False):
                    reference=score_items(model,enc,mask,tokens[:1024],args.chunk_size)
                    error=float((scores[:1024]-reference).abs().max())
                    if error>2e-5:
                        raise AssertionError(f"Cached/teacher-forced score mismatch: {error}")
                    save_json(output/"catalog_cache_audit.json",{"items":len(reference),"max_abs_error":error,
                              "reference":"original full teacher-forced scoring", "chunk_size":args.chunk_size})
            all_scores.append(scores.cpu().numpy())
            if (count + 1) % 32 == 0:
                print(f"prior {count + 1}/{len(indices)}", flush=True)
        values_all = np.asarray(all_scores, dtype=np.float64)
        max_scores = values_all.max(axis=0)
        values = {"geometric": values_all.mean(0),
                  "arithmetic": max_scores + np.log(np.exp(values_all - max_scores).mean(0))}
        np.savez_compressed(output / "sampled_history_scores.npz", scores=values_all,
                            indices=indices)
    history_seconds=time.perf_counter()-start
    null_started=time.perf_counter()
    ids = torch.zeros((1, 50 * args.sid_length), dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    mask[0, -1] = 1
    with autocast(args):
        enc = encode(model, ids, mask)
        values["null"] = catalog_scores(model, enc, mask, tokens, args).cpu().numpy()
    null_seconds=time.perf_counter()-null_started
    values["logcount"] = np.log1p(data["pop_counts"])
    values["poprank"] = np.argsort(np.argsort(data["pop_counts"], kind="stable"), kind="stable").astype(float)
    np.savez_compressed(path, **values)
    save_json(output / "offline_cost.json", {"estimation_seconds": time.perf_counter() - start,
               "history_estimation_seconds": history_seconds, "null_estimation_seconds": null_seconds,
               "legacy_marginal_reused": args.legacy_prior,
               "sampled_histories": 512 if args.legacy_prior else args.marginal_m,
               "geometric_vector_bytes_float32": 4 * len(tokens),
               "timing_gpu_shared": args.gpu_shared,
               "note": "No latency inference from a shared-GPU run; reused legacy marginal cost is excluded."})
    return values


def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else contextlib.nullcontext()


def group_analysis(data, histories, targets, base_hit, hit):
    counts = data["pop_counts"]
    order = np.argsort(counts, kind="stable")
    quintile = np.empty(len(counts), dtype=int)
    for q, indices in enumerate(np.array_split(order, 5)):
        quintile[indices] = q
    history_lengths = (histories >= 0).sum(1)
    head_share = (data["head_mask"][histories.clip(min=0)] * (histories >= 0)).sum(1) / history_lengths.clip(min=1)
    masks = {f"target_popularity_q{q + 1}": quintile[targets] == q for q in range(5)}
    masks.update({"history_1to5": history_lengths <= 5,
                  "history_6to10": (history_lengths > 5) & (history_lengths <= 10),
                  "history_over10": history_lengths > 10,
                  "head_affinity_low": head_share < 1/3,
                  "head_affinity_middle": (head_share >= 1/3) & (head_share < 2/3),
                  "head_affinity_high": head_share >= 2/3,
                  "popular_target_high_head_affinity": data["head_mask"][targets] & (head_share >= 2/3)})
    return {name: {"n": int(mask.sum()), "baseline_hits": int(base_hit[mask].sum()),
                   "corrected_hits": int(hit[mask].sum()),
                   "paired": paired_ci(base_hit[mask], hit[mask])}
            for name, mask in masks.items()}


@torch.no_grad()
def evaluate_partition(model, data, estimators, args, output, partition):
    n = args.n_valid if partition == "valid" else args.n_test
    if not n:
        return {}, {}, np.empty(0, dtype=int)
    all_targets = data[partition + "_target"]
    indices = (np.random.default_rng(7).permutation(len(all_targets))[:n] if partition == "valid"
               else data["legacy_test_idx"][:n])
    targets = all_targets[indices]
    histories = data[partition + "_history"][indices]
    z = {name: standardize(values) for name, values in estimators.items()}
    trie = make_trie(data["item_tokens"], z["geometric"])
    tokens = torch.as_tensor(data["item_tokens"], device="cuda", dtype=torch.long)
    tops = {"raw": []}
    for name in z:
        for beta in BETAS[1:]:
            tops[f"{name}:{beta}"] = []
    for beta in args.bound_betas:
        tops[f"bound:{beta}"] = []
    diagnostics = []
    candidate_items, candidate_scores, candidate_offsets = [], [], [0]
    for start in range(0, len(histories), args.eval_batch_size):
        ids, masks = tokens_for_histories(histories[start:start + args.eval_batch_size], data["item_tokens"], "cuda")
        with autocast(args):
            encodings = encode(model, ids, masks)
            raw_batch = beam_search_batch(model, encodings, masks, trie, z["geometric"], args.beam)
            bound_batch = {beta: beam_search_batch(model, encodings, masks, trie, z["geometric"], args.beam, beta, "bound")
                           for beta in args.bound_betas}
            for local, raw in enumerate(raw_batch):
                row = start + local
                enc, mask = encodings[local:local+1], masks[local:local+1]
                candidate_items.extend(raw["items"].tolist())
                candidate_scores.extend(raw["raw_scores"].tolist())
                candidate_offsets.append(len(candidate_items))
                tops["raw"].append(raw["top"])
                for name, prior in z.items():
                    for beta in BETAS[1:]:
                        tops[f"{name}:{beta}"].append(stable_topk(raw["raw_scores"] - beta * prior[raw["items"]], ids=raw["items"]))
                bound = {beta: result[local] for beta, result in bound_batch.items()}
                for beta in args.bound_betas:
                    tops[f"bound:{beta}"].append(bound[beta]["top"])
                if row < args.exact_n:
                    exact = catalog_scores(model, enc, mask, tokens, args).cpu().numpy().astype(float)
                    audit = {"sample_index": int(indices[row]),
                             "score_max_abs_error_raw_candidates": float(np.max(np.abs(exact[raw["items"]] - raw["raw_scores"]))),
                             "comparisons": {}}
                    for beta in sorted(set([.75, 1.5] + args.bound_betas)):
                        exact_scores = exact - beta * z["geometric"]
                        exact_top = stable_topk(exact_scores)
                        leaf_top = tops[f"geometric:{beta}"][-1]
                        replay = stable_topk(exact_scores[raw["items"]], ids=raw["items"])
                        audit["comparisons"][str(beta)] = {
                            "in_raw_beam": float(np.isin(exact_top, raw["items"]).mean()),
                            "leaf_overlap": float(np.isin(leaf_top, exact_top).mean()),
                            "common_score_replay_overlap": float(np.isin(replay, exact_top).mean()),
                            "leaf_exact_top": exact_top.tolist(),
                            "bound_overlap": float(np.isin(bound[beta]["top"], exact_top).mean()) if beta in bound else None,
                            "in_bound_beam": float(np.isin(exact_top, bound[beta]["items"]).mean()) if beta in bound else None}
                    diagnostics.append(audit)
        done = start + len(raw_batch)
        if done // 100 != start // 100 or done == len(indices):
            print(f"{partition}: {done}/{len(indices)}", flush=True)
            save_json(output / f"progress_{partition}.json", {"done": done, "total": len(indices)})
    rows, hits, arrays = {}, {}, {"indices": indices, "targets": targets}
    for name, lists in tops.items():
        top = np.asarray(lists)
        rows[name], hits[name] = metrics(top, targets, data["head_mask"], len(tokens))
        arrays[name] = top
        if name != "raw":
            rows[name]["paired_overall"] = paired_ci(hits["raw"], hits[name])
            tail = ~data["head_mask"][targets]
            rows[name]["paired_tail"] = paired_ci(hits["raw"][tail], hits[name][tail])
    save_json(output / f"{partition}_metrics.json", rows)
    save_json(output / f"{partition}_exact_audit.json", diagnostics)
    np.savez_compressed(output / f"{partition}_top10.npz", **arrays)
    np.savez_compressed(output / f"{partition}_candidates.npz", indices=indices, targets=targets,
                       items=np.asarray(candidate_items), raw=np.asarray(candidate_scores),
                       offsets=np.asarray(candidate_offsets))
    groups = {name: group_analysis(data, histories, targets, hits["raw"], hits[name])
              for name in ["geometric:0.5", "geometric:0.75", "geometric:1.5"]}
    save_json(output / f"{partition}_segments.json", groups)
    # Direct per-item gained/lost exact-target hits, including popular items.
    corrected = hits["geometric:0.75"]
    np.savez_compressed(output / f"{partition}_item_hits.npz",
             pop_counts=data["pop_counts"], head_mask=data["head_mask"],
             target_count=np.bincount(targets, minlength=len(tokens)),
             baseline_hits=np.bincount(targets[hits["raw"]], minlength=len(tokens)),
             corrected_hits=np.bincount(targets[corrected], minlength=len(tokens)),
             gained=np.bincount(targets[corrected & ~hits["raw"]], minlength=len(tokens)),
             lost=np.bincount(targets[~corrected & hits["raw"]], minlength=len(tokens)))
    print(json.dumps({"partition": partition, "raw": rows["raw"],
                      "geometric:0.75": rows["geometric:0.75"]}), flush=True)
    return rows, hits, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--sid-length", type=int, default=3)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--n-valid", type=int, default=5000)
    ap.add_argument("--exact-n", type=int, default=128)
    ap.add_argument("--marginal-m", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--cached-catalog-scoring", action="store_true")
    ap.add_argument("--bound-betas", nargs="*", type=float, default=[.75, 1.5])
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--legacy-prior", action="store_true")
    ap.add_argument("--gpu-shared", action="store_true")
    args = ap.parse_args()
    if args.beam < 10 or args.n_test > 5000:
        raise ValueError("Require B>=10 and n_test<=5000 for fixed exported subsets")
    if any(beta not in BETAS for beta in args.bound_betas):
        raise ValueError("Bound strengths must belong to the prespecified grid")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "checkpoint_sha256": sha256(args.checkpoint),
              "torch": torch.__version__, "tie_rule": "descending score then ascending item ID",
              "beam_rule": "lexicographic prefixes on score ties; complete items sorted by ID",
              "log_softmax": "float32 for both exact and beam", "timestamp": time.time(),
              "quality_only": True, "betas": BETAS}
    if (output / "config.json").exists():
        old = json.loads((output / "config.json").read_text())
        for key in [*vars(args), "checkpoint_sha256"]:
            if old.get(key, False if key == "cached_catalog_scoring" else None) != config[key]:
                raise ValueError(f"Refusing to mix different evaluation settings: {key}")
    if (output / "complete.json").exists():
        print("Already complete:", output, flush=True)
        return
    save_json(output / "config.json", config)
    seed_all(20260927)
    torch.set_num_threads(2)
    data = load_bundle(args.split, args.data_dir, args.sid_length)
    model = load_model(args.checkpoint, sid_length=args.sid_length).eval()
    estimators = priors(model, data, args, output)
    val, val_hit, val_tgt = evaluate_partition(model, data, estimators, args, output, "valid")
    selection = select_validation(val, val_hit, val_tgt, data["head_mask"]) if val else {}
    save_json(output / "validation_selection.json", selection)
    test, _, _ = evaluate_partition(model, data, estimators, args, output, "test")
    selected = {family: {rule: {"selected_method": choices[rule], "test": test.get(choices[rule])}
                  for rule in ["strict", "budget_5pct", "simultaneous_ci_budget_5pct"]}
                for family, choices in selection.items()}
    save_json(output / "selected_test.json", selected)
    save_json(output / "complete.json", {"completed_unix": time.time(),
              "checkpoint_sha256": config["checkpoint_sha256"], "n_test": args.n_test, "n_valid": args.n_valid})


if __name__ == "__main__":
    main()
