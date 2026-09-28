"""Export and verify compact, model-free inputs for local result analysis.

The export plan explicitly lists prediction files; weights, full score matrices,
training examples, and candidate caches are never traversed or copied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(args):
    root, out = Path(args.root), Path(args.output)
    plan = json.loads(Path(args.plan).read_text())
    datasets = {}
    manifest = {"format_version": 1, "records": [], "catalogs": {}, "priors": {}}
    for split in sorted({r["dataset"] for r in plan}):
        source = Path(args.data_root) / f"{split}.npz"
        with np.load(source, allow_pickle=False) as z:
            data = {k: z[k] for k in ["pop_counts", "head_mask", "valid_history",
                                      "test_history", "valid_target", "test_target"]}
            catalog = {k: z[k] for k in ["pop_counts", "head_mask", "item_codes", "item_tokens"] if k in z}
        datasets[split] = data
        path = out / "catalogs" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **catalog)
        manifest["catalogs"][split] = {"path": str(path.relative_to(out)),
            "items": len(data["pop_counts"]), "source_sha256": digest(source)}
    for entry in plan:
        source = root / entry["source"]
        data = datasets[entry["dataset"]]
        with np.load(source, allow_pickle=False) as z:
            indices, targets = z["indices"], z["targets"]
            methods = [k for k in z.files if k not in ("indices", "targets")]
            # Group methods by user to compress repeated rankings across the grid.
            tops = np.stack([z[k] for k in methods], axis=1)
        assert tops.shape == (len(targets), len(methods), 10), source
        assert tops.min() >= 0 and tops.max() < len(data["pop_counts"]), source
        assert max(int(tops.max()), int(indices.max())) < np.iinfo(np.int32).max
        partition = entry["partition"]
        np.testing.assert_array_equal(targets, data[partition + "_target"][indices])
        histories = data[partition + "_history"][indices]
        lengths = (histories >= 0).sum(1)
        head_count = (data["head_mask"][histories.clip(min=0)] * (histories >= 0)).sum(1)
        path = out / "predictions" / entry["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, indices=indices.astype(np.int32), targets=targets.astype(np.int32),
            methods=np.asarray(methods), top10=tops.astype(np.int32),
            history_length=lengths.astype(np.int16), history_head_count=head_count.astype(np.int16))
        metric_source = root / entry["metrics_source"]
        metric_path = path.with_suffix(".json")
        metrics = json.loads(metric_source.read_text())
        if "metrics_key" in entry:
            metrics = metrics[entry["metrics_key"]]
        write_json(metric_path, metrics)
        manifest["records"].append({k: entry[k] for k in ["run", "dataset", "partition"]} | {
            "path": str(path.relative_to(out)), "metrics": str(metric_path.relative_to(out)),
            "users": len(targets), "methods": len(methods),
            "source_sha256": digest(source), "metrics_source_sha256": digest(metric_source)})
        prior_source = source.parent / "priors.npz"
        if prior_source.exists() and entry["run"] not in manifest["priors"]:
            prior_path = out / "priors" / entry["run"] / "priors.npz"
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_path.write_bytes(prior_source.read_bytes())
            manifest["priors"][entry["run"]] = {"path": str(prior_path.relative_to(out)),
                "sha256": digest(prior_path)}
        marginal_source = source.parent / "marginal.npy"
        if marginal_source.exists() and entry["run"] not in manifest["priors"]:
            prior_path = out / "priors" / entry["run"] / "priors.npz"
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prior_path, geometric=np.load(marginal_source, allow_pickle=False))
            manifest["priors"][entry["run"]] = {"path": str(prior_path.relative_to(out)),
                "sha256": digest(prior_path), "source_sha256": digest(marginal_source),
                "statistic": "Mean item logit over sampled histories; not standardized."}
        print(f"Exported {entry['name']}: {len(targets)} users, {len(methods)} methods", flush=True)
    manifest["files"] = {str(p.relative_to(out)): {"sha256": digest(p), "bytes": p.stat().st_size}
                         for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)


def metrics(top, targets, head):
    matches = top == targets[:, None]
    hit = matches.any(axis=1)
    target_head = head[targets]
    counts = np.bincount(top.ravel(), minlength=len(head))
    positive = counts[counts > 0] / counts.sum()
    n = len(head)
    return {"n_users": len(targets), "hits": int(hit.sum()), "R10": float(hit.mean()),
        "N10": float((hit / np.log2(matches.argmax(axis=1) + 2)).mean()),
        "headR10": float(hit[target_head].mean()) if target_head.any() else None,
        "tailR10": float(hit[~target_head].mean()) if (~target_head).any() else None,
        "cov10": float(np.count_nonzero(counts) / n),
        "entropy": float(-(positive * np.log(positive)).sum() / np.log(n)),
        "gini": float(np.dot(2 * np.arange(1, n + 1) - n - 1, np.sort(counts)) / (n * counts.sum()))}, hit


def paired(base, new, repetitions):
    delta = new.astype(float) - base
    probabilities = [(delta == x).mean() for x in (-1, 0, 1)]
    draws = np.random.default_rng(20260927).multinomial(len(delta), probabilities, size=repetitions)
    ci = np.quantile((draws[:, 2] - draws[:, 0]) / len(delta), [.025, .975])
    return {"delta": float(delta.mean()), "ci95": ci.tolist(), "n": len(delta),
            "gained_hits": int((delta == 1).sum()), "lost_hits": int((delta == -1).sum())}


def verify(args):
    root = Path(args.directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
    comparisons = intervals = 0
    for record in manifest["records"]:
        expected = json.loads((root / record["metrics"]).read_text())
        expected = expected.get("metrics", expected)
        with np.load(root / manifest["catalogs"][record["dataset"]]["path"], allow_pickle=False) as catalog:
            head = catalog["head_mask"]
        with np.load(root / record["path"], allow_pickle=False) as z:
            methods, targets, top = z["methods"].tolist(), z["targets"], z["top10"]
        base = (top[:, methods.index("raw")] == targets[:, None]).any(1)
        for j, name in enumerate(methods):
            actual, hit = metrics(top[:, j], targets, head)
            for key, value in actual.items():
                reference = expected[name][key]
                assert value is None and reference is None or np.isclose(value, reference, rtol=0, atol=1e-12), (record["path"], name, key, value, reference)
                comparisons += 1
            for group, mask in [("overall", np.ones(len(targets), dtype=bool)), ("tail", ~head[targets])]:
                reference = expected[name].get("paired_" + group)
                if reference and mask.any():
                    actual_pair = paired(base[mask], hit[mask], reference.get("resamples", 2000))
                    for key, value in actual_pair.items():
                        if key in reference:
                            assert np.allclose(value, reference[key], rtol=0, atol=1e-12), (record["path"], name, group, key)
                    intervals += 1
    report = {"status": "passed", "records": len(manifest["records"]),
              "catalogs": len(manifest["catalogs"]), "metric_comparisons": comparisons,
              "paired_intervals": intervals, "verified_files": len(manifest["files"])}
    if args.report:
        write_json(Path(args.report), report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    for name in ["root", "data-root", "plan", "output"]:
        p.add_argument("--" + name, required=True)
    p.set_defaults(func=export)
    p = sub.add_parser("verify")
    p.add_argument("directory")
    p.add_argument("--report")
    p.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
