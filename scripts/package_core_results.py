"""Package compact results and CPU plotting code; verify the extracted archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/core_results")
    args = parser.parse_args()
    compact = ROOT / "results/core_analysis"
    if not (compact / "manifest.json").exists():
        parser.error("Export and synchronize results/core_analysis first.")
    args.output.mkdir(parents=True, exist_ok=True)
    # Analysis inputs are intentionally ignored by Git. Package them explicitly
    # from the local archive instead of relying on the Git index.
    files = {}
    for path in (ROOT / "results").rglob("*"):
        relative = path.relative_to(ROOT)
        if (path.is_file() and path.suffix in {".json", ".csv", ".png"}
                and "raw" not in relative.parts and "core_analysis" not in relative.parts):
            files[str(relative)] = path
    for path in (ROOT / "assets").rglob("*"):
        if path.is_file() and path.suffix in {".npz", ".png", ".pdf", ".md", ".txt"}:
            files[str(path.relative_to(ROOT))] = path
    code = ["experiments/__init__.py", "experiments/make_figures.py",
            "experiments/benchmark/__init__.py", "experiments/benchmark/settings.py",
            "experiments/benchmark/summarize.py", "popcontrast/__init__.py",
            "scripts/check_results.py", "scripts/core_results.py"]
    code += [str(p.relative_to(ROOT)) for p in (ROOT / "experiments/figures").glob("*.py")]
    for name in code:
        if (ROOT / name).is_file():
            files[name] = ROOT / name
    for path in compact.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(ROOT))] = path
    for name in ["PopContrast_WWW2027_updated_source.zip", "PopContrast_WWW2027_updated.pdf"]:
        path = ROOT / "reports/www2027" / name
        if path.exists():
            files["manuscript/" + name] = path
    readme = '''# PopContrast: local analysis archive

This archive contains the retained original results, 33 primary benchmark settings
over 8 datasets, the matched baselines and diagnostic controls, and compact
per-user predictions. It also includes the current manuscript source and PDF when
available. No GPU, server connection, model weights, training corpus, item
embeddings, full-catalog score matrices, or candidate caches are required.

## Rebuild and check on a local CPU

```sh
python -m pip install -r requirements.txt
python scripts/core_results.py verify results/core_analysis
python scripts/check_results.py
python -m experiments.benchmark.summarize
python -m experiments.figures.build_tables
python -m experiments.figures.estimator_stability --output-dir results/tables
python -m experiments.figures.publication
```

Figures are written to `results/figures/`;
tables are written to `results/tables/`. The original pink/blue style,
framework illustration, and aggregate inputs are included. The manuscript ZIP
under `manuscript/` has its own build instructions and needs a LaTeX installation.

## Result layout

- `results/benchmark/completed_results.csv`: primary metrics, frozen validation
  choices, checkpoint identities and paired intervals.
- `results/benchmark/runs/`: full metric grids, configurations, training histories,
  selection rules and audits; only the curated retained runs.
- `results/benchmark/estimator_resampling/`: retained finite-history sensitivity
  summaries. The large sampled-history score banks are intentionally excluded.
- `assets/plot_data/`: aggregate inputs for the original figures.
- `results/core_analysis/manifest.json`: dataset/run inventory and file checksums.
- `results/core_analysis/catalogs/`: popularity counts, head/tail masks and SID
  codes, without item text, embeddings or interaction histories.
- `results/core_analysis/predictions/`: all saved Top-10 methods for each retained
  partition, including the complete correction grids and matched baselines.
- `results/core_analysis/priors/`: available per-item correction vectors.

## Use compact predictions

```python
import numpy as np
path = 'results/core_analysis/predictions/beauty_s0_l3/evaluation/test_top10.npz'
with np.load(path, allow_pickle=False) as z:
    methods = z['methods'].tolist()
    raw = z['top10'][:, methods.index('raw'), :]
    corrected = z['top10'][:, methods.index('geometric:0.75'), :]
    targets = z['targets']
    raw_hit = (raw == targets[:, None]).any(axis=1)
    corrected_hit = (corrected == targets[:, None]).any(axis=1)
    history_length = z['history_length']
    history_head_share = z['history_head_count'] / history_length.clip(min=1)
```

`top10` has shape `(users, methods, 10)` with lossless int32 item IDs; `indices`
are anonymous row indices in the original evaluation split. Match users by these
indices, not by row position across different cohorts. Item IDs are specific to
each dataset. Together with the catalog masks, these arrays support recomputing
recall, NDCG, coverage, entropy, Gini, paired user bootstrap intervals, item-level
gained/lost hits and the recorded history/popularity groups. They do not support
fresh model inference, arbitrary new correction strengths or resampling a new
history prior. The saved original-study summaries and aggregate figures remain
available; only the retained expanded benchmark runs have compact predictions.

`LOCAL_VERIFICATION.json` records the actual archive rebuild checks.
'''
    with tempfile.TemporaryDirectory(prefix="popcontrast_core_package_") as temporary:
        work = Path(temporary)
        for name, source in files.items():
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (work / "README.md").write_text(readme)
        (work / "requirements.txt").write_text((ROOT / "requirements-analysis.txt").read_text())
        pngs = [name for name in files if name.startswith("results/figures/") and name.endswith(".png")]
        for name in pngs:
            (work / name).unlink()
        checks = [[sys.executable, "scripts/core_results.py", "verify", "results/core_analysis",
                   "--report", "ANALYSIS_VERIFICATION.json"],
                  [sys.executable, "scripts/check_results.py"],
                  [sys.executable, "-m", "experiments.benchmark.summarize"],
                  [sys.executable, "-m", "experiments.figures.build_tables"],
                  [sys.executable, "-m", "experiments.figures.estimator_stability", "--output-dir", "results/tables"],
                  [sys.executable, "-m", "experiments.figures.publication"]]
        for command in checks:
            subprocess.run(command, cwd=work, check=True)
        for name in pngs:
            assert sha(work / name) == sha(ROOT / name), name
        for name in ["completed_results.json", "completed_results.csv", "summary_sources.json",
                     "matched_baseline_comparisons.csv", "full_user_sensitivity.csv"]:
            relative = "results/benchmark/" + name
            assert sha(work / relative) == sha(ROOT / relative), relative
        report = {"status": "passed", "server_required": False, "weights_required": False,
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "identical_rebuilt_pngs": len(pngs), "identical_rebuilt_summaries": 5,
                  "compact_analysis": json.loads((work / "ANALYSIS_VERIFICATION.json").read_text())}
        (work / "LOCAL_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
        # Include regenerated tables, PDF plots and source manifests in the bundle.
        payload = [p for p in work.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        checksums = {str(p.relative_to(work)): sha(p) for p in sorted(payload)}
        (work / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
        payload.append(work / "CHECKSUMS.json")
        archive = args.output / "PopContrast_core_results.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(payload):
                z.write(path, str(path.relative_to(work)))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name, expected in checksums.items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        report.update(archive=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), files=len(payload))
        (args.output / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "README.md").write_text(readme)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
