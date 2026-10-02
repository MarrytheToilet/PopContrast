"""Preview Figure 10 as aligned, pairwise tradeoff panels.

Run ``python -m experiments.figures.tradeoff_matrix``. This reads the same
frozen test sweeps and validation choices as build_tables.py, without changing
the existing paper figure. No checkpoints or score caches are needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

from .sources import source_path
from .style import apply_reference, BG, BLUE, GOLD, INK, MUTE, PINK, TEXT_WIDTH, REFERENCE_LABEL

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/benchmark"
RUNS = [
    ("Beauty", "fair_baselines_beauty_seed0"),
    ("Sports", "fair_baselines_sports_seed1"),
    ("Toys", "fair_baselines_toys_seed1"),
]
METHODS = [
    ("arithmetic", "Arithmetic"),
    ("null", "Null context"),
    ("logcount", "Item frequency"),
    ("mmr", "MMR"),
    ("fusion", "SASRec fusion"),
]
RULE = "budget_5pct"


def read_evidence():
    sources = {}

    def load(relative):
        actual = source_path(BASE / relative)
        sources[str((BASE / relative).relative_to(ROOT))] = hashlib.sha256(
            actual.read_bytes()).hexdigest()
        return json.loads(actual.read_text())

    summary_path = source_path(BASE / "matched_baseline_comparisons.csv")
    sources[str((BASE / "matched_baseline_comparisons.csv").relative_to(ROOT))] = (
        hashlib.sha256(summary_path.read_bytes()).hexdigest())
    with summary_path.open() as handle:
        table = {(r["run"], r["family"], r["rule"]): r
                 for r in csv.DictReader(handle)}

    evidence = []
    for name, run in RUNS:
        metrics = load(Path("runs") / run / "test_metrics.json")
        selection = load(Path("runs") / run / "validation_selection.json")
        raw = metrics["raw"]
        curves = {}
        for family in ["geometric"] + [f for f, _ in METHODS]:
            keys = ["raw"] + sorted(
                (k for k in metrics if k.startswith(family + ":")),
                key=lambda k: float(k.split(":")[1]))
            selected = selection[family][RULE]
            assert selected in keys, (run, family, selected)
            points = []
            for key in keys:
                row = metrics[key]
                assert row["n_users"] == raw["n_users"] == 5000
                assert all(np.isfinite(row[k]) and 0 <= row[k] <= 1
                           for k in ["R10", "cov10", "tailR10"])
                points.append(dict(
                    key=key, selected=(key == selected),
                    R10=row["R10"], cov10=row["cov10"],
                    delta_recall_pp=100 * (row["R10"] - raw["R10"]),
                    delta_coverage_pp=100 * (row["cov10"] - raw["cov10"])))
            # Cross-check plotted selections against the manuscript's table source.
            summary = table[run, family, RULE]
            for metric in ["R10", "tailR10", "cov10"]:
                assert np.isclose(metrics[selected][metric], float(summary[metric]),
                                  rtol=0, atol=1e-12), (run, family, metric)
            curves[family] = points
        evidence.append(dict(dataset=name, run=run, n_users=raw["n_users"],
                             baseline=raw, curves=curves))
    return evidence, sources


def draw_curve(ax, points, color, marker, zorder):
    xs = [p["delta_recall_pp"] for p in points]
    ys = [p["delta_coverage_pp"] for p in points]
    ax.plot(xs, ys, color=color, marker=marker, markersize=1.5,
            markeredgewidth=.25, markeredgecolor=BG, linewidth=.8,
            linestyle="-" if marker == "o" else "--", zorder=zorder)
    chosen = next(p for p in points if p["selected"])
    ax.scatter(chosen["delta_recall_pp"], chosen["delta_coverage_pp"],
               marker=marker, s=10, facecolors=color, edgecolors=INK,
               linewidths=.5, zorder=zorder + 3)


def draw_matrix(evidence, output, stem='matched_tradeoffs_matrix'):
    apply_reference(plt)
    fig, axes = plt.subplots(3, 5, figsize=(TEXT_WIDTH, 2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.14, right=.99, bottom=.16, top=.86,
                        wspace=.13, hspace=.23)
    # Common delta scales make all panels directly comparable. Include every
    # strength, reversals, coverage losses, and raw origins without clipping.
    points = [p for d in evidence for curve in d["curves"].values() for p in curve]
    xs = [p["delta_recall_pp"] for p in points]
    ys = [p["delta_coverage_pp"] for p in points]
    xmin, xmax = np.floor(min(xs)), np.ceil(max(xs))
    ymin, ymax = min(-1., np.floor(min(ys))), np.ceil(max(ys) / 2) * 2
    signed = FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:+g}")

    for row, data in enumerate(evidence):
        for col, (family, label) in enumerate(METHODS):
            ax = axes[row, col]
            ax.set_xlim(xmin - .12, xmax + .12)
            ax.set_ylim(ymin, ymax)
            ax.set_axisbelow(True)
            ax.grid(False)
            ax.yaxis.grid(True, linewidth=.35)
            ax.axhline(0, color=MUTE, linewidth=.45, zorder=1)
            ax.axvline(0, color=MUTE, linewidth=.45, linestyle=(0, (2, 2)), zorder=1)
            ax.xaxis.set_major_locator(FixedLocator([xmin, 0, xmax]))
            ax.yaxis.set_major_locator(FixedLocator([0, 5, 10]))
            ax.xaxis.set_major_formatter(signed)
            ax.yaxis.set_major_formatter(signed)
            ax.tick_params(length=2, pad=2)
            draw_curve(ax, data["curves"][family], BLUE, "s", 3)
            draw_curve(ax, data["curves"]["geometric"], PINK, "o", 4)
            # The hollow star keeps a selected raw marker visible beneath it.
            ax.scatter(0, 0, marker="*", s=20, facecolor="none", edgecolor=GOLD,
                       linewidth=.7, zorder=10)
        pos=axes[row,0].get_position()
        fig.text(.068,(pos.y0+pos.y1)/2,data['dataset'],rotation=90,
                 va='center',ha='center',fontsize=REFERENCE_LABEL,color=INK)
    for ax,(_,label) in zip(axes[0],METHODS):
        pos=ax.get_position()
        fig.text((pos.x0+pos.x1)/2,pos.y1+.012,label,
                 va='bottom',ha='center',fontsize=REFERENCE_LABEL,color=INK)
    fig.text(.015, .46, r'$\Delta$Coverage@10 (pp)', rotation=90,
             va="center", ha="center", fontsize=REFERENCE_LABEL)
    fig.text(.54, .035, r'$\Delta$Overall Recall@10 (pp)',
             ha="center", fontsize=REFERENCE_LABEL)
    legend = [
        Line2D([], [], color=PINK, marker="o", ms=2, lw=.8, label="PopContrast"),
        Line2D([], [], color=BLUE, marker="s", ms=2, lw=.8, ls="--", label="Comparator"),
        Line2D([], [], color=INK, marker="o", mfc=BG, mec=INK, ms=3.3,
               lw=0, label="Validation-selected"),
        Line2D([], [], color=GOLD, marker="*", mfc="none", ms=4,
               lw=0, label="Raw beam"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.53, .975),
               ncol=4, borderaxespad=0, handlelength=1.3, columnspacing=1.1)
    fig.canvas.draw()
    assert all(not ax.get_title(loc) for ax in axes.flat for loc in ['left','center','right'])
    # Check the renderer, not just layout parameters, for clipped text.
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1, text.get_text()
    font_min=min(label.get_fontsize() for label in fig.findobj(matplotlib.text.Text)
                 if label.get_visible() and label.get_text())
    for extension in ["pdf", "png"]:
        kwargs={'metadata': {'CreationDate': None}} if extension=='pdf' else {}
        fig.savefig(output / f"{stem}.{extension}", dpi=300, **kwargs)
    plt.close(fig)
    return dict(xlim=[xmin - .12, xmax + .12], ylim=[ymin, ymax],
                units="percentage-point changes relative to each dataset's raw beam",
                panels=15, figure_inches=[TEXT_WIDTH, 2.75],titles=[],
                minimum_vector_font_pt=font_min,
                row_order=[d['dataset'] for d in evidence],
                column_order=[label for _,label in METHODS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figure10")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence, sources = read_evidence()
    layout = draw_matrix(evidence, args.output_dir)
    # Geometric is repeated to support pairwise visual comparisons, not treated
    # as five independent replications. Preserve each curve only once here.
    manifest = dict(
        sources=sources, selection_rule=RULE,
        selection_note="Recorded validation choices; no test-set selection or interpolation.",
        geometry_note="Straight segments follow tested strength order; no smoothing or Pareto filtering.",
        layout=layout, datasets=evidence,
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in args.output_dir.glob("matched_tradeoffs_matrix.*")})
    (args.output_dir / "source_values.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output_dir=str(args.output_dir), panels=15,
                         datasets=len(evidence), families_per_dataset=6,
                         all_strengths_preserved=True, selected_values_verified=True)))


if __name__ == "__main__":
    main()
