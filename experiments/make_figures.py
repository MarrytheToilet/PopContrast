"""Publication figures for PopContrast — soft pink/blue palette.

Renders results/figures/*.png (300 dpi) from the JSON results and figdata npz files;
missing figdata_<split>.npz are rebuilt from cache_scores_<split>.pt automatically
(prepare_figdata; needs torch + dataset, otherwise skipped gracefully).

Uses the paper's three main splits by default; set FIG_SPLITS=beauty,clothing,...
to change. Figure-only runs on existing results work anywhere (no GPU).
"""
from __future__ import annotations
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager
from pathlib import Path

from popcontrast import RESULTS_DIR as RES
FIG = os.path.join(RES, "figures")
PLOT_DATA = Path(__file__).resolve().parents[1] / "assets/plot_data"
os.makedirs(FIG, exist_ok=True)

# ---- soft palette (light pink + light blue) ----
PINK   = "#E06C97"; PINK_SOFT = "#F6C6D8"; PINK_GLOW = "#FBE6EE"
BLUE   = "#5B93C9"; BLUE_SOFT = "#C6DCF0"; BLUE_GLOW = "#E7F0FA"
INK    = "#39323F"; MUTE = "#8A8494"; BG = "#FDFCFE"; GRID = "#EDE9F1"
GOLD   = "#E8B04B"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#D9D3E0", "axes.linewidth": 1.1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 14.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "legend.fontsize": 12.5, "lines.linewidth": 2.8,
    "font.family": "DejaVu Sans", "figure.dpi": 300, "savefig.dpi": 300,
})


# Paper figures use the three main splits; Clothing is an appendix replication.
# Override with e.g. FIG_SPLITS=beauty,clothing,sports,toys to include it.
FIG_SPLITS = os.environ.get("FIG_SPLITS", "beauty,sports,toys").split(",")


def load_panels():
    panels = {}
    for p in sorted(glob.glob(os.path.join(RES, "main_panel_*.json"))):
        split = os.path.basename(p)[len("main_panel_"):-len(".json")]
        if split in FIG_SPLITS:
            panels[split] = json.load(open(p))
    return panels


def _series(panel, prior_key, metric):
    d = panel[prior_key]
    betas = sorted(float(k) for k in d)
    ys = [d[str(b) if str(b) in d else f"{b:g}"][metric] if (str(b) in d) else d[[k for k in d if float(k)==b][0]][metric] for b in betas]
    return betas, ys




def fig2_diversification(panels):
    splits = list(panels)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.4*len(splits), 4.6), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        P = panels[sp]; base = P["baseline"]
        d = P["model_pmi"]; ks = sorted(d, key=float)
        bs = [0.0] + [float(k) for k in ks]
        cov = [base["cov10"]] + [d[k]["cov10"] for k in ks]
        ent = [base["ent"]] + [d[k]["ent"] for k in ks]
        ax.fill_between(bs, cov, base["cov10"], color=PINK_GLOW, zorder=1)
        ax.plot(bs, cov, "-o", color=PINK, lw=2.6, ms=6, mec="white", mew=1.3,
                label="Coverage@10", zorder=3)
        ax2 = ax.twinx(); ax2.grid(False); ax2.spines["top"].set_visible(False)
        ax2.plot(bs, ent, "-s", color=BLUE, lw=2.4, ms=5, mec="white", mew=1.2,
                 label="Norm. entropy", zorder=3)
        ax2.set_ylabel("recommendation entropy", color=BLUE)
        ax2.tick_params(axis="y", colors=BLUE)
        ax.set_xlabel("β  (debiasing strength)")
        ax.set_ylabel("Coverage@10", color=PINK)
        ax.tick_params(axis="y", colors=PINK)
        ax.set_title(sp.capitalize(), color=INK)
    fig.suptitle("Genuine diversification: coverage and entropy both rise with β",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_diversification.png"), bbox_inches="tight")
    plt.close(fig)


def prepare_figdata(beta=1.0):
    """Regenerate any missing figdata_<split>.npz from cache_scores_<split>.pt.

    Per split: per-item log-popularity, model marginal, top-10 exposure counts at
    baseline and at model-PMI beta, and the tail mask. Needs torch + the dataset
    (run from genrec/ root); skipped gracefully when unavailable so that
    figure-only runs on existing figdata still work anywhere.
    """
    caches = sorted(glob.glob(os.path.join(RES, "cache_scores_*.pt")))
    missing = [p for p in caches
               if not os.path.exists(os.path.join(
                   RES, f"figdata_{os.path.basename(p)[len('cache_scores_'):-len('.pt')]}.npz"))
               and not (PLOT_DATA / f"figdata_{os.path.basename(p)[len('cache_scores_'):-len('.pt')]}.npz").exists()
               and os.path.basename(p)[len('cache_scores_'):-len('.pt')] in FIG_SPLITS]
    if not missing:
        return
    try:
        import torch
        from popcontrast.data_utils import build_dataset, compute_popularity
    except ImportError as e:
        print(f"[figdata] {len(missing)} split(s) missing but deps unavailable ({e}); skipping")
        return
    for cache_path in missing:
        split = os.path.basename(cache_path)[len("cache_scores_"):-len(".pt")]
        print(f"[figdata] building figdata_{split}.npz from cache", flush=True)
        cache = torch.load(cache_path)
        scores = cache["scores"]                 # (N, I) cpu
        marginal = cache["marginal"]             # (I,)
        ds = build_dataset(split=split, train_test_split="train")
        pop = compute_popularity(ds)
        I = scores.shape[1]
        prior = (marginal - marginal.mean()) / marginal.std().clamp_min(1e-6)
        base_top = scores.topk(10, dim=1).indices.reshape(-1)
        beta_top = (scores - beta * prior[None, :]).topk(10, dim=1).indices.reshape(-1)
        np.savez(os.path.join(RES, f"figdata_{split}.npz"),
                 log_pop=np.log1p(pop.counts[:I]), marginal=marginal.numpy(),
                 cnt_base=np.bincount(base_top.numpy(), minlength=I),
                 cnt_beta=np.bincount(beta_top.numpy(), minlength=I),
                 tail_mask=(pop.bucket[:I] == "tail"), beta=beta)


def load_figdata():
    fd = {}
    for p in sorted(glob.glob(str(PLOT_DATA / "figdata_*.npz"))) + sorted(glob.glob(os.path.join(RES, "figdata_*.npz"))):
        split = os.path.basename(p)[len("figdata_"):-len(".npz")]
        if split in FIG_SPLITS:
            fd[split] = np.load(p)
    return fd


def _lorenz(counts):
    x = np.sort(counts.astype(float))
    c = np.cumsum(x)
    c = c / c[-1] if c[-1] > 0 else c
    n = len(x)
    xs = np.arange(1, n + 1) / n
    return np.concatenate([[0], xs]), np.concatenate([[0], c])


def fig3_lorenz(fd):
    """Item-exposure Lorenz curves: baseline vs debiased. Closer to diagonal = fairer."""
    splits = list(fd)
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6*len(splits), 4.5), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        d = fd[sp]; beta = float(d["beta"])
        ax.plot([0, 1], [0, 1], color=MUTE, lw=1.2, ls=(0, (3, 3)), alpha=.6, zorder=1)
        xb, yb = _lorenz(d["cnt_base"]); xd, yd = _lorenz(d["cnt_beta"])
        ax.fill_between(xb, yb, xd, color=PINK_GLOW, zorder=1)
        ax.plot(xb, yb, color=BLUE, lw=2.6, label="baseline", zorder=3)
        ax.plot(xd, yd, color=PINK, lw=2.6, label=f"PopContrast (β={beta:g})", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("items (least → most exposed)")
        ax.set_ylabel("cumulative recommendation share")
        ax.set_title(sp.capitalize(), color=INK)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=10.5)
    fig.suptitle("Exposure fairness (Lorenz): debiasing pulls the curve toward the diagonal",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_exposure_lorenz.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_marginal_pop(fd):
    """Marginal vs popularity hexbin, compact 1x3 for a single column."""
    from scipy.stats import spearmanr
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pinkblue", [BLUE_GLOW, BLUE_SOFT, PINK, "#B83A6B"])
    splits = list(fd)
    with plt.rc_context({"font.size": 19, "axes.titlesize": 22, "axes.labelsize": 19,
                         "xtick.labelsize": 15, "ytick.labelsize": 15}):
        fig, axes = plt.subplots(1, len(splits), figsize=(10.4, 3.7), squeeze=False)
        for k_ax, (ax, sp) in enumerate(zip(axes[0], splits)):
            d = fd[sp]; lp = d["log_pop"]; mg = d["marginal"]
            ax.hexbin(lp, mg, gridsize=30, cmap=cmap, mincnt=1, linewidths=0.15, edgecolors=BG)
            z = np.polyfit(lp, mg, 1); xs = np.linspace(lp.min(), lp.max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=INK, lw=2.4, ls=(0, (4, 2)), alpha=.85)
            rho, _ = spearmanr(mg, lp)
            ax.text(0.05, 0.90, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=17,
                    fontweight="bold", color="#B83A6B",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PINK_SOFT))
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("log item popularity")
            if k_ax == 0:
                ax.set_ylabel("marginal score")
            ax.set_title(sp.capitalize(), color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig4_marginal_vs_popularity.png"), bbox_inches="tight")
        plt.close(fig)






def fig8_rankshift(path=None):
    path = path or (Path(RES) / "rankshift_beauty.npz" if (Path(RES) / "rankshift_beauty.npz").exists() else PLOT_DATA / "rankshift_beauty.npz")
    """Mechanism view: per-item mean rank change under PopContrast (β=0.75) vs
    popularity. Positive Δrank = demoted. Shows the correction acts as a smooth,
    popularity-proportional re-ranking — not an indiscriminate shuffle."""
    if not os.path.exists(path):
        return
    from matplotlib.colors import LinearSegmentedColormap
    d = np.load(path); lp, dr = d["logpop"], d["drank"]
    cmap = LinearSegmentedColormap.from_list("pinkblue", [BLUE_GLOW, BLUE_SOFT, PINK, "#B83A6B"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hexbin(lp, dr, gridsize=42, cmap=cmap, mincnt=1, linewidths=0.15, edgecolors=BG)
    ax.axhline(0, color=INK, lw=1.4, ls=(0, (4, 2)), alpha=.7)
    # binned median curve
    bins = np.quantile(lp, np.linspace(0, 1, 13))
    mids, meds = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (lp >= a) & (lp <= b)
        if m.sum() > 20:
            mids.append((a + b) / 2); meds.append(np.median(dr[m]))
    ax.plot(mids, meds, "-o", color=INK, lw=2.4, ms=5, mec="white", mew=1.2,
            label="median shift (binned)")  # noqa
    ax.annotate("head items demoted", xy=(0.97, 0.94), xycoords="axes fraction",
                ha="right", color="#B83A6B", fontsize=13.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
    ax.annotate("tail items promoted", xy=(0.03, 0.06), xycoords="axes fraction",
                ha="left", color=BLUE, fontsize=13.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
    ax.set_xlabel("log item popularity")
    ax.set_ylabel("mean rank change  (+ = demoted)")
    ax.set_title("Rank shift vs. popularity (Beauty)", color=INK)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_rankshift_mechanism.png"), bbox_inches="tight")
    plt.close(fig)


def fig9b_quintile_heatmap(path=os.path.join(RES, "enrich_analysis.json")):
    """Diverging heatmap of %-change Recall@10 per (quintile x beta), 1x3 single-column."""
    if not os.path.exists(path):
        return
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("divpb", [BLUE, "#EFEDF2", PINK])
    D = json.load(open(path))
    splits = [s for s in ("beauty", "sports", "toys") if s in D]
    betas = ["0.25", "0.5", "0.75", "1.0"]
    with plt.rc_context({"font.size": 19, "axes.titlesize": 22,
                         "xtick.labelsize": 16.5, "ytick.labelsize": 17}):
        fig, axes = plt.subplots(1, len(splits), figsize=(10.4, 2.9), squeeze=False)
        for k_ax, (ax, sp) in enumerate(zip(axes[0], splits)):
            Q = D[sp]["quintile_recall"]
            base = Q["0.0"]["by_quintile"]
            grid = np.full((5, len(betas)), np.nan)
            fromzero = np.zeros((5, len(betas)), dtype=bool)
            for j, b in enumerate(betas):
                cur = Q[b]["by_quintile"]
                for q in range(5):
                    if base[q] and base[q] > 0:
                        grid[q, j] = 100 * (cur[q] - base[q]) / base[q]
                    elif cur[q] and cur[q] > 0:
                        grid[q, j] = 200.0; fromzero[q, j] = True
            norm = TwoSlopeNorm(vmin=-200, vcenter=0, vmax=200)
            ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
            for q in range(5):
                for j in range(len(betas)):
                    v = grid[q, j]
                    if np.isnan(v): txt, col = "-", MUTE
                    elif fromzero[q, j]: txt, col = "0→>0", "white"
                    else:
                        txt = f"{min(v,200):+.0f}%"
                        col = "white" if abs(min(v, 200)) > 120 else INK
                    ax.text(j, q, txt, ha="center", va="center", fontsize=13,
                            color=col, fontweight="bold")
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([".25", ".5", ".75", "1"])
            ax.set_xlabel(r"$\beta$")
            if k_ax == 0:
                ax.set_yticks(range(5)); ax.set_yticklabels(["q1", "q2", "q3", "q4", "q5"])
                ax.set_ylabel("popularity quintile", fontsize=17)
            else:
                ax.set_yticks(range(5)); ax.set_yticklabels([])
            ax.set_title(sp.capitalize(), color=INK)
            ax.grid(False)
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig9_quintile_heatmap.png"), bbox_inches="tight")
        plt.close(fig)






def fig10_exposure_stream():
    """Exposure shares by training-popularity quintile from frozen counts."""
    data = json.loads((Path(RES) / 'exposure_shares.json').read_text())
    splits = [s for s in FIG_SPLITS if s in data]
    with plt.rc_context({'font.size': 15, 'axes.labelsize': 15,
                         'axes.titlesize': 18, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13}):
        fig, axes = plt.subplots(1, len(splits), figsize=(10.5, 3.05), sharey=True)
        colors = [BLUE, '#8CB6DD', '#B6D0EA', '#D6E5F3', BLUE_GLOW]
        for ax, split in zip(np.atleast_1d(axes), splits):
            beta = data[split]['betas']
            shares = np.asarray(data[split]['shares'])
            assert shares.shape == (len(beta), 5)
            assert np.allclose(shares.sum(1), 1)
            ax.stackplot(beta, *shares[:, ::-1].T, colors=colors,
                         edgecolor='white', linewidth=.8)
            cumulative = np.cumsum(shares[-1, ::-1])
            centers = cumulative - shares[-1, ::-1] / 2
            for q, y in zip(range(5, 0, -1), centers):
                ax.text(beta[-1] * .96, y, f'q{q}', ha='right', va='center',
                        color='white' if q == 5 else INK, fontsize=10, fontweight='bold')
            ax.set(xlim=(beta[0], beta[-1]), ylim=(0, 1), xlabel=r'$\beta$')
            ax.set_title(split.capitalize())
            ax.grid(visible=False)
        np.atleast_1d(axes)[0].set_ylabel('Top-10 exposure share')
        fig.tight_layout(pad=.6)
        fig.savefig(Path(FIG) / 'fig10_exposure_stream.png', bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    prepare_figdata()
    panels = load_panels(); fd = load_figdata()
    print(f"datasets: {list(panels)}")
    fig2_diversification(panels)
    if fd:
        fig3_lorenz(fd)
        fig4_marginal_pop(fd)
    fig8_rankshift()
    fig9b_quintile_heatmap()
    fig10_exposure_stream()
    print(f"figures -> {FIG}")
    for f in sorted(glob.glob(os.path.join(FIG, "*.png"))):
        print("  ", os.path.basename(f))
