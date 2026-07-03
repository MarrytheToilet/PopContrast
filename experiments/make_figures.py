"""Publication figures for PopContrast — soft pink/blue palette, non-stale designs.

Renders to results/figures/*.png (PNG only):
  fig1_pareto_trajectory  : accuracy-vs-coverage PHASE-SPACE PATH as beta sweeps
                            (model rises up-right = free diversification; naive collapses).
  fig2_diversification    : coverage & entropy vs beta as an OPENING FAN
                            (model diversifies; naive degenerates).
  fig3_diagnostic_triad   : the "not representation / not search / IS distribution"
                            ruling-out schematic (conceptual, hand-laid).

Reads whatever results/main_panel_<split>.json exist. Run anywhere (no GPU).
"""
from __future__ import annotations
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager

RES = "/home/hanyu/research/PopSteer/results"
FIG = os.path.join(RES, "figures")
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


def load_panels():
    panels = {}
    for p in sorted(glob.glob(os.path.join(RES, "main_panel_*.json"))):
        split = os.path.basename(p)[len("main_panel_"):-len(".json")]
        panels[split] = json.load(open(p))
    return panels


def _series(panel, prior_key, metric):
    d = panel[prior_key]
    betas = sorted(float(k) for k in d)
    ys = [d[str(b) if str(b) in d else f"{b:g}"][metric] if (str(b) in d) else d[[k for k in d if float(k)==b][0]][metric] for b in betas]
    return betas, ys


def fig1_pareto(panels):
    """Accuracy-coverage trajectories, compact 1x3 for a single column."""
    splits = list(panels)
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({"font.size": 19, "axes.titlesize": 22, "axes.labelsize": 19,
                         "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16}):
        fig, axes = plt.subplots(1, len(splits), figsize=(10.6, 3.8), squeeze=False)
        for k_ax, (ax, sp) in enumerate(zip(axes[0], splits)):
            P = panels[sp]; base = P["baseline"]
            def path(prior):
                d = P[prior]; ks = sorted(d, key=float)
                xs = [base["R10"]] + [d[k]["R10"] for k in ks]
                ys = [base["cov10"]] + [d[k]["cov10"] for k in ks]
                bs = [0.0] + [float(k) for k in ks]
                return np.array(xs), np.array(ys), bs
            ax.axvline(base["R10"], color=MUTE, lw=1, ls=(0, (2, 3)), alpha=.5)
            ax.axhline(base["cov10"], color=MUTE, lw=1, ls=(0, (2, 3)), alpha=.5)
            for prior, col, lab in [("model_pmi", PINK, "PopContrast"),
                                    ("naive", BLUE, "naive discount")]:
                xs, ys, bs = path(prior)
                ax.plot(xs, ys, "-", color=col, lw=3.0, alpha=.95, zorder=3,
                        solid_capstyle="round", label=lab)
                ax.scatter(xs, ys, s=52, color=col, edgecolor="white", lw=1.3, zorder=4)
                if prior == "model_pmi":
                    for b, xi, yi in zip(bs, xs, ys):
                        if b in (1.0, 2.0):
                            ax.annotate(f"β={b:g}", (xi, yi), textcoords="offset points",
                                        xytext=(6, 5), fontsize=14, color=col, fontweight="bold")
            ax.scatter([base["R10"]], [base["cov10"]], s=130, marker="*",
                       color=GOLD, edgecolor="white", lw=1.3, zorder=5)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("overall R@10")
            if k_ax == 0:
                ax.set_ylabel("Coverage@10")
                ax.legend(loc="upper left", frameon=False, handlelength=1.4)
            ax.set_title(sp.capitalize(), color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig1_pareto_trajectory.png"), bbox_inches="tight")
        plt.close(fig)


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


def load_figdata():
    fd = {}
    for p in sorted(glob.glob(os.path.join(RES, "figdata_*.npz"))):
        split = os.path.basename(p)[len("figdata_"):-len(".npz")]
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


def fig5_head_tail_slope(panels):
    """Slopegraph: head vs tail Recall@10 redistribution as β increases."""
    splits = list(panels)
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2*len(splits), 4.6), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        P = panels[sp]; d = P["model_pmi"]; base = P["baseline"]
        ks = sorted(d, key=float); betas = [0.0] + [float(k) for k in ks]
        heads = [base["headR10"]] + [d[k]["headR10"] for k in ks]
        tails = [base["tailR10"]] + [d[k]["tailR10"] for k in ks]
        bmax = max(betas)
        for b, h, t in zip(betas, heads, tails):
            shade = 0.25 + 0.6 * (b / bmax)
            col = PINK if b > 0 else MUTE
            ax.plot([0, 1], [h, t], "-", color=col, alpha=shade, lw=2.2,
                    zorder=3 if b > 0 else 2)
            ax.scatter([0, 1], [h, t], s=30, color=col, alpha=shade, zorder=4, ec="white", lw=1)
            if b in (0.0, bmax):
                ax.annotate(f"β={b:g}", (1, t), textcoords="offset points", xytext=(8, 0),
                            fontsize=9.5, color=col, va="center")
        ax.set_xlim(-0.25, 1.4); ax.set_xticks([0, 1]); ax.set_xticklabels(["head", "tail"])
        ax.set_ylabel("Recall@10")
        ax.set_title(sp.capitalize(), color=INK)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Recall redistribution: head ↓ tail ↑ as β increases (darker = stronger β)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_head_tail_slope.png"), bbox_inches="tight")
    plt.close(fig)


def fig7_tokenizer(split="beauty"):
    """Bias BYPASSES the tokenizer: code-density ⊥ popularity, but marginal ↔ popularity."""
    from scipy.stats import spearmanr
    from matplotlib.colors import LinearSegmentedColormap
    p = os.path.join(RES, f"tokdata_{split}.npz")
    if not os.path.exists(p):
        return
    d = np.load(p); lp, dens, mg = d["logpop"], d["density"], d["marginal"]
    cmap = LinearSegmentedColormap.from_list("pinkblue", [BLUE_GLOW, BLUE_SOFT, PINK, "#B83A6B"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, y, ylab, title in [(a1, dens, "code density (tokenizer)", "Tokenizer: code density"),
                               (a2, mg, "model marginal (decoder)", "Decoder: marginal")]:
        ax.hexbin(lp, y, gridsize=34, cmap=cmap, mincnt=1, linewidths=0.15, edgecolors=BG)
        z = np.polyfit(lp, y, 1); xs = np.linspace(lp.min(), lp.max(), 40)
        ax.plot(xs, np.polyval(z, xs), color=INK, lw=2, ls=(0, (4, 2)), alpha=.8)
        rho, _ = spearmanr(lp, y)
        col = MUTE if abs(rho) < 0.2 else "#B83A6B"
        ax.text(0.05, 0.92, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=14,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PINK_SOFT))
        ax.set_xlabel("log item popularity"); ax.set_ylabel(ylab); ax.set_title(title, color=INK)
    fig.suptitle(f"Popularity bias BYPASSES the tokenizer ({split.capitalize()}): "
                 "code assignment ⊥ popularity, but the decoder marginal tracks it",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_tokenizer_bypass.png"), bbox_inches="tight")
    plt.close(fig)


def fig8_rankshift(path=os.path.join(RES, "rankshift_beauty.npz")):
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


def fig9_quintiles(path=os.path.join(RES, "enrich_analysis.json")):
    """Recall@10 by popularity quintile (q0 least → q4 most popular) as β grows —
    a redistribution ladder, finer than the head/tail split."""
    if not os.path.exists(path):
        return
    D = json.load(open(path))
    splits = [s for s in ("beauty", "sports", "toys") if s in D]
    betas = ["0.0", "0.5", "1.0"]
    shades = [MUTE, PINK_SOFT, PINK]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6*len(splits), 4.4), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        Q = D[sp]["quintile_recall"]
        x = np.arange(5)
        for b, col in zip(betas, shades):
            key = b if b in Q else str(float(b))
            ys = Q[key]["by_quintile"]
            ax.plot(x, ys, "-o", color=col, lw=2.4, ms=6, mec="white", mew=1.2,
                    label=fr"$\beta$={float(b):g}")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(["q1\n(least pop.)", "q2", "q3", "q4", "q5\n(most pop.)"], fontsize=9)
        ax.set_ylabel("Recall@10 (log scale)")
        ax.set_title(sp.capitalize(), color=INK)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("Recall by popularity quintile: gains concentrate on the least-popular items "
                 "while the top quintile degrades gracefully", fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_quintile_ladder.png"), bbox_inches="tight")
    plt.close(fig)


PINK_RAMP = ["#FBE6EE", "#F6C6D8", "#EE9DBD", "#E06C97", "#B83A6B"]  # q1(light)→q5(dark)
BLUE_RAMP = ["#EAF3FB", "#CFE2F3", "#A9CBE8", "#7FAED9", "#5B93C9"]  # q1(light)→q5(soft blue)


def fig10_exposure_stream(path=os.path.join(RES, "exposure_shares.json")):
    """Stacked-area exposure stream, 1x3 sized for a single column."""
    if not os.path.exists(path):
        return
    D = json.load(open(path))
    splits = list(D)
    with plt.rc_context({"font.size": 20, "axes.titlesize": 23, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 18}):
        fig, axes = plt.subplots(1, len(splits), figsize=(10.4, 3.3), squeeze=False, sharey=True)
        for k_ax, (ax, sp) in enumerate(zip(axes[0], splits)):
            bs = D[sp]["betas"]; sh = np.array(D[sp]["shares"])
            order = [4, 3, 2, 1, 0]
            ys = sh[:, order].T
            ax.stackplot(bs, ys, colors=[BLUE_RAMP[q] for q in order],
                         edgecolor=BG, linewidth=1.4)
            cum = np.cumsum(ys, axis=0)
            for k, q in enumerate(order):
                y_mid = (cum[k, -1] + (cum[k-1, -1] if k else 0)) / 2
                if ys[k, -1] > 0.05:
                    ax.text(bs[-1]*0.98, y_mid, f"q{q+1}", ha="right", va="center",
                            fontsize=17, fontweight="bold",
                            color=("white" if q == 4 else INK))
            ax.set_xlim(bs[0], bs[-1]); ax.set_ylim(0, 1)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xlabel(r"$\beta$")
            if k_ax == 0:
                ax.set_ylabel("exposure share")
            ax.set_title(sp.capitalize(), color=INK)
            ax.grid(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig10_exposure_stream.png"), bbox_inches="tight")
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


def fig2b_delta_board(panels):
    """Diverging Δ-board (replaces the dual-axis chart): signed %Δ vs baseline for
    each metric at the near-free operating point. Pink = up, blue = down."""
    ops = {"beauty": "0.25", "sports": "0.5", "toys": "0.5"}
    metrics = [("tailR10", "Tail R@10"), ("cov10", "Coverage@10"), ("ent", "Entropy"),
               ("R10", "Overall R@10"), ("headR10", "Head R@10"), ("gini", "Gini (↓ better)")]
    splits = [s for s in ops if s in panels]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.4*len(splits), 3.7), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        P = panels[sp]; base = P["baseline"]; cur = P["model_pmi"][ops[sp]]
        vals = [100*(cur[k]-base[k])/base[k] for k, _ in metrics]
        y = np.arange(len(metrics))[::-1]
        cols = [PINK if v >= 0 else BLUE for v in vals]
        ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", linewidth=1.2)
        ax.axvline(0, color=INK, lw=1.2)
        for yi, v in zip(y, vals):
            ax.text(v + (2.5 if v >= 0 else -2.5), yi, f"{v:+.0f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([n for _, n in metrics], fontsize=9.5)
        ax.set_xlabel("% change vs. baseline")
        ax.set_title(f"{sp.capitalize()}  (β={ops[sp]})", color=INK)
        lim = max(25, max(abs(v) for v in vals) * 1.3)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Near-free operating points: what moves, by how much",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_delta_board.png"), bbox_inches="tight")
    plt.close(fig)


def fig6_d3(path=os.path.join(RES, "d3_comparison.json")):
    """Method comparison in (tail Recall@10, Coverage@10) space — ours up-right, D3 down-left."""
    if not os.path.exists(path):
        return
    D = json.load(open(path)); splits = [s for s in D if not s.startswith("_")]
    style = {"baseline": (GOLD, "*", "baseline (TIGER)"),
             "ours_b0.75": (PINK, "o", "PopContrast (ours, no ext. model)"),
             "d3_a0.7": (BLUE, "s", "D³ (TIGER+SASRec fusion)"),
             "d3+ours": ("#9B6FB0", "D", "D³ + ours")}
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6*len(splits), 4.5), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        P = D[sp]
        for key, (col, mk, lab) in style.items():
            if key not in P:
                continue
            ax.scatter(P[key]["tailR10"], P[key]["cov10"], s=170, marker=mk, color=col,
                       edgecolor="white", lw=1.6, zorder=4, label=lab if sp == splits[0] else None)
        # baseline reference cross
        b = P["baseline"]
        ax.axvline(b["tailR10"], color=MUTE, lw=1, ls=(0, (2, 3)), alpha=.4)
        ax.axhline(b["cov10"], color=MUTE, lw=1, ls=(0, (2, 3)), alpha=.4)
        ax.set_xlabel("tail Recall@10  (→ better)")
        ax.set_ylabel("Coverage@10  (↑ better)")
        ax.set_title(sp.capitalize(), color=INK)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.suptitle("Ours vs D³: only PopContrast moves up-right (more tail + coverage), "
                 "and needs no external model", fontsize=14.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_vs_d3.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    panels = load_panels(); fd = load_figdata()
    print(f"datasets: {list(panels)}")
    fig1_pareto(panels)
    fig2_diversification(panels)
    if fd:
        fig3_lorenz(fd)
        fig4_marginal_pop(fd)
    fig5_head_tail_slope(panels)
    fig6_d3()
    fig7_tokenizer()
    fig8_rankshift()
    fig9b_quintile_heatmap()
    fig10_exposure_stream()
    fig2b_delta_board(panels)
    print(f"figures -> {FIG}")
    for f in sorted(glob.glob(os.path.join(FIG, "*.png"))):
        print("  ", os.path.basename(f))
