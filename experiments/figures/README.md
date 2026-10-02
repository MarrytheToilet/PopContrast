# Paper figures and tables

All plots are rebuilt from local experimental records and aggregate arrays. Restore `results/` and `assets/plot_data/` from the separately retained analysis archive first; both directories are ignored and absent from a Git clone. The analysis environment needs no GPU, model checkpoint, or private manuscript:

```bash
pip install -r requirements-analysis.txt
make summaries
make figures
make check
```

Default outputs are `results/figures/` and `results/tables/`. To build individual components:

```bash
python -m experiments.figures.build_tables
python -m experiments.figures.estimator_stability --output-dir results/tables
python -m experiments.figures.publication
```

## Figure map

| Paper figure | Output stem | Evidence |
|---|---|---|
| 1 | `marginal_single_column` | Popularity and cross-history scores on Beauty, Sports, and Toys |
| 2 | `popcontrast_overview` | Schematic offline-reference / online-correction workflow |
| 3 | `pareto_single_column` | Full-catalog accuracy–coverage sweeps |
| 4 | `exposure_single_column` | Exposure shares across popularity quintiles |
| 5 | `quintile_single_column` | Recall changes by popularity quintile |
| 6 | `rankshift_single_column` | Mean item-rank changes on Beauty |
| 7 | `model_seed_evidence` | Four LLM configurations and three matched seeds on Beauty/Clothing |
| 8 | `lorenz_single_column` | Exposure Lorenz curves |
| 9 | `diversification_single_column` | Coverage and exposure entropy trajectories |
| 10 | `matched_tradeoffs_compact` | Matched beam comparisons across the recorded strength grids |

Each stem has PNG and PDF outputs. Figure 2 copies the supplied artwork in `assets/diagrams/` byte for byte; rebuilding does not invoke image generation. The README teaser `assets/pr.png` is a separate schematic illustration, with its prompts in `assets/pr.prompt.txt`.

The six original diagnostic renderers are shared with `experiments/make_figures.py`. Their local aggregate NPZ inputs are under `assets/plot_data/`. Model/seed plots and matched-baseline trajectories use local `results/benchmark/` records. Plotting preserves every recorded sweep point in parameter order and does not select a test-derived frontier or smooth results. Only two curated README previews are versioned under `assets/readme/`; the generated figure directory is ignored.

## Tables and provenance

`build_tables.py` reconstructs the generative comparisons, matched baselines, candidate recovery, ISD composition, seed/cohort summaries, and appendix settings. `estimator_stability.py` separately recomputes the M=32 means and sample standard deviations from 20 recorded subsets per dataset, including each subset's validation-selected strength.

Matched baseline tables include overall recall, NDCG, tail recall, coverage, normalized entropy, and Gini. Geometric and arithmetic references are both identified as PopContrast variants. Best and second-best displayed values are marked with ties sharing rank; lower Gini is better. All displayed gains are calculated from unrounded source values. Complete baseline metrics are retained in `baselines_full.tex`.

The generated `results/asset_sources.json` and `results/figure_sources.json` record input hashes and displayed values. The figure manifest also records dimensions, font sizes, panel identities, and output hashes. The publication renderer checks text bounds and keeps dataset labels separate from legends and axes.

Set `POPCONTRAST_PUBLICATION_DIR` to an existing manuscript directory to direct publication outputs there. This is optional; the public repository's default build does not depend on a manuscript checkout.

For an independent preview of Figure 10, run `python -m experiments.figures.tradeoff_matrix`. It writes a preview and `source_values.json` under the ignored `reports/figure10/` directory. The selected markers follow the stored validation rule; its budget is not a guarantee of test loss.
