# Matched recommendation benchmark

All commands run from the repository root. New models and large prediction caches are written under ignored `runs/`; compact frozen records live under `results/benchmark/runs/`. The primary inventory is defined in `settings.py` and includes every reported checkpoint, including weak or unchanged outcomes.

## Data and model settings

Beauty, Sports, Toys, and Clothing preserve the original item mapping, RQ-VAE IDs, and chronological validation/test split. `export_data.py` checks exported targets and head/tail labels against the original score caches. `prepare_extended.py` builds MovieLens-1M and Amazon Reviews 2023 Games, Arts, and Books using the documented training-only encoding pipeline.

`run.py --list` lists the 33 primary settings. `run.py --run <setting> --stage train --print-command` reconstructs the configuration from its recorded training metadata. The three matched seeds use the same length-grouping protocol; separate padded controls remain identified. Three-, four-, five-, and six-token IDs preserve their recorded encoding definitions. The diagnostic checkpoint's training seed is unrecorded and is not invented by the runner.

Training budgets, checkpoint selection, adapter configurations, dataset hashes, and frozen evaluation strengths remain with each run. Qwen1.5B completes four epochs with a six-epoch scheduler horizon; `--stop-after-epochs` preserves that distinction. Evaluation explicitly uses FP32 for all four reported LLM settings, with repetition/no-repeat penalties disabled and complete-SID scores checked against direct teacher forcing.

## Paired evaluation

`evaluate.py` and `lcrec_evaluate.py` compare scores on identical validation/test users and width-20 candidate sets. The three validation rules are no overall-recall loss, a 5% relative loss budget, and a simultaneous-bootstrap-bound budget. Each includes raw ranking. Select tail recall first, then overall recall, then weaker correction. Fixed beta 0.75 is also reported. User intervals use paired resampling conditional on a checkpoint; training-seed SD is a separate statistic.

## Baselines and controls

The following examples assume completed training/evaluation caches under `runs/`:

```bash
# Tune priors, MMR, and auxiliary-model fusion on the same candidates.
python -m experiments.benchmark.rerank_baselines --split beauty \
  --checkpoint runs/beauty_s0_l3/best_model.pt \
  --assistant-checkpoint runs/sasrec_beauty_s1/best_model.pt \
  --prior-file runs/beauty_s0_l3/evaluation/priors.npz \
  --candidate-source runs/beauty_s0_l3/evaluation \
  --output runs/fair_baselines_beauty_seed0

# Extend the user cohort with frozen model, prior, and validation choices.
python -m experiments.benchmark.evaluate_full_users \
  --source runs/clothing_s0_l3/evaluation --output runs/full_users_clothing_s0

# Evaluate wider beams on the same sampled users.
python -m experiments.benchmark.evaluate_width_sweep \
  --source runs/beauty_s1_l4/evaluation --output runs/beauty_s1_l4_widths

# Finite-history-bank sensitivity; repeat for each retained checkpoint.
python -m experiments.benchmark.estimator_resampling --split beauty \
  --cache-dir runs/beauty_s1_l3_fast/evaluation \
  --background-dir runs/beauty_s1_l3_fast/evaluation \
  --output runs/beauty_history_sampling

# Measure request latency with an otherwise idle selected GPU.
python -m experiments.benchmark.benchmark --split beauty \
  --checkpoint checkpoints/legacy/beauty.pt --output runs/latency_beauty_b20
```

`evaluate_lookahead.py` retains the terminal-correction and search-potential controls. `audit_legacy_precision.py` and `analyze_legacy_precision.py` compare the original decoder with shared-score replay. `audit_llm_scores.py` and `audit_batched_search.py` check cached/direct and batched/single-history numerical agreement. These use validation examples for numerical diagnostics.

`audit_result_integrity.py` independently reconstructs recommendation metrics from saved per-user top-10 arrays. `audit_selected_estimators.py` compares frozen estimator selections and user segments. `audit_candidate_relevance.py` separates candidate availability from ranking within candidates. `audit_width_results.py` checks shared users, item identities, metrics, and inclusion/overlap relationships. `diagnose_prior_coupling.py` and `audit_classical_prior.py` quantify the retained model references.

The underlying prediction arrays and checkpoints are regenerable runtime artifacts and are ignored. The repository includes compact metric/configuration records, including all nine finite-history-bank sensitivity studies, all three full Clothing cohorts, longer-ID controls, and the measured latency study.

`python -m experiments.benchmark.compare_reference_variants` summarizes the
geometric and arithmetic estimator variants without retraining or selecting on
test results. It verifies score-bank hashes, computes the concentration of each
history's influence on the arithmetic log-reference, checks the Jensen-gap
identity, and summarizes the recorded 32-history resampling results. The compact
output is `results/benchmark/reference_variant_analysis.json`. Recomputing the
influence analysis requires the local `sampled_history_scores.npz` caches;
unavailable or mismatched banks are explicitly excluded. These influence weights
describe derivatives with respect to log-scores, not user relevance or statistical
effective sample sizes.

## Summaries and figures

```bash
python -m experiments.benchmark.summarize
python scripts/check_results.py
python -m experiments.figures.build_tables
python -m experiments.figures.publication
```

The summaries contain all five operating rules for the 33 primary settings, 63 matched baseline rows, and 24 cohort rows. Only complete, distinct primary checkpoints enter the inventory. Encodings, precision controls, and finite-bank subsamples do not inflate the model count. Figures retain all recorded path points; no test-derived envelope or smoothing is applied.

`python -m experiments.benchmark.analyze_history_support` examines reference-gap
associations with paired arithmetic/geometric hits in four retained TIGER caches.
It fixes the existing validation choices, reconstructs selected rankings, and
controls for training frequency, raw rank groups, and the Top-10 score margin.
Outputs in `results/benchmark/history_support_analysis/` include the analysis
protocol, source hashes, target-item-clustered OLS intervals, raw-hit losses and
raw-miss recoveries, paired user intervals, and validation-to-test prediction
checks. It requires the local score-bank, candidate, top-10 and item-hit NPZ
files. Beauty is excluded because its matched candidate cache is unavailable.
These are exploratory target-conditioned associations, not causal explanations
or a new deployment policy. No recommender is trained and no test-driven strength is
selected. The finite-bank resampling study remains a separate analysis.

## Search complementarity with ISD

`isd_probe.py` independently implements the training-statistics provider and
prefix/final reciprocal-rank fusion from arXiv:2607.24995. It uses local legacy
TIGER checkpoints, verified 512-history geometric references, and no new training.
The paper's beam 50 is adapted to 20, with J=50 and kappa=60. Co-consumption uses
a fixed symmetric ten-position window (the source does not numerically specify
this window). All arms mask observed consumed items at completion, so these
results are separate from the existing primary tables.

```bash
python -m unittest experiments.benchmark.test_isd_probe
python -m experiments.benchmark.isd_probe --dataset beauty --device cuda
python -m experiments.benchmark.isd_probe --dataset toys --device cuda
python -m experiments.benchmark.isd_probe --dataset beauty --device cuda --n-test 7000 --test-offset 3000 --selection-source results/benchmark/isd_probe/beauty/selection.json --output results/benchmark/isd_confirmation
python -m experiments.benchmark.isd_probe --dataset toys --device cuda --n-test 7000 --test-offset 3000 --selection-source results/benchmark/isd_probe/toys/selection.json --output results/benchmark/isd_confirmation
python -m experiments.benchmark.summarize_isd_probe
```

The initial run fixes beta using 1,000 validation histories, then tests 3,000
histories. Confirmation uses a disjoint 7,000-history slice without retuning.
Each correction family has the same 5% relative validation accuracy budget
against its corresponding uncorrected pipeline. The summary retains every arm,
including ordering-only, search-only, complete ISD, and both composition choices.
Reports and prediction audits are written to `reports/isd_probe/`. CUDA runs
use two CPU threads, FP32, batch size four, and a 12% device-memory limit.
