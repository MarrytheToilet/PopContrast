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

## Summaries and figures

```bash
python -m experiments.benchmark.summarize
python scripts/check_results.py
python -m experiments.figures.build_tables
python -m experiments.figures.publication
```

The summaries contain all five operating rules for the 33 primary settings, 63 matched baseline rows, and 24 cohort rows. Only complete, distinct primary checkpoints enter the inventory. Encodings, precision controls, and finite-bank subsamples do not inflate the model count. Figures retain all recorded path points; no test-derived envelope or smoothing is applied.
