.PHONY: summaries figures check analysis-inputs test

analysis-inputs:
	python scripts/check_results.py --check-inputs

summaries: analysis-inputs
	python -m experiments.benchmark.summarize

figures: analysis-inputs
	python -m experiments.figures.build_tables
	python -m experiments.figures.estimator_stability --output-dir results/tables
	python -m experiments.figures.publication

check: analysis-inputs
	python scripts/check_results.py

# CPU checks for scoring, ties, samplers, and model/cache equivalence.
test:
	python -m unittest experiments.benchmark.test_search experiments.benchmark.test_cached_t5 experiments.benchmark.test_lcrec experiments.benchmark.test_estimator_resampling experiments.benchmark.test_sampler experiments.benchmark.test_prior_geometry
