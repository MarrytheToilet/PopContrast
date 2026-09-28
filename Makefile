.PHONY: summaries figures check test

summaries:
	python -m experiments.benchmark.summarize

figures:
	python -m experiments.make_figures
	python -m experiments.figures.build_tables
	python -m experiments.figures.publication

check:
	python scripts/check_results.py

# CPU checks for scoring, ties, samplers, and model/cache equivalence.
test:
	python -m unittest experiments.benchmark.test_search experiments.benchmark.test_cached_t5 experiments.benchmark.test_lcrec experiments.benchmark.test_estimator_resampling experiments.benchmark.test_sampler experiments.benchmark.test_prior_geometry
