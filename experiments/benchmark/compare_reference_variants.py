"""Compare reference estimators using archived banks and resampling results.

This is descriptive analysis, with no model fitting or test-driven selection.
Missing or mismatched score banks are excluded from the mechanistic analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"beauty": "beauty_s1_l3", "clothing": "clothing_s1_l3",
        "ml1m": "ml1m_s1_l4", "sports": "sports_s1_l3", "toys": "toys_s1_l3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/benchmark/reference_variant_analysis.json")
    args = parser.parse_args()
    sources, downstream, mechanism, excluded = {}, {}, {}, {}

    def read(path):
        content = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    for dataset, run in RUNS.items():
        folder = ROOT / "results/benchmark/estimator_resampling" / dataset
        config, results = read(folder / "config.json"), read(folder / "results.json")
        draws = [d for d in results["draws"] if d["M"] == 32]
        assert len(draws) == config["repetitions"] == 20
        families = {}
        for family in ["geometric", "arithmetic"]:
            selected = [d["selected_test"][family]["budget_5pct"]["test"] for d in draws]
            families[family] = {
                metric: {"mean_percent": float(np.mean([100 * d[metric] for d in selected])),
                         "sd_pp": float(np.std([100 * d[metric] for d in selected], ddof=1))}
                for metric in ["R10", "tailR10", "cov10"]}
        downstream[dataset] = {"M": 32, "draws": len(draws), "metrics": families,
            "geometric_tail_sd_reduction_percent": 100 * (1 -
                families["geometric"]["tailR10"]["sd_pp"] /
                families["arithmetic"]["tailR10"]["sd_pp"]),
            "raw_percent": {key: 100 * results["raw_test"][key]
                            for key in ["R10", "tailR10", "cov10"]}}

        path = ROOT / "results/benchmark/runs" / run / "evaluation/sampled_history_scores.npz"
        if not path.exists():
            excluded[dataset] = "Archived score bank unavailable."
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config["background_sha256"]:
            excluded[dataset] = {"reason": "Available bank differs from the resampling bank.",
                                 "expected_sha256": config["background_sha256"],
                                 "available_sha256": digest}
            continue
        sources[str(path.relative_to(ROOT))] = digest
        with np.load(path) as bank:
            scores = bank["scores"].astype(np.float64)
        assert np.isfinite(scores).all()
        count = len(scores)
        geometric = scores.mean(axis=0)
        normalizer = logsumexp(scores, axis=0)
        arithmetic = normalizer - np.log(count)
        log_weight = scores - normalizer
        weight = np.exp(log_weight)
        gap = arithmetic - geometric
        kl = (-np.log(count) - log_weight).mean(axis=0)
        assert gap.min() > -1e-10
        np.testing.assert_allclose(gap, kl, atol=1e-12)
        np.testing.assert_allclose(weight.sum(axis=0), 1, atol=1e-12)
        # This measures concentration of the estimator's derivative across histories.
        # It is NOT a sample-size estimate for statistical inference.
        influence_count = 1 / (weight ** 2).sum(axis=0)
        standardized = lambda x: (x - x.mean()) / x.std()
        normalized_scores = scores - logsumexp(scores, axis=1, keepdims=True)
        geometric_shift_error = float(np.max(np.abs(
            standardized(geometric) - standardized(normalized_scores.mean(axis=0)))))
        assert geometric_shift_error < 1e-10
        mechanism[dataset] = {
            "histories": count, "items": scores.shape[1],
            "arithmetic_influence_count_quantiles_10_50_90": np.quantile(
                influence_count, [.1, .5, .9]).tolist(),
            "geometric_influence_count": count,
            "arithmetic_largest_history_weight_median": float(np.median(weight.max(axis=0))),
            "jensen_gap_nats_quantiles_10_50_90": np.quantile(gap, [.1, .5, .9]).tolist(),
            "reference_rank_correlation": float(spearmanr(geometric, arithmetic).statistic),
            "geometric_standardized_history_offset_invariance_error": geometric_shift_error,
            "jensen_gap_KL_identity_max_error": float(np.max(np.abs(gap - kl)))}

    report = {"sources": sources, "downstream_resampling": downstream,
              "matched_bank_mechanism": mechanism, "excluded_mechanism_banks": excluded,
              "scope": [
                  "Downstream SD conditions on finite banks and includes validation strength selection.",
                  "Influence concentration concerns derivatives with respect to log-scores.",
                  "No subgroup relevance or causal link from concentration to recall is established.",
                  "Raw reference differences do not directly compare penalties after separate standardization.",
                  "Scenario recommendations are hypotheses; estimator choice must use validation data."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "matched_banks": list(mechanism),
        "excluded_banks": list(excluded),
        "tail_sd_reduction_percent": {k: round(v["geometric_tail_sd_reduction_percent"], 1)
                                      for k, v in downstream.items()},
        "median_arithmetic_influence_count": {k: round(v["arithmetic_influence_count_quantiles_10_50_90"][1], 1)
                                               for k, v in mechanism.items()}}, indent=2))


if __name__ == "__main__":
    main()
