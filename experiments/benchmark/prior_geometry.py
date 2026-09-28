"""Exact estimator identities; no causal interpretation of a model prior."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from .common import load_bundle,save_json,sha256
from .evaluate import standardize


def personalization_decomposition(scores):
    scores=np.asarray(scores,dtype=np.float64)
    centered=scores-scores.mean(1,keepdims=True)
    item_effect=centered.mean(0)
    residual=centered-item_effect
    total=float(np.mean(centered**2))
    marginal=float(np.mean(item_effect**2))
    interaction=float(np.mean(residual**2))
    np.testing.assert_allclose(total,marginal+interaction,rtol=1e-10,atol=1e-12)
    logp=scores-logsumexp(scores,axis=1,keepdims=True)
    logq=logsumexp(logp,axis=0)-np.log(len(scores))
    js=float(np.mean(np.sum(np.exp(logp)*(logp-logq),axis=1)))
    return {"row_centered_total_score_variance":total,
            "item_main_effect_variance":marginal,"history_item_residual_variance":interaction,
            "item_main_effect_fraction":marginal/total if total>0 else None,
            "catalog_softmax_history_JS_nats":max(0.,js),
            "scope":"descriptive personalization of catalog-normalized scores; not harmful-bias identification or a calibrated applicability threshold"}


def analyze(scores,counts):
    scores=np.asarray(scores,dtype=np.float64)
    geometric=scores.mean(0)
    arithmetic=logsumexp(scores,axis=0)-np.log(len(scores))
    normalizer=logsumexp(scores,axis=1,keepdims=True)
    normalized=scores-normalizer
    arithmetic_catalog=logsumexp(normalized,axis=0)-np.log(len(scores))
    gap=arithmetic-geometric
    invariant=float(np.max(np.abs(standardize(geometric)-standardize(normalized.mean(0)))))
    if gap.min() < -1e-10 or invariant>1e-10:
        raise AssertionError("Jensen or common-history-offset identity failed")
    return arithmetic_catalog,{
        "histories":len(scores),"items":scores.shape[1],
        "geometric_catalog_normalization_max_standardized_error":invariant,
        "jensen_gap_quantiles":np.quantile(gap,[0,.25,.5,.75,1]).tolist(),
        "rho_jensen_gap_train_popularity":float(spearmanr(gap,np.log1p(counts)).statistic),
        "rho_geometric_arithmetic":float(spearmanr(geometric,arithmetic).statistic),
        "rho_geometric_catalog_arithmetic":float(spearmanr(geometric,arithmetic_catalog).statistic),
        "per_history_catalog_log_normalizer_quantiles":np.quantile(normalizer,[0,.25,.5,.75,1]).tolist(),
        "personalization":personalization_decomposition(scores),
        "interpretation":"geometric correction = arithmetic log-ratio + item Jensen gap, before standardization/coefficient rescaling; neither isolates harmful popularity",
        "catalog_normalization":"softmax over complete-item score vectors; colliding SID item labels are retained separately"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",required=True)
    ap.add_argument("--background-file",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    with np.load(args.background_file) as f:scores=f["scores"]
    _,result=analyze(scores,load_bundle(args.split)["pop_counts"])
    save_json(args.output,{**vars(args),"background_sha256":sha256(args.background_file),**result})


if __name__=="__main__":main()
