"""Conditional history-sampling variability on frozen candidate caches.

Subsets are drawn from an existing finite background bank. This measures
estimator sensitivity conditional on one trained model and that bank, not
training-seed uncertainty or an independent population sampling experiment.
All validation choices are written before this script reads test candidates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from .common import ROOT, metrics, paired_ci, save_json, sha256
from .evaluate import BETAS, select_validation, standardize


class CandidateTable:
    def __init__(self, path):
        with np.load(path) as f:
            self.targets = f["targets"]
            self.indices = f["indices"]
            offsets, items, scores = f["offsets"], f["items"], f["raw"]
        lengths = np.diff(offsets)
        if lengths.min() < 10:
            raise ValueError("Fewer than ten candidates")
        self.items = np.zeros((len(lengths), lengths.max()), dtype=np.int64)
        self.scores = np.full(self.items.shape, -np.inf)
        for row, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            self.items[row, :b-a] = items[a:b]
            self.scores[row, :b-a] = scores[a:b]

    def rank(self, prior=None, beta=0.):
        score = self.scores if prior is None else self.scores - beta * prior[self.items]
        order = np.lexsort((self.items, -score), axis=1)[:, :10]
        return np.take_along_axis(self.items, order, axis=1)


def priors_for(background, sizes, repetitions, base_seed):
    for size in sizes:
        for draw in range(1 if size == len(background) else repetitions):
            chosen = np.random.default_rng(base_seed+draw).permutation(len(background))[:size]
            values = background[chosen]
            yield size, draw, chosen, {
                "geometric": standardize(values.mean(0)),
                "arithmetic": standardize(logsumexp(values, axis=0)-np.log(size))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--background-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repetitions", type=int, default=20)
    args = ap.parse_args()
    out, source, background_dir = Path(args.output), Path(args.cache_dir), Path(args.background_dir)
    if (out/"complete.json").exists():
        return
    source_config = json.loads((source/"config.json").read_text())
    bg_config = json.loads((background_dir/"config.json").read_text())
    if not source_config.get("checkpoint_sha256") or source_config["checkpoint_sha256"] != bg_config.get("checkpoint_sha256"):
        raise ValueError("Background and candidates must come from the same frozen checkpoint")
    with np.load(background_dir/"sampled_history_scores.npz") as f:
        background = f["scores"].astype(np.float64)
    with np.load(background_dir/"priors.npz") as f:
        null = standardize(f["null"])
    with np.load(ROOT/f"results/benchmark/data/{args.split}.npz") as f:
        head, counts = f["head_mask"], f["pop_counts"]
    sizes = [m for m in [1,4,8,16,32,64,128] if m <= len(background)]
    config = {**vars(args), "sizes": sizes, "betas": BETAS, "base_seed": 991,
              "background_sha256": sha256(background_dir/"sampled_history_scores.npz"),
              "checkpoint_sha256": source_config["checkpoint_sha256"],
              "scope": "finite-bank sampling sensitivity; not training randomness or population uncertainty",
              "selection": "per-family validation only; same beta grid and recall constraints",
              "nested_subsets_across_M": True, "created_unix": time.time()}
    if (out/"config.json").exists():
        old = json.loads((out/"config.json").read_text())
        if any(old.get(k) != v for k,v in config.items() if k != "created_unix"):
            raise ValueError("Existing analysis has different settings")
    else:
        save_json(out/"config.json", config)
    reference = standardize(background.mean(0))
    valid = CandidateTable(source/"valid_candidates.npz")
    base_metrics, base_hit = metrics(valid.rank(),valid.targets,head,len(head))
    validation, estimates = [], []
    for size, draw, chosen, families in priors_for(background,sizes,args.repetitions,991):
        rows, hits = {"raw":base_metrics}, {"raw":base_hit}
        for family, prior in families.items():
            for beta in BETAS[1:]:
                name = f"{family}:{beta}"
                rows[name],hits[name] = metrics(valid.rank(prior,beta),valid.targets,head,len(head))
        selection = select_validation(rows,hits,valid.targets,head)
        record = {"M":size,"draw":draw,"background_rows":chosen.tolist(),"selection":selection,
                  "rho_geometric_full_bank":float(spearmanr(families["geometric"],reference).statistic),
                  "standardized_rmse_to_full_bank":float(np.sqrt(np.mean((families["geometric"]-reference)**2))),
                  "rho_geometric_train_popularity":float(spearmanr(families["geometric"],counts).statistic),
                  "validation_metrics":rows}
        validation.append(record)
        estimates.append(families)
        if draw == 0:
            print("validation M",size,flush=True)
    null_rows, null_hits = {"raw":base_metrics}, {"raw":base_hit}
    for beta in BETAS[1:]:
        name=f"null:{beta}"
        null_rows[name],null_hits[name]=metrics(valid.rank(null,beta),valid.targets,head,len(head))
    null_selection=select_validation(null_rows,null_hits,valid.targets,head)
    frozen={"draws":validation,"null_selection":null_selection,"null_validation":null_rows,
            "frozen_unix":time.time()}
    save_json(out/"validation_frozen.json",frozen)
    # Only now open test data; there is no test-driven choice of M or estimator.
    test=CandidateTable(source/"test_candidates.npz")
    test_base,base_hit=metrics(test.rank(),test.targets,head,len(head))
    tail=~head[test.targets]
    rules=["strict","budget_5pct","simultaneous_ci_budget_5pct"]
    def selected_metrics(families,selection):
        result={}
        for family, choices in selection.items():
            result[family]={}
            for rule in rules:
                name=choices[rule]
                top=test.rank() if name=="raw" else test.rank(families[family],float(name.split(":")[1]))
                row,hit=metrics(top,test.targets,head,len(head))
                row.update(paired_overall=paired_ci(base_hit,hit),paired_tail=paired_ci(base_hit[tail],hit[tail]))
                result[family][rule]={"method":name,"test":row}
        return result
    records=[]
    for row,families in zip(validation,estimates):
        records.append({"M":row["M"],"draw":row["draw"],
                        "selected_test":selected_metrics(families,row["selection"])})
    null_test=selected_metrics({"null":null},null_selection)
    summary=[]
    for size in sizes:
        matching=[r for r in records if r["M"]==size]
        for family in ["geometric","arithmetic"]:
            for rule in rules:
                values=[r["selected_test"][family][rule] for r in matching]
                row={"M":size,"family":family,"rule":rule,"draws":len(values),
                     "selection_counts":dict(Counter(v["method"] for v in values))}
                for metric in ["R10","tailR10","cov10"]:
                    x=np.array([v["test"][metric] for v in values])
                    row[metric]={"mean":float(x.mean()),"sd":float(x.std(ddof=1)) if len(x)>1 else None,
                                 "range":[float(x.min()),float(x.max())]}
                summary.append(row)
    save_json(out/"results.json",{"raw_test":test_base,"null_test":null_test,"draws":records,"summary":summary})
    save_json(out/"complete.json",{"finished_unix":time.time(),"validation_frozen_sha256":sha256(out/"validation_frozen.json")})
    print("completed",args.split,len(records),"draw configurations",flush=True)


if __name__=="__main__":
    main()
