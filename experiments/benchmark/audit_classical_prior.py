"""Read-only unsupervised score geometry for completed classical checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .classical import build,iter_scores
from .common import load_bundle,save_json,seed_all,sha256
from .prior_geometry import personalization_decomposition


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--runs",nargs="+",required=True)
    args=ap.parse_args();seed_all(20260927);torch.set_num_threads(2)
    for value in args.runs:
        run=Path(value)
        if not (run/"complete.json").exists():raise ValueError("Use completed training only")
        config=json.loads((run/"config.json").read_text())
        data=load_bundle(config["split"])
        model=build(config["model"],len(data["pop_counts"])).cuda()
        model.load_state_dict(torch.load(run/"best_model.pt",weights_only=True,map_location="cpu"))
        histories=data["train_history"][data["legacy_marginal_idx"][:128]]
        scores=np.concatenate(list(iter_scores(model,config["model"],histories))).astype(np.float64)
        result={"run":str(run),"dataset":config["split"],"model":config["model"],
                "checkpoint_sha256":sha256(run/"best_model.pt"),"histories":len(histories),
                "rho_model_prior_item_train_count":float(spearmanr(scores.mean(0),data["pop_counts"]).statistic),
                "personalization":personalization_decomposition(scores),"test_labels_used":False}
        save_json(run/"prior_geometry.json",result);print(json.dumps(result),flush=True)
        del model,scores,data
        torch.cuda.empty_cache()


if __name__=="__main__":main()
