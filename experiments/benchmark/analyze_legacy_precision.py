"""Join re-executed beam scores to the exact cache used by the submitted table."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .common import metrics, save_json, sha256, stable_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--original-cache", default="results/cache_scores_beauty.pt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cache = torch.load(args.original_cache, map_location="cpu", weights_only=False)
    scores, marginal = cache["scores"].float(), cache["marginal"].float()
    z = ((marginal - marginal.mean()) / marginal.std()).numpy()
    n_items = scores.shape[1]
    rows = {}
    for path in sorted(Path(args.audit_dir).glob("*.npz")):
        with np.load(path) as packed:
            beam = {key: packed[key] for key in packed.files}
        n = len(beam["indices"])
        if not np.array_equal(beam["targets"], cache["targets"][:n].numpy()):
            raise ValueError("Historical cache target alignment failed")
        for width in [20, 100]:
            for beta in [.75, 1.5]:
                exact_scores = (scores[:n] - beta * torch.as_tensor(z)[None]).numpy()
                inclusion, overlap, replay_overlap, score_error = [], [], [], []
                hits, coverage = [], set()
                for user in range(n):
                    a,b = beam[f"B{width}_offsets"][user:user+2]
                    items = beam[f"B{width}_items"][a:b].astype(int)
                    raw = beam[f"B{width}_raw"][a:b].astype(np.float32)
                    exact_top = stable_topk(exact_scores[user])
                    corrected = stable_topk(raw - beta * z[items], ids=items)
                    replay = stable_topk(exact_scores[user,items], ids=items)
                    inclusion.append(float(np.isin(exact_top,items).mean()))
                    overlap.append(float(np.isin(exact_top,corrected).mean()))
                    replay_overlap.append(float(np.isin(exact_top,replay).mean()))
                    score_error.extend(np.abs(scores[user,items].numpy() - raw).tolist())
                    hits.append(beam["targets"][user] in corrected)
                    coverage.update(corrected)
                # Under a common score and deterministic tie rule, these must coincide.
                np.testing.assert_allclose(inclusion, replay_overlap, atol=0, rtol=0)
                tail = ~cache["seg_head"][:n].numpy()
                rows[f"{path.stem}/B{width}/beta{beta}"] = {
                    "n": n, "candidate_inclusion": float(np.mean(inclusion)),
                    "different_score_overlap": float(np.mean(overlap)),
                    "common_score_replay_overlap": float(np.mean(replay_overlap)),
                    "raw_candidate_score_abs_error_quantiles": np.quantile(score_error,[.5,.9,.99,1.]).tolist(),
                    "R10": float(np.mean(hits)), "tailR10": float(np.asarray(hits)[tail].mean()),
                    "cov10": len(coverage)/n_items}
    save_json(args.output, {"original_cache_sha256": sha256(args.original_cache), "results": rows,
              "ties": "deterministic item-ID ties for both reference and replay; submitted GPU tie order not assumed",
              "scope": "historical source on current hardware; original cached reference retained; not exact old-kernel replay"})
    print(json.dumps({key:value for key,value in rows.items() if "B100/beta0.75" in key}), flush=True)


if __name__ == "__main__":
    main()
