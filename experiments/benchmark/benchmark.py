"""Matched online decode timing, requiring an otherwise idle selected GPU."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_bundle, load_model, save_json, stable_topk, tokens_for_histories
from .evaluate import standardize
from .search import beam_search, encode, make_trie


def assert_exclusive():
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    if device not in ["2", "3"]:
        raise RuntimeError("Only explicitly assigned GPUs 2 and 3 are allowed")
    uuid = subprocess.check_output(["nvidia-smi", f"--id={device}", "--query-gpu=uuid",
                                   "--format=csv,noheader"], timeout=15).decode().strip()
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], timeout=15).decode().splitlines()
    foreign = [line for line in rows if line.split(",")[0].strip() == uuid and int(line.split(",")[1]) != os.getpid()]
    if foreign:
        raise RuntimeError(f"GPU has other compute processes; timing would be invalid: {foreign}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--users", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    assert_exclusive()
    torch.set_num_threads(2)
    data = load_bundle(args.split)
    model = load_model(args.checkpoint).eval()
    prior = standardize(data["legacy_marginal"])
    trie = make_trie(data["item_tokens"], prior)
    indices = data["legacy_test_idx"][:args.users]
    ids, mask = tokens_for_histories(data["test_history"][indices], data["item_tokens"], "cuda")
    beta = .75
    def run(method, row):
        enc = encode(model, ids[row:row+1], mask[row:row+1])
        result = beam_search(model, enc, mask[row:row+1], trie, prior, args.beam,
                             beta if method == "bound" else 0., "bound" if method == "bound" else "raw")
        if method == "leaf":
            return stable_topk(result["raw_scores"] - beta * prior[result["items"]], ids=result["items"])
        return result["top"]
    for i in range(30):
        for method in ["raw", "leaf", "bound"]:
            run(method, i % len(indices))
    assert_exclusive()
    methods = ["raw", "leaf", "bound"]
    timings = {key: [] for key in methods}
    tops = {key: [] for key in methods}
    rng = np.random.default_rng(27)
    for row in range(len(indices)):
        for method in rng.permutation(methods):
            torch.cuda.synchronize()
            start = time.perf_counter()
            top = run(method, row)
            torch.cuda.synchronize()
            timings[method].append((time.perf_counter() - start) * 1000)
            tops[method].append(top)
    assert_exclusive()
    arrays = {key: np.asarray(value) for key, value in timings.items()}
    summary = {key: {"mean_ms": float(value.mean()), "p50_ms": float(np.median(value)),
                         "p95_ms": float(np.quantile(value, .95))} for key, value in arrays.items()}
    for key in ["leaf", "bound"]:
        delta = arrays[key] - arrays["raw"]
        draws = rng.integers(len(delta), size=(2000, len(delta)))
        summary[key]["paired_mean_overhead_ms"] = float(delta.mean())
        summary[key]["paired_ci95_ms"] = np.quantile(delta[draws].mean(1), [.025, .975]).tolist()
    from .common import metrics
    quality = {key: metrics(value, data["test_target"][indices], data["head_mask"], len(prior))[0]
               for key, value in tops.items()}
    save_json(out / "timing.json", {"configuration": vars(args), "gpu": torch.cuda.get_device_name(),
              "precision": "FP32 forward and log-softmax", "exclusive_checks_passed": True,
              "scope": "single-request encoder + trie beam + final ranking; pretokenized GPU inputs",
              "offline_estimation_excluded": True, "warmup_requests_per_method": 30,
              "order": "random paired method order per identical user", "summary": summary,
              "matched_subset_quality": quality, "marginal_vector_bytes_fp32": 4 * len(prior)})
    np.savez_compressed(out / "requests.npz", indices=indices, **arrays)
    save_json(out / "complete.json", {"finished_unix": time.time()})


if __name__ == "__main__":
    main()
