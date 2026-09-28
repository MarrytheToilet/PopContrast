"""Real-checkpoint parity check for evaluation throughput optimization."""
import json
import time
import numpy as np
import torch

from .common import load_bundle, load_model, save_json, seed_all, tokens_for_histories
from .evaluate import standardize
from .search import beam_search, beam_search_batch, encode, make_trie


@torch.no_grad()
def main():
    seed_all(20260927)
    torch.set_num_threads(2)
    data = load_bundle("beauty")
    model = load_model("checkpoints/legacy/beauty.pt").eval()
    prior = standardize(data["legacy_marginal"])
    trie = make_trie(data["item_tokens"], prior)
    indices = np.random.default_rng(7).permutation(len(data["valid_target"]))[:64]
    result = {}
    for mode, beta in [("raw", 0.), ("bound", .75), ("bound", 1.5)]:
        single, batched = [], []
        started = time.perf_counter()
        for idx in indices:
            ids, mask = tokens_for_histories(data["valid_history"][[idx]], data["item_tokens"], "cuda")
            single.append(beam_search(model, encode(model, ids, mask), mask, trie, prior, 20, beta, mode))
        single_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for start in range(0, len(indices), 8):
            ids, mask = tokens_for_histories(data["valid_history"][indices[start:start+8]], data["item_tokens"], "cuda")
            batched.extend(beam_search_batch(model, encode(model, ids, mask), mask, trie, prior, 20, beta, mode))
        batch_seconds = time.perf_counter() - started
        overlap, max_error, identical = [], 0., 0
        for first, second in zip(single, batched):
            overlap.append(np.isin(first["top"], second["top"]).mean())
            identical += int(np.array_equal(first["top"], second["top"]))
            one, two = dict(zip(first["items"], first["raw_scores"])), dict(zip(second["items"], second["raw_scores"]))
            max_error = max(max_error, max(abs(one[i] - two[i]) for i in one.keys() & two.keys()))
        result[f"{mode}:{beta}"] = {"users": len(indices), "identical_ordered_top10": identical,
              "set_overlap": float(np.mean(overlap)), "max_common_candidate_score_error": max_error,
              "single_seconds_shared_gpu": single_seconds, "batch_seconds_shared_gpu": batch_seconds}
        if identical != len(indices) or max_error > 1e-4:
            raise AssertionError(result)
    save_json("runs/batched_search_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
