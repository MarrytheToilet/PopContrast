"""Frozen LLM marginal correction on matched trie-beam candidate sets.

KV reuse avoids repeating the history forward pass for every catalog item.
Scores use the full language-model vocabulary, including the normalizer; no
SID-only probability renormalization is introduced by the trie constraint.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import load_bundle, metrics, paired_ci, save_json, seed_all, stable_topk, sha256
from .evaluate import BETAS, group_analysis, select_validation, standardize
from .lcrec import Collate, Examples, init_model, recommend, tokenizer_and_codes, trie_lookup


@torch.no_grad()
def cached_item_scores(model, prompt, item_tokens, chunk_size=16):
    """Exact complete-SID log-probabilities, sharing identical SID prefixes.

    The last model input is the penultimate SID token. Items sharing that
    complete input need one forward pass and different final-token gathers.
    """
    model.eval()
    prompt = prompt.reshape(1, -1)
    first = model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                  use_cache=True, logits_to_keep=1)
    first_logp = first.logits[0, -1].float().log_softmax(-1)
    if item_tokens.shape[1] == 1:
        return first_logp[item_tokens[:, 0]].cpu().numpy()
    prefixes, inverse = torch.unique(item_tokens[:, :-1], dim=0, return_inverse=True)
    values = torch.empty(len(item_tokens), device=prompt.device, dtype=torch.float32)
    for start in range(0, len(prefixes), chunk_size):
        tokens = prefixes[start:start + chunk_size]
        cache = copy.deepcopy(first.past_key_values)
        cache.batch_repeat_interleave(len(tokens))
        mask = torch.ones((len(tokens), prompt.shape[1] + tokens.shape[1]),
                          device=prompt.device, dtype=torch.long)
        output = model(input_ids=tokens, attention_mask=mask,
                       past_key_values=cache, use_cache=True)
        logp = output.logits.float().log_softmax(-1)
        prefix_score = first_logp[tokens[:, 0]] + logp[:, :-1].gather(2, tokens[:, 1:, None]).squeeze(-1).sum(-1)
        selected = (inverse >= start) & (inverse < start + len(tokens))
        local = inverse[selected] - start
        values[selected] = prefix_score[local] + logp[local, -1, item_tokens[selected, -1]]
    return values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--marginal-m", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--n-valid", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--precision", choices=["bf16", "fp32"])
    args = ap.parse_args()
    source, out = Path(args.training_run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    precision_policy = (json.loads((source / "evaluation_precision.json").read_text())
                        if (source / "evaluation_precision.json").exists() else {})
    args.precision = args.precision or precision_policy.get("precision", "bf16")
    if (source / "generation_policy_audit.json").exists() and not (source / "adapter_selection.json").exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, "-u", "-m", "experiments.benchmark.lcrec_select",
                        "--training-run", str(source)], check=True)
    adapter = source / ("evaluation_adapter" if (source / "evaluation_adapter").exists() else "best_adapter")
    fingerprint = sha256(adapter / "adapter_model.safetensors")
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        if old["adapter_sha256"] != fingerprint or any(old.get(k, "bf16" if k == "precision" else None) != v for k, v in vars(args).items()):
            raise ValueError("Evaluation directory already contains different model/settings")
    if (out / "complete.json").exists():
        return
    save_json(out / "config.json", {**vars(args), "adapter_sha256": fingerprint,
              "checkpoint_sha256": fingerprint,
              "adapter_source": str(adapter), "repetition_penalty": 1.0,
              "vectorized_trie_mask": True, "shared_history_prefill": False,
              "precision_policy": precision_policy,
              "training_config": config, "quality_only": True, "gpu_shared": True,
              "score": "full-vocabulary SID log-probability, EOS excluded", "betas": BETAS})
    torch.set_num_threads(4)
    seed_all(20260927)
    data = load_bundle(config["split"], sid_length=config["sid_length"])
    tokenizer, code_ids = tokenizer_and_codes(config["base_model"], config["sid_length"])
    if code_ids.tolist() != config["sid_codes"]:
        raise RuntimeError("Training/evaluation SID mapping mismatch")
    model = init_model(config["base_model"], tokenizer, code_ids, str(adapter)).eval()
    if args.precision == "fp32":
        model.float()
    score_tolerance = 1e-3 if args.precision == "fp32" else .3
    train = Examples(data, "train", tokenizer, code_ids, prompt_format=config.get("prompt_format", "legacy"))
    item_tokens = torch.tensor(train.item_ids, device="cuda", dtype=torch.long)
    children, leaves = trie_lookup(train.item_ids)
    started = time.perf_counter()
    if not (out / "priors.npz").exists():
        indices = data["legacy_marginal_idx"][:args.marginal_m]
        scores = []
        for number, index in enumerate(indices):
            prompt, _, _ = train[int(index)]
            scores.append(cached_item_scores(model, torch.tensor(prompt, device="cuda"), item_tokens, args.chunk_size))
            print("marginal", number + 1, len(indices), flush=True)
        scores = np.asarray(scores, dtype=np.float64)
        maxima = scores.max(0)
        history_seconds=time.perf_counter()-started
        null_started=time.perf_counter()
        null_prompt = torch.tensor(train.prefix + train.suffix, device="cuda")
        null = cached_item_scores(model, null_prompt, item_tokens, args.chunk_size)
        null_seconds=time.perf_counter()-null_started
        np.savez_compressed(out / "priors.npz", geometric=scores.mean(0),
                            arithmetic=maxima + np.log(np.exp(scores - maxima).mean(0)),
                            null=null, logcount=np.log1p(data["pop_counts"]))
        np.savez_compressed(out / "sampled_history_scores.npz", scores=scores, indices=indices)
        save_json(out / "offline_cost.json", {"elapsed_seconds": time.perf_counter() - started,
                   "history_estimation_seconds": history_seconds,"null_estimation_seconds": null_seconds,
                   "gpu_shared": True, "histories": len(indices), "vector_bytes_fp32": len(item_tokens) * 4,
                   "unique_scored_sid_prefixes": int(torch.unique(item_tokens[:, :-1], dim=0).shape[0]),
                   "catalog_items": len(item_tokens), "prefix_sharing": True})
    with np.load(out / "priors.npz") as packed:
        z = {key: standardize(packed[key]) for key in packed.files}
    for partition in ["valid", "test"]:
        indices = (np.random.default_rng(7).permutation(len(data["valid_target"]))[:args.n_valid]
                   if partition == "valid" else data["legacy_test_idx"][:args.n_test])
        ds = Examples(data, partition, tokenizer, code_ids, indices, config.get("prompt_format", "legacy"))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2,
                            collate_fn=Collate(tokenizer.pad_token_id, False))
        targets = data[partition + "_target"][indices]
        tops = {"raw": []}
        tops.update({f"{name}:{b}": [] for name in z for b in BETAS[1:]})
        cached_items, cached_scores, offsets = [], [], [0]
        score_audit = []
        for batch, (ids, mask, _) in enumerate(loader):
            candidates = recommend(model, ids.cuda(), mask.cuda(), children, leaves,
                                   config["sid_length"], args.beam, tokenizer)
            for local, (items, raw) in enumerate(candidates):
                if batch == 0:
                    prompt = ids[local][mask[local].bool()].cuda()
                    reference = cached_item_scores(model, prompt, item_tokens[items], args.chunk_size)
                    score_audit.append({"index": int(indices[local]),
                        "max_abs_error": float(np.max(np.abs(reference - raw))),
                        "mean_abs_error": float(np.mean(np.abs(reference - raw)))})
                    if np.max(np.abs(reference - raw)) > score_tolerance:
                        save_json(out / "score_audit_failure.json", {
                            "partition": partition, "batch": batch, "checks": score_audit,
                            "tolerance": score_tolerance, "precision": args.precision, "adapter_sha256": fingerprint,
                            "candidate_items": items.tolist(), "beam_scores": raw.tolist(),
                            "teacher_forced_scores": reference.tolist()})
                        raise RuntimeError("Beam/teacher-forced score discrepancy exceeds precision-specific audit tolerance")
                cached_items.extend(items.tolist())
                cached_scores.extend(raw.tolist())
                offsets.append(len(cached_items))
                tops["raw"].append(stable_topk(raw, ids=items))
                for name, prior in z.items():
                    for beta in BETAS[1:]:
                        tops[f"{name}:{beta}"].append(stable_topk(raw - beta * prior[items], ids=items))
            if (batch + 1) % 50 == 0:
                print(partition, len(tops["raw"]), len(indices), flush=True)
                save_json(out / f"progress_{partition}.json", {"done": len(tops["raw"]), "total": len(indices)})
        rows, hits = {}, {}
        for name, top in tops.items():
            rows[name], hits[name] = metrics(np.asarray(top), targets, data["head_mask"], len(item_tokens))
            if name != "raw":
                rows[name]["paired_overall"] = paired_ci(hits["raw"], hits[name])
                tail = ~data["head_mask"][targets]
                rows[name]["paired_tail"] = paired_ci(hits["raw"][tail], hits[name][tail])
        save_json(out / f"{partition}_metrics.json", rows)
        save_json(out / f"{partition}_score_audit.json", {"candidate_score_checks": score_audit,
                  "precision": args.precision, "tolerance": score_tolerance,
                  "note": "Different matrix shapes can introduce rounding; repetition and ngram penalties explicitly disabled"})
        np.savez_compressed(out / f"{partition}_top10.npz", indices=indices, targets=targets, **tops)
        np.savez_compressed(out / f"{partition}_candidates.npz", indices=indices, targets=targets,
                            items=np.asarray(cached_items), raw=np.asarray(cached_scores), offsets=np.asarray(offsets))
        save_json(out / f"{partition}_segments.json", group_analysis(data, data[partition + "_history"][indices],
                  targets, hits["raw"], hits["geometric:0.75"]))
        if partition == "valid":
            selection = select_validation(rows, hits, targets, data["head_mask"])
            save_json(out / "validation_selection.json", selection)
        else:
            save_json(out / "selected_test.json", {name: {rule: {"method": choices[rule], "test": rows[choices[rule]]}
                      for rule in ["strict", "budget_5pct", "simultaneous_ci_budget_5pct"]}
                      for name, choices in selection.items()})
    save_json(out / "complete.json", {"finished_unix": time.time(), "adapter_sha256": fingerprint})


if __name__ == "__main__":
    main()
