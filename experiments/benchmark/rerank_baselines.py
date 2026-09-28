"""Fairly tuned rerankers on identical frozen trie-beam candidates/users.

SASRec score fusion is an auxiliary-model baseline, not a claim of reproducing
every component of D3. MMR and all score corrections receive a validation grid.
Raw candidate scores are cached for subsequent CPU-only development analyses.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .classical import build, hidden
from .common import ROOT, load_bundle, load_model, metrics, paired_ci, save_json, sha256, stable_topk, tokens_for_histories
from .evaluate import BETAS, standardize, select_validation
from .search import beam_search, encode, make_trie


def mmr(items, scores, similarity, weight, k=10):
    relevance = (scores - scores.mean()) / max(float(scores.std()), 1e-8)
    selected, active = [], np.ones(len(items), dtype=bool)
    diversity = np.zeros(len(items))
    for _ in range(min(k, len(items))):
        objective = weight * relevance - (1 - weight) * diversity
        available = np.flatnonzero(active)
        idx = available[np.lexsort((items[available], -objective[available]))[0]]
        selected.append(int(items[idx]))
        active[idx] = False
        diversity = similarity[:, idx] if len(selected) == 1 else np.maximum(diversity, similarity[:, idx])
    return np.asarray(selected)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--assistant-checkpoint", required=True)
    ap.add_argument("--prior-file", required=True)
    ap.add_argument("--candidate-source", help="Reuse a completed FP32 primary beam cache exactly")
    ap.add_argument("--output", required=True)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--n-valid", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    torch.set_num_threads(2)
    data = load_bundle(args.split)
    n_items = len(data["pop_counts"])
    source = Path(args.candidate_source) if args.candidate_source else None
    if source:
        source_config=json.loads((source/'config.json').read_text())
        if not (source/'complete.json').exists():raise ValueError('Primary candidate source incomplete')
        expected={'checkpoint_sha256':sha256(args.checkpoint),'split':args.split,
                  'beam':args.beam,'sid_length':3}
        if any(source_config.get(key)!=value for key,value in expected.items()):
            raise ValueError('Primary candidate source model/data/beam mismatch')
        if source_config.get('precision','fp32')!='fp32':raise ValueError('Expected FP32 source')
        if sha256(args.prior_file)!=sha256(source/'priors.npz'):
            raise ValueError('Prior differs from the frozen primary evaluation')
    model = load_model(args.checkpoint).eval() if source is None else None
    assistant = build("sasrec", n_items).cuda().eval()
    assistant.load_state_dict(torch.load(args.assistant_checkpoint, weights_only=True))
    embedding_path=ROOT / f"results/benchmark/data/{args.split}_embeddings.npy"
    emb = np.load(embedding_path)
    if emb.shape[0]!=n_items:raise ValueError('Embedding and catalog sizes differ')
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    with np.load(args.prior_file) as packed:
        prior = {key: standardize(packed[key]) for key in packed.files}
    trie = make_trie(data["item_tokens"], prior["geometric"])
    weights = [0., .1, .25, .5, .75, .9, .95, .99, 1.]
    fusion_weights = [.01, .03, .1, .25, .5, .75, 1.]
    save_json(out / "config.json", {**vars(args), "checkpoint_sha256": sha256(args.checkpoint),
             "assistant_sha256": sha256(args.assistant_checkpoint), "correction_betas": BETAS,
             "mmr_weights": weights, "fusion_weights": fusion_weights,
             "fusion": "(1-alpha)*TIGER log-prob + alpha*SASRec logit; SASRec log-normalizer cancels per user",
             "mmr": "per-user standardized raw relevance, cosine redundancy, exact item-ID tie breaking",
             "candidate_source_config_sha256":sha256(source/'config.json') if source else None,
             "embedding_sha256":sha256(embedding_path),
             "quality_only": True, "matched_candidates": True, "matched_users": True})
    for partition in ["valid", "test"]:
        n = args.n_valid if partition == "valid" else args.n_test
        indices = (np.random.default_rng(7).permutation(len(data["valid_target"]))[:n]
                   if partition == "valid" else data["legacy_test_idx"][:n])
        targets = data[partition + "_target"][indices]
        histories = data[partition + "_history"][indices]
        cache_file = out / f"{partition}_candidates.npz"
        if not cache_file.exists() and source:
            with np.load(source/f'{partition}_candidates.npz') as packed:
                source_cache={key:packed[key] for key in packed.files}
            if not np.array_equal(indices,source_cache['indices'][:n]) or not np.array_equal(targets,source_cache['targets'][:n]):
                raise ValueError('Primary candidate source users/targets mismatch')
            offsets=source_cache['offsets'][:n+1]
            items=source_cache['items'][:offsets[-1]]
            raw=source_cache['raw'][:offsets[-1]]
            auxiliary=[]
            for start in range(0,n,64):
                users=torch.as_tensor(histories[start:start+64]+1,device='cuda',dtype=torch.long)
                states=hidden(assistant,'sasrec',users)
                for local,state in enumerate(states):
                    row=start+local;a,b=offsets[row:row+2]
                    item_ids=torch.as_tensor(items[a:b]+1,device='cuda',dtype=torch.long)
                    auxiliary.extend((assistant.item_embedding(item_ids)@state).float().cpu().numpy())
                if start//500 != (start+len(states))//500:print(partition,'auxiliary',start+len(states),n,flush=True)
            np.savez_compressed(cache_file,items=items,raw=raw,auxiliary=np.asarray(auxiliary),
                                offsets=offsets,indices=indices,targets=targets)
            save_json(out/f'{partition}_candidate_source.json',{'path':str(source/f'{partition}_candidates.npz'),
                      'sha256':sha256(source/f'{partition}_candidates.npz'),'users':n,'TIGER_candidates_unchanged':True})
        if not cache_file.exists():
            items_list, raw_list, aux_list, offsets = [], [], [], [0]
            for row, history in enumerate(histories):
                ids, mask = tokens_for_histories(history[None], data["item_tokens"], "cuda")
                enc = encode(model, ids, mask)
                result = beam_search(model, enc, mask, trie, prior["geometric"], args.beam)
                items, raw = result["items"], result["raw_scores"]
                user = torch.tensor(history[None] + 1, device="cuda", dtype=torch.long)
                state = hidden(assistant, "sasrec", user)[0]
                item_ids = torch.tensor(items + 1, device="cuda", dtype=torch.long)
                auxiliary = (assistant.item_embedding(item_ids) @ state).float().cpu().numpy()
                items_list.extend(items.tolist()); raw_list.extend(raw.tolist()); aux_list.extend(auxiliary.tolist())
                offsets.append(len(items_list))
                if (row + 1) % 500 == 0:
                    print(partition, row + 1, len(indices), flush=True)
            np.savez_compressed(cache_file, items=np.asarray(items_list), raw=np.asarray(raw_list),
                       auxiliary=np.asarray(aux_list), offsets=np.asarray(offsets), indices=indices, targets=targets)
        with np.load(cache_file) as packed:
            cache = {key: packed[key] for key in packed.files}
        if not np.array_equal(indices, cache["indices"]) or not np.array_equal(targets, cache["targets"]):
            raise ValueError("Candidate cache users/labels differ")
        tops = {"raw": []}
        tops.update({f"{name}:{b}": [] for name in prior for b in BETAS[1:]})
        tops.update({f"mmr:{w}": [] for w in weights})
        tops.update({f"fusion:{w}": [] for w in fusion_weights})
        for row in range(len(indices)):
            a, b = cache["offsets"][row:row+2]
            items, raw, auxiliary = cache["items"][a:b], cache["raw"][a:b], cache["auxiliary"][a:b]
            tops["raw"].append(stable_topk(raw, ids=items))
            for name, z in prior.items():
                for beta in BETAS[1:]:
                    tops[f"{name}:{beta}"].append(stable_topk(raw - beta * z[items], ids=items))
            similarity = emb[items] @ emb[items].T
            for weight in weights:
                tops[f"mmr:{weight}"].append(mmr(items, raw, similarity, weight))
            for weight in fusion_weights:
                tops[f"fusion:{weight}"].append(stable_topk((1-weight)*raw + weight*auxiliary, ids=items))
        rows, hits = {}, {}
        for name, values in tops.items():
            rows[name], hits[name] = metrics(np.asarray(values), targets, data["head_mask"], n_items)
            if name != "raw":
                rows[name]["paired_overall"] = paired_ci(hits["raw"], hits[name])
                tail = ~data["head_mask"][targets]
                rows[name]["paired_tail"] = paired_ci(hits["raw"][tail], hits[name][tail])
        save_json(out / f"{partition}_metrics.json", rows)
        np.savez_compressed(out / f"{partition}_top10.npz", indices=indices, targets=targets, **tops)
        if partition == "valid":
            selection = select_validation(rows, hits, targets, data["head_mask"])
            save_json(out / "validation_selection.json", selection)
        else:
            save_json(out / "selected_test.json", {name: {rule: {"method": choices[rule], "test": rows[choices[rule]]}
                      for rule in ["strict", "budget_5pct", "simultaneous_ci_budget_5pct"]}
                      for name, choices in selection.items()})
    save_json(out / "complete.json", {"finished_unix": time.time()})


if __name__ == "__main__":
    main()
