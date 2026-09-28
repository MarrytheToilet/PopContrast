"""Matched trie decoders and a prefix-bound research variant.

The research variant ranks a prefix by log p(prefix|u) + max descendant penalty.
This is an admissible upper bound on every corrected completion because future
token log-probabilities are nonpositive. Finite-width beam search remains
approximate. No claim of a new general search principle is made here.
"""
import heapq
from dataclasses import dataclass

import numpy as np
import torch

from .common import stable_topk


@dataclass
class Trie:
    children: dict
    leaves: dict
    min_prior: dict
    length: int
    potential: dict = None


def make_trie(item_tokens, prior):
    children, leaves, min_prior = {}, {}, {}
    for item, row in enumerate(item_tokens):
        row = tuple(int(t) for t in row)
        leaves.setdefault(row, []).append(item)
        for depth in range(len(row) + 1):
            prefix = row[:depth]
            min_prior[prefix] = min(min_prior.get(prefix, float("inf")), float(prior[item]))
            if depth < len(row):
                children.setdefault(prefix, set()).add(row[depth])
    return Trie({key: tuple(sorted(value)) for key, value in children.items()},
                leaves, min_prior, len(item_tokens[0]))


def marginal_potential(item_tokens, marginal, prior, beta):
    """Log expected terminal tilt under the offline within-prefix prior.

    h(v) = logsumexp_{i below v}(m_i - beta*z_i)
           - logsumexp_{i below v}(m_i).
    This is an approximate lookahead, not an admissible bound. It estimates
    future correction using model prior mass, avoiding a single extreme tail
    leaf dominating a whole branch. The final item objective stays unchanged.
    """
    base, tilted = {}, {}
    for row, value, penalty in zip(item_tokens, marginal, prior):
        row = tuple(int(t) for t in row)
        for depth in range(len(row) + 1):
            prefix = row[:depth]
            base[prefix] = np.logaddexp(base.get(prefix, -np.inf), value)
            tilted[prefix] = np.logaddexp(tilted.get(prefix, -np.inf), value - beta * penalty)
    return {prefix: float(tilted[prefix] - value) for prefix, value in base.items()}


@torch.no_grad()
def encode(model, input_ids, attention_mask):
    return model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]


@torch.no_grad()
def score_items(model, enc, mask, tokens, chunk=1024):
    values = []
    for start in range(0, len(tokens), chunk):
        labels = tokens[start:start + chunk]
        out = model(encoder_outputs=(enc.expand(len(labels), -1, -1),),
                    attention_mask=mask.expand(len(labels), -1), labels=labels)
        logp = torch.log_softmax(out.logits.float(), -1)
        values.append(logp.gather(-1, labels[..., None]).squeeze(-1).sum(-1))
    return torch.cat(values)


@torch.no_grad()
def score_items_cached(model, enc, mask, tokens, chunk=1024):
    """Teacher-forced SID scores with shared BOS/encoder attention caches.

    The original scorer repeats encoder key/value projections for every
    candidate. Here one user's immutable cross-attention cache is expanded
    as views; autoregressive self-attention updates produce new tensors.
    """
    first=model(encoder_outputs=(enc,),attention_mask=mask,
                decoder_input_ids=torch.zeros((1,1),dtype=torch.long,device=enc.device),use_cache=True)
    initial_logp=first.logits[0,-1].float().log_softmax(-1)
    if tokens.shape[1]==1:
        return initial_logp[tokens[:,0]]
    values=[]
    for start in range(0,len(tokens),chunk):
        labels=tokens[start:start+chunk]
        cache=tuple(tuple(tuple(tensor.expand(len(labels),*tensor.shape[1:]) for tensor in pair) for pair in layer)
                    for layer in first.past_key_values)
        output=model(encoder_outputs=(enc.expand(len(labels),-1,-1),),
                     attention_mask=mask.expand(len(labels),-1),decoder_input_ids=labels[:,:-1],
                     past_key_values=cache,use_cache=True)
        logp=output.logits.float().log_softmax(-1)
        values.append(initial_logp[labels[:,0]]+logp.gather(-1,labels[:,1:,None]).squeeze(-1).sum(-1))
    return torch.cat(values)


@torch.no_grad()
def next_logp(model, enc, mask, prefixes):
    decoder_ids = torch.tensor([(0,) + prefix for prefix in prefixes],
                               device=enc.device, dtype=torch.long)
    out = model(encoder_outputs=(enc.expand(len(prefixes), -1, -1),),
                attention_mask=mask.expand(len(prefixes), -1),
                decoder_input_ids=decoder_ids, use_cache=False)
    return torch.log_softmax(out.logits[:, -1].float(), -1).cpu().numpy()


@torch.no_grad()
def beam_search(model, enc, mask, trie, prior, width=20, beta=0.0, mode="raw"):
    if beta < 0 or mode not in {"raw", "bound", "terminal", "potential"}:
        raise ValueError("Nonnegative beta and supported search mode required")
    if mode == "potential" and trie.potential is None:
        raise ValueError("Potential search requires a precomputed marginal potential")
    prefixes, raw_scores, expanded = [()], np.zeros(1), 0
    for depth in range(trie.length):
        logp = next_logp(model, enc, mask, prefixes)
        expanded += len(prefixes)
        new_prefixes, raw, ranking = [], [], []
        for b, prefix in enumerate(prefixes):
            for token in trie.children[prefix]:
                child = prefix + (token,)
                score = float(raw_scores[b] + logp[b, token])
                new_prefixes.append(child)
                raw.append(score)
                if mode == "bound" or (mode in {"terminal", "potential"} and depth + 1 == trie.length):
                    ranking.append(score - beta * trie.min_prior[child])
                elif mode == "potential":
                    ranking.append(score + trie.potential[child])
                else:
                    ranking.append(score)
        # Lexicographic prefix order breaks equal-score pruning deterministically.
        order = sorted(range(len(raw)), key=lambda j: (-ranking[j], new_prefixes[j]))[:width]
        prefixes = [new_prefixes[j] for j in order]
        raw_scores = np.asarray(raw)[order]
    items, scores = [], []
    for prefix, raw in zip(prefixes, raw_scores):
        for item in trie.leaves[prefix]:
            items.append(item)
            scores.append(raw)
    items, scores = np.asarray(items), np.asarray(scores)
    corrected = scores - beta * prior[items]
    return {"items": items, "raw_scores": scores,
            "top": stable_topk(corrected, ids=items), "expanded_prefixes": expanded,
            "decoder_calls": trie.length}


@torch.no_grad()
def beam_search_batch(model, enc, mask, trie, prior, width=20, beta=0.0, mode="raw"):
    """Independent per-user beams, sharing forwards without sharing candidates.

    Prefix and item tie rules match beam_search. FP32 batched GEMMs can have
    tiny rounding differences; real-model parity is audited separately.
    """
    if beta < 0 or mode not in {"raw", "bound", "terminal", "potential"}:
        raise ValueError("Nonnegative beta and supported search mode required")
    if mode == "potential" and trie.potential is None:
        raise ValueError("Potential search requires a precomputed marginal potential")
    batches = [[()] for _ in range(len(enc))]
    values = [np.zeros(1) for _ in range(len(enc))]
    expanded = [0] * len(enc)
    for depth in range(trie.length):
        owners = torch.tensor([u for u, rows in enumerate(batches) for _ in rows], device=enc.device)
        decoder = torch.tensor([(0,) + prefix for rows in batches for prefix in rows], device=enc.device)
        out = model(encoder_outputs=(enc.index_select(0, owners),),
                    attention_mask=mask.index_select(0, owners), decoder_input_ids=decoder, use_cache=False)
        logp = torch.log_softmax(out.logits[:, -1].float(), -1).cpu().numpy()
        offset = 0
        for user, (prefixes, raw_scores) in enumerate(zip(batches, values)):
            expanded[user] += len(prefixes)
            new_prefixes, raw, ranking = [], [], []
            for beam, prefix in enumerate(prefixes):
                for token in trie.children[prefix]:
                    child = prefix + (token,)
                    score = float(raw_scores[beam] + logp[offset + beam, token])
                    new_prefixes.append(child)
                    raw.append(score)
                    if mode == "bound" or (mode in {"terminal", "potential"} and depth + 1 == trie.length):
                        ranking.append(score - beta * trie.min_prior[child])
                    elif mode == "potential":
                        ranking.append(score + trie.potential[child])
                    else:
                        ranking.append(score)
            offset += len(prefixes)
            order = sorted(range(len(raw)), key=lambda j: (-ranking[j], new_prefixes[j]))[:width]
            batches[user] = [new_prefixes[j] for j in order]
            values[user] = np.asarray(raw)[order]
    results = []
    for user, (prefixes, scores) in enumerate(zip(batches, values)):
        items, raw = [], []
        for prefix, value in zip(prefixes, scores):
            for item in trie.leaves[prefix]:
                items.append(item)
                raw.append(value)
        items, raw = np.asarray(items), np.asarray(raw)
        results.append({"items": items, "raw_scores": raw,
                        "top": stable_topk(raw - beta * prior[items], ids=items),
                        "expanded_prefixes": expanded[user], "decoder_calls": trie.length})
    return results


@torch.no_grad()
def certified_search(model, enc, mask, trie, prior, beta, k=10, max_expansions=50000):
    """Best-first reference with an explicit certificate or honest budget stop.

One prefix per forward is a correctness reference, not a low-latency claim.
Equal-score bounds are expanded to preserve exact item-ID tie breaking.
"""
    heap = [(beta * trie.min_prior[()], (), 0.0)]
    found = {}
    expansions, calls = 0, 0
    certified = False
    while heap:
        if len(found) >= k:
            ranked = sorted(found, key=lambda i: (-found[i], i))
            if -heap[0][0] < found[ranked[k - 1]]:
                certified = True
                break
        if expansions >= max_expansions:
            break
        _, prefix, raw = heapq.heappop(heap)
        if len(prefix) == trie.length:
            for item in trie.leaves[prefix]:
                found[item] = raw - beta * float(prior[item])
            continue
        lp = next_logp(model, enc, mask, [prefix])[0]
        expansions += 1
        calls += 1
        for token in trie.children[prefix]:
            child = prefix + (token,)
            score = raw + float(lp[token])
            upper = score - beta * trie.min_prior[child]
            heapq.heappush(heap, (-upper, child, score))
    if not heap:
        certified = True
    items = np.asarray(list(found), dtype=int)
    scores = np.asarray([found[i] for i in items])
    top = stable_topk(scores, k=k, ids=items)
    return {"top": top, "certified": certified, "expanded_prefixes": expansions,
            "decoder_calls": calls, "remaining_upper_bound": -heap[0][0] if heap else None}


@torch.no_grad()
def batched_best_first(model, enc, mask, trie, prior, beta, seed_candidates=None,
                       k=10, max_expansions=161, batch_size=32, numerical_tolerance=2e-5):
    """Anytime batched branch-and-bound, seeded by an existing raw beam.

    Bounds use nonpositive future log-probabilities and the descendant prior
    minimum. Batching may do extra work, but cannot invalidate a certificate.
    The budget counts *additional* prefix evaluations beyond the seed beam.
    This is a standard search construction, not a new general search claim.
    """
    if beta<0 or max_expansions<0 or batch_size<1:
        raise ValueError("Invalid search budget")
    queues=[[] for _ in range(trie.length)]
    heapq.heappush(queues[0],(beta*trie.min_prior[()],(),0.))
    found={} if seed_candidates is None else dict(zip(seed_candidates["items"].tolist(),seed_candidates["raw_scores"].tolist()))
    expansions=calls=0
    def ranked():
        return sorted(found,key=lambda item:(-(found[item]-beta*prior[item]),item))
    def remaining():
        return max((-q[0][0] for q in queues if q),default=-np.inf)
    certified=False
    while any(queues):
        order=ranked()
        threshold=found[order[k-1]]-beta*prior[order[k-1]] if len(order)>=k else -np.inf
        if remaining()<threshold-numerical_tolerance:
            certified=True;break
        if expansions>=max_expansions:break
        depth=min((d for d,q in enumerate(queues) if q),key=lambda d:queues[d][0])
        batch=[]
        while queues[depth] and len(batch)<min(batch_size,max_expansions-expansions):
            bound,prefix,raw=heapq.heappop(queues[depth])
            if -bound<threshold-numerical_tolerance:continue
            batch.append((prefix,raw))
        if not batch:continue
        probabilities=next_logp(model,enc,mask,[prefix for prefix,_ in batch])
        calls+=1;expansions+=len(batch)
        for (prefix,raw),lp in zip(batch,probabilities):
            for token in trie.children[prefix]:
                child=prefix+(token,);score=raw+float(lp[token])
                if len(child)==trie.length:
                    for item in trie.leaves[child]:found[item]=score
                else:
                    bound=score-beta*trie.min_prior[child]
                    if bound>=threshold-numerical_tolerance:
                        heapq.heappush(queues[depth+1],(-bound,child,score))
    if not any(queues):certified=True
    order=ranked();upper=remaining()
    threshold=found[order[k-1]]-beta*prior[order[k-1]] if len(order)>=k else -np.inf
    items=np.array(list(found),dtype=int)
    return {"top":np.array(order[:k],dtype=int),"items":items,"raw_scores":np.array([found[i] for i in items]),
            "certified":certified,"expanded_prefixes":expansions,"decoder_calls":calls,
            "remaining_upper_bound":float(upper) if np.isfinite(upper) else None,
            "model_score_gap_upper_bound":float(max(0.,upper-threshold)) if np.isfinite(threshold) else None}
