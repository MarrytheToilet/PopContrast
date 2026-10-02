"""Resource-bounded, matched ISD-statistics adaptation on existing TIGER models.

Source: https://arxiv.org/html/2607.24995, equations 13--17, appendix A.3.
This is an independent implementation on different datasets, with beam 20
instead of the source's 50. The unspecified co-consumption window is fixed at
10 positions; pairs are symmetric. All methods mask observed consumed items
after completion. Targets already consumed remain in the evaluation denominator.
No retraining or test-selected hyperparameters. Existing geometric priors are
reused only after checkpoint and dataset SHA256 verification.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix

from .common import (ROOT, load_bundle, load_model, paired_ci, save_json,
                     sha256, stable_topk, tokens_for_histories)
from .evaluate import BETAS, standardize
from .search import beam_search_batch, encode, make_trie


def build_statistics(data, window=10):
    """Each training event appears once, despite overlapping history windows."""
    n = len(data['pop_counts'])
    histories, targets = data['train_history'], data['train_target']
    _, first = np.unique(data['train_user'], return_index=True)
    initial = histories[first, -1]
    assert (initial >= 0).all()
    counts = np.bincount(np.r_[targets, initial], minlength=n).astype(float)
    transitions = coo_matrix((np.ones(len(targets)),
                             (histories[:, -1], targets)), shape=(n, n)).tocsr()
    transitions.data = np.log1p(transitions.data)
    prev = histories[:, -window:].ravel()
    dest = np.repeat(targets, window)
    keep = prev >= 0
    prev, dest = prev[keep], dest[keep]
    co = coo_matrix((np.ones(2 * len(prev)),
                    (np.r_[prev, dest], np.r_[dest, prev])), shape=(n, n)).tocsr()
    rows = np.repeat(np.arange(n), np.diff(co.indptr))
    co.data /= np.sqrt(np.maximum(counts[rows] * counts[co.indices], 1))
    return transitions, co, counts


def item_ranks(history, statistics, top_j=50):
    transitions, co, _ = statistics
    history = history[history >= 0]
    n = transitions.shape[0]
    eligible = np.ones(n, dtype=bool)
    eligible[history] = False
    if len(history):
        a = transitions[history[-1]].toarray().ravel()
        c = np.asarray(co[history[-10:]].mean(axis=0)).ravel()
    else:
        a = c = np.zeros(n)
    def z(x):
        sd = x[eligible].std()
        return (x - x[eligible].mean()) / sd if sd > 0 else np.zeros(n)
    score = .5 * z(a) + .5 * z(c)
    score[~eligible] = -np.inf
    ids = np.flatnonzero(eligible)
    top = stable_topk(score[ids], k=min(top_j, len(ids)), ids=ids)
    ranks = np.full(n, np.inf)
    ranks[top] = np.arange(1, len(top) + 1)
    return ranks, eligible


@torch.no_grad()
def isd_search_batch(model, enc, mask, trie, supports, width=20, kappa=60):
    beams = [[()] for _ in supports]
    values = [np.zeros(1) for _ in supports]
    for _ in range(trie.length):
        owners = torch.tensor([u for u, rows in enumerate(beams) for _ in rows], device=enc.device)
        decoder = torch.tensor([(0,) + p for rows in beams for p in rows], device=enc.device)
        out = model(encoder_outputs=(enc.index_select(0, owners),),
                    attention_mask=mask.index_select(0, owners), decoder_input_ids=decoder,
                    use_cache=False)
        logp = out.logits[:, -1].float().log_softmax(-1).cpu().numpy()
        offset = 0
        for u, (prefixes, scores) in enumerate(zip(beams, values)):
            children, raw = [], []
            for b, prefix in enumerate(prefixes):
                for token in trie.children[prefix]:
                    children.append(prefix + (token,))
                    raw.append(scores[b] + float(logp[offset + b, token]))
            offset += len(prefixes)
            decoder_order = sorted(range(len(raw)), key=lambda j: (-raw[j], children[j]))
            rank = np.empty(len(raw), dtype=float)
            rank[decoder_order] = np.arange(1, len(raw) + 1)
            item_rank = np.array([supports[u].get(p, np.inf) for p in children])
            fused = 1 / (kappa + rank) + 1 / (kappa + item_rank)
            order = sorted(range(len(raw)), key=lambda j: (-fused[j], children[j]))[:width]
            beams[u] = [children[j] for j in order]
            values[u] = np.asarray(raw)[order]
    results = []
    for prefixes, scores in zip(beams, values):
        ids, raw = [], []
        for p, s in zip(prefixes, scores):
            ids.extend(trie.leaves[p]); raw.extend([s] * len(trie.leaves[p]))
        results.append({'items': np.asarray(ids), 'raw_scores': np.asarray(raw)})
    return results


def rerank(items, raw, ranks, eligible, prior, beta=0., fusion=False, kappa=60):
    keep = eligible[items]
    ids, score = items[keep], raw[keep] - beta * prior[items[keep]]
    if fusion and len(ids):
        order = np.lexsort((ids, -score))
        decoder_rank = np.empty(len(ids), dtype=float)
        decoder_rank[order] = np.arange(1, len(ids) + 1)
        score = 1 / (kappa + decoder_rank) + 1 / (kappa + ranks[ids])
    top = stable_topk(score, ids=ids)
    return np.pad(top, (0, 10 - len(top)), constant_values=-1)


def evaluate(top, targets, head_mask, n_items):
    top = np.asarray(top)
    hit = (top == targets[:, None]).any(axis=1)
    tail = ~head_mask[targets]
    ndcg = hit / np.log2((top == targets[:, None]).argmax(axis=1) + 2)
    counts = np.bincount(top[top >= 0], minlength=n_items)
    return {'users': len(targets), 'hits': int(hit.sum()), 'tail_users': int(tail.sum()),
            'tail_hits': int(hit[tail].sum()), 'R10': float(hit.mean()),
            'N10': float(ndcg.mean()), 'tailR10': float(hit[tail].mean()),
            'headR10': float(hit[~tail].mean()),
            'cov10': float(np.count_nonzero(counts) / n_items)}, hit


BASES = {'popcontrast': 'raw', 'isd_search_popcontrast': 'isd_search_raw',
         'isd_popcontrast': 'isd'}


def choose(rows):
    selected = {}
    for family, base in BASES.items():
        names = [f'{family}:{beta}' for beta in BETAS]
        feasible = [k for k in names if rows[k]['R10'] >= .95 * rows[base]['R10'] - 1e-12]
        selected[family] = max(feasible, key=lambda k: (rows[k]['tailR10'], rows[k]['R10'],
                                                       -float(k.split(':')[1])))
    return selected


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.output) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / f'genrec/out/tiger/amazon/{args.dataset}/best_model.pt'
    bundle = ROOT / f'results/benchmark/data/{args.dataset}.npz'
    meta = json.loads(bundle.with_suffix('.json').read_text())
    assert sha256(checkpoint) == meta['checkpoint_sha256'], 'Legacy prior/checkpoint mismatch'
    assert sha256(bundle) == meta['bundle_sha256'], 'Dataset changed'
    data = load_bundle(args.dataset)
    model = load_model(checkpoint, device=args.device).eval()
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.12)
    prior = standardize(data['legacy_marginal'])
    trie = make_trie(data['item_tokens'], prior)
    started = time.perf_counter()
    statistics = build_statistics(data)
    statistics_seconds = time.perf_counter() - started
    protocol = {**vars(args), 'checkpoint_sha256': sha256(checkpoint), 'bundle_sha256': sha256(bundle),
                'runner_sha256': sha256(Path(__file__)),
                'source': 'https://arxiv.org/html/2607.24995', 'J': 50, 'kappa': 60,
                'history_window': 10, 'co_consumption_window': 10, 'co_consumption': 'symmetric pairs',
                'source_window_ambiguity': 'Paper does not numerically specify w; fixed to 10 before evaluation',
                'mask': 'observed-history items removed from Q and every completed-candidate ranking',
                'beam_candidates': 'same catalog trie; consumed items not pruned before SID completion',
                'selection': 'maximize validation tailR10 at >=95% own uncorrected pipeline R10; ties overall then smaller beta',
                'betas': BETAS, 'combination': 'correct decoder score before final RRF; preserve prefix support',
                'geometric_prior': 'checkpoint-matched existing 512-history reference; no new estimation',
                'statistics_seconds': statistics_seconds, 'torch': torch.__version__,
                'timings': 'exploratory batched shared-device timings, not a serving benchmark',
                'status': 'protocol fixed before validation and test'}
    save_json(out / 'protocol.json', protocol)
    selected = None
    if args.selection_source:
        source = Path(args.selection_source)
        original = json.loads(source.read_text())
        source_protocol = json.loads((source.parent/'protocol.json').read_text())
        for key in ['checkpoint_sha256', 'bundle_sha256', 'beam', 'J', 'kappa',
                    'mask', 'betas', 'history_window', 'co_consumption_window', 'combination']:
            assert protocol[key] == source_protocol[key], f'Confirmation protocol changed: {key}'
        selected = original['selected']
        save_json(out/'selection.json', {**original, 'frozen_source': str(source),
                                        'frozen_source_sha256': sha256(source)})
    partitions = [('test', args.n_test)] if selected is not None else [('valid', args.n_valid), ('test', args.n_test)]
    for partition, n in partitions:
        offset = args.test_offset if partition == 'test' else 0
        indices = np.random.default_rng(20261002 if partition == 'valid' else 20261003).permutation(
            len(data[f'{partition}_target']))[offset:offset+n]
        assert len(indices) == n
        targets = data[f'{partition}_target'][indices]
        histories = data[f'{partition}_history'][indices]
        settings = {'raw': (False, 0., False), 'ordering_only': (False, 0., True),
                    'isd_search_raw': (True, 0., False), 'isd': (True, 0., True)}
        for family in BASES:
            betas = BETAS if selected is None else [float(selected[family].split(':')[1])]
            for beta in betas:
                settings[f'{family}:{beta}'] = (family != 'popcontrast', beta, family == 'isd_popcontrast')
        predictions = {k: [] for k in settings}
        available = {'raw': [], 'isd': []}
        item_lists, raw_lists, offsets = {'raw': [], 'isd': []}, {'raw': [], 'isd': []}, {'raw': [0], 'isd': [0]}
        durations = {'statistics': 0., 'encoder': 0., 'raw_search': 0., 'isd_search': 0.}
        def sync():
            if args.device.startswith('cuda'): torch.cuda.synchronize()
        start_partition = time.perf_counter()
        for start in range(0, n, args.batch_size):
            batch = histories[start:start+args.batch_size]
            t = time.perf_counter()
            rank_masks = [item_ranks(h, statistics) for h in batch]
            supports = []
            for ranks, _ in rank_masks:
                support = {}
                for item in np.flatnonzero(np.isfinite(ranks)):
                    sid = tuple(data['item_tokens'][item])
                    for depth in range(1, len(sid) + 1):
                        p = sid[:depth]; support[p] = min(support.get(p, np.inf), ranks[item])
                supports.append(support)
            durations['statistics'] += time.perf_counter() - t
            x, mask = tokens_for_histories(batch, data['item_tokens'], args.device)
            sync(); t = time.perf_counter(); enc = encode(model, x, mask); sync()
            durations['encoder'] += time.perf_counter() - t
            t = time.perf_counter(); raw_results = beam_search_batch(model, enc, mask, trie, prior, args.beam); sync()
            durations['raw_search'] += time.perf_counter() - t
            t = time.perf_counter(); isd_results = isd_search_batch(model, enc, mask, trie, supports, args.beam); sync()
            durations['isd_search'] += time.perf_counter() - t
            if start == 0:
                disabled = isd_search_batch(model, enc, mask, trie, [{} for _ in batch], args.beam)
                for raw, check in zip(raw_results, disabled):
                    assert np.array_equal(raw['items'], check['items']), 'Empty support must match raw beam'
                    assert np.allclose(raw['raw_scores'], check['raw_scores'], atol=1e-7)
            for local, (raw, isd, (ranks, eligible)) in enumerate(zip(raw_results, isd_results, rank_masks)):
                for name, result in [('raw', raw), ('isd', isd)]:
                    ids = result['items']
                    available[name].append(bool(targets[start+local] in ids[eligible[ids]]))
                    item_lists[name].extend(ids); raw_lists[name].extend(result['raw_scores'])
                    offsets[name].append(len(item_lists[name]))
                for key, (search, beta, fusion) in settings.items():
                    r = isd if search else raw
                    predictions[key].append(rerank(r['items'], r['raw_scores'], ranks, eligible, prior, beta, fusion))
            if start % 200 == 0:
                print(args.dataset, partition, min(start+len(batch), n), '/', n,
                      'seconds', round(time.perf_counter()-start_partition, 1), flush=True)
        rows, hits = {}, {}
        for key, top in predictions.items():
            rows[key], hits[key] = evaluate(top, targets, data['head_mask'], len(prior))
        tail = ~data['head_mask'][targets]
        for key in rows:
            base = BASES.get(key.split(':')[0], 'raw')
            rows[key]['comparison_base'] = base
            rows[key]['paired_overall'] = paired_ci(hits[base], hits[key])
            rows[key]['paired_tail'] = paired_ci(hits[base][tail], hits[key][tail])
            rows[key]['test_budget_violation'] = bool(rows[key]['R10'] < .95*rows[base]['R10'] - 1e-12)
        ceilings = {k: {'overall': float(np.mean(v)), 'tail': float(np.mean(np.asarray(v)[tail])),
                        'tail_targets': int(np.asarray(v)[tail].sum())} for k, v in available.items()}
        if partition == 'valid':
            for family, base in BASES.items():
                assert np.array_equal(predictions[f'{family}:0.0'], predictions[base])
            selected = choose(rows)
            save_json(out/'selection.json', {'partition': 'valid', 'selected': selected,
                                            'protocol_sha256': sha256(out/'protocol.json')})
            print('LOCKED', args.dataset, selected, flush=True)
        report = {'metrics': rows, 'candidate_recall': ceilings,
                  'target_already_consumed': int(sum(t in h for t, h in zip(targets, histories))),
                  'seconds': durations, 'wall_seconds': time.perf_counter()-start_partition,
                  'selection_sha256': sha256(out/'selection.json'), 'selected': selected}
        save_json(out/f'{partition}_results.json', report)
        np.savez_compressed(out/f'{partition}_predictions.npz', indices=indices, targets=targets,
                            **{k: np.asarray(v) for k,v in predictions.items()})
        for name in available:
            np.savez_compressed(out/f'{partition}_{name}_candidates.npz', indices=indices, targets=targets,
                                items=np.asarray(item_lists[name]), raw=np.asarray(raw_lists[name]),
                                offsets=np.asarray(offsets[name]))
        print('FINISHED', args.dataset, partition, report['wall_seconds'], flush=True)
    save_json(out/'complete.json', {'protocol_sha256': sha256(out/'protocol.json'),
                                  'selection_sha256': sha256(out/'selection.json'),
                                  'test_results_sha256': sha256(out/'test_results.json'),
                                  'new_training_runs': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['beauty', 'toys'], required=True)
    p.add_argument('--output', default='results/benchmark/isd_probe')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-valid', type=int, default=1000)
    p.add_argument('--n-test', type=int, default=3000)
    p.add_argument('--beam', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--selection-source', help='Frozen validation selection for an independent confirmation cohort')
    p.add_argument('--test-offset', type=int, default=0)
    run(p.parse_args())
