"""Post-hoc cohort-size sensitivity, with frozen validation choices and models.

Reuse primary predictions for its 5,000 users and score the full complement.
No new method or parameter is selected from the expanded test cohort.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import load_bundle, load_model, metrics, paired_ci, save_json, sha256, stable_topk, tokens_for_histories
from .evaluate import standardize
from .search import beam_search_batch, encode, make_trie


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    source, out = Path(args.source), Path(args.output)
    if (out / 'complete.json').exists():
        return
    if not (source / 'complete.json').exists():
        raise ValueError('Primary evaluation must be complete')
    cfg = json.loads((source / 'config.json').read_text())
    if cfg['split'] != 'clothing' or cfg.get('beam') != 20 or cfg.get('precision', 'fp32') != 'fp32':
        raise ValueError('This extension is frozen to Clothing FP32 B20')
    if sha256(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    selection = json.loads((source / 'validation_selection.json').read_text())
    names = {'raw', 'geometric:0.75'} | {selection['geometric'][rule]
        for rule in ['strict', 'budget_5pct', 'simultaneous_ci_budget_5pct']}
    torch.set_num_threads(4)
    data = load_bundle('clothing', sid_length=cfg['sid_length'])
    save_json(out / 'config.json', {**cfg, 'source': str(source), 'batch_size': args.batch_size,
        'n_test': len(data['test_target']), 'primary_n_test': cfg.get('n_test', 5000),
        'cohort': 'all exported Clothing test users', 'new_test_selection': False,
        'role': 'posthoc_cohort_size_sensitivity', 'selection_sha256': sha256(source / 'validation_selection.json'),
        'primary_top10_sha256': sha256(source / 'test_top10.npz'), 'prior_sha256': sha256(source / 'priors.npz')})
    save_json(out / 'validation_selection.json', selection)
    indices = np.arange(len(data['test_target']))
    targets = data['test_target']
    tops = {name: np.full((len(indices), 10), -1, dtype=np.int64) for name in names}
    with np.load(source / 'test_top10.npz') as packed:
        existing = packed['indices']
        np.testing.assert_array_equal(packed['targets'], targets[existing])
        for name in names:
            tops[name][existing] = packed[name]
    remaining = np.setdiff1d(indices, existing)
    with np.load(source / 'priors.npz') as packed:
        prior = standardize(packed['geometric'])
    model = load_model(cfg['checkpoint'], sid_length=cfg['sid_length']).eval()
    trie = make_trie(data['item_tokens'], prior)
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start:start + args.batch_size]
        ids, mask = tokens_for_histories(data['test_history'][batch], data['item_tokens'], 'cuda')
        candidates = beam_search_batch(model, encode(model, ids, mask), mask, trie, prior, 20)
        for index, candidate in zip(batch, candidates):
            items, raw = candidate['items'], candidate['raw_scores']
            for name in names:
                score = raw if name == 'raw' else raw - float(name.split(':')[1]) * prior[items]
                tops[name][index] = stable_topk(score, ids=items)
        if start // 2000 != (start + len(batch)) // 2000:
            print('new users', start + len(batch), len(remaining), flush=True)
    if any((top < 0).any() for top in tops.values()):
        raise ValueError('Incomplete predictions')
    rows, hits = {}, {}
    tail = ~data['head_mask'][targets]
    for name in ['raw'] + sorted(names - {'raw'}):
        rows[name], hits[name] = metrics(tops[name], targets, data['head_mask'], len(prior))
        if name != 'raw':
            rows[name]['paired_overall'] = paired_ci(hits['raw'], hits[name])
            rows[name]['paired_tail'] = paired_ci(hits['raw'][tail], hits[name][tail])
    save_json(out / 'test_metrics.json', rows)
    np.savez_compressed(out / 'test_top10.npz', indices=indices, targets=targets, **tops)
    save_json(out / 'complete.json', {'finished_unix': time.time(), 'users': len(indices),
        'reused_primary_users': len(existing), 'new_users': len(remaining), 'new_test_selection': False})
    print(json.dumps({'raw': rows['raw'], 'selected': rows[selection['geometric']['budget_5pct']]}), flush=True)


if __name__ == '__main__':
    main()
