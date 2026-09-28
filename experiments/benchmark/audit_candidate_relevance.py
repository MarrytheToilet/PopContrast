"""Separate relevant-candidate availability from ranking within fixed beams.

Uses held-out targets retrospectively. These are oracle ceilings and accounting
identities, not label-free applicability predictors or causal explanations.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .common import load_bundle, save_json, sha256


def audit(folder):
    required = ['complete.json', 'config.json', 'validation_selection.json',
                'test_candidates.npz', 'valid_candidates.npz']
    if not all((folder / name).exists() for name in required):
        return None
    if not (folder / 'test_metrics.json').exists():
        return None
    if json.loads((folder / 'test_metrics.json').read_text()).get('raw', {}).get('n_users') != 5000:
        return None
    cfg = json.loads((folder / 'config.json').read_text())
    train = cfg.get('training_config', cfg)
    if 'split' not in train:
        return None
    out = folder / 'candidate_relevance_audit.json'
    if out.exists():
        return json.loads(out.read_text())
    data = load_bundle(train['split'], sid_length=train.get('sid_length', 3))
    selection = json.loads((folder / 'validation_selection.json').read_text())
    if 'geometric' not in selection:
        return None
    result = {'dataset': train['split'], 'sid_length': train.get('sid_length', 3),
        'interpretation': 'label-aware retrospective oracle ceilings; no parameter selection',
        'identity': 'tail recall = P(target in raw beam | tail) * P(hit | tail and target in raw beam)',
        'source_sha256': {name: sha256(folder / name) for name in required}, 'partitions': {}}
    for partition in ['valid', 'test']:
        rows = json.loads((folder / f'{partition}_metrics.json').read_text())
        with np.load(folder / f'{partition}_candidates.npz') as packed:
            targets, offsets, items = packed['targets'], packed['offsets'], packed['items']
            if not np.array_equal(targets, data[partition + '_target'][packed['indices']]):
                raise ValueError('Candidate labels disagree with user indices')
            available = np.array([target in items[a:b]
                for target, a, b in zip(targets, offsets[:-1], offsets[1:])])
        tail = ~data['head_mask'][targets]
        ceiling = float(available[tail].mean())
        names = {'raw'}
        for family in ['geometric', 'arithmetic', 'null']:
            for rule in ['strict', 'budget_5pct', 'simultaneous_ci_budget_5pct']:
                if family in selection:
                    names.add(selection[family][rule])
        methods = {}
        with np.load(folder / f'{partition}_top10.npz') as top:
            for name in sorted(names):
                hit = (top[name] == targets[:, None]).any(1)
                if np.any(hit & ~available):
                    raise ValueError('A fixed-candidate method hit outside its raw beam')
                conditional = float(hit[tail & available].mean()) if (tail & available).any() else None
                if conditional is not None and not np.isclose(ceiling * conditional, rows[name]['tailR10']):
                    raise ValueError('Tail recall accounting mismatch')
                methods[name] = {'tail_hits': int(hit[tail].sum()),
                    'tail_recall': rows[name]['tailR10'],
                    'conditional_tail_hit_rate_given_reachable': conditional}
        result['partitions'][partition] = {'users': len(targets), 'tail_users': int(tail.sum()),
            'tail_targets_available': int((tail & available).sum()), 'candidate_tail_recall_ceiling': ceiling,
            'candidate_overall_recall_ceiling': float(available.mean()),
            'candidate_count_quantiles': np.quantile(np.diff(offsets), [0, .5, .9, 1]).tolist(),
            'methods': methods}
    result['finished_unix'] = time.time()
    save_json(out, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs')
    args = ap.parse_args()
    summaries = []
    for marker in sorted(Path(args.root).rglob('complete.json')):
        result = audit(marker.parent)
        if result:
            summaries.append({'run': str(marker.parent), 'dataset': result['dataset'],
                              'test': result['partitions']['test']})
    save_json(Path(args.root) / 'candidate_relevance_summary.json', summaries)
    print(json.dumps({'audited_evaluations': len(summaries)}))


if __name__ == '__main__':
    main()
