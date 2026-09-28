"""Independently recompute saved recommendation metrics on completed runs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .common import load_bundle, save_json, sha256


def close(actual, recorded, label):
    if not np.isclose(actual, recorded, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{label}: recomputed {actual}, recorded {recorded}')


def audit(folder):
    required = ['complete.json', 'config.json', 'test_top10.npz', 'test_metrics.json']
    if not all((folder / name).exists() for name in required):
        return None
    cfg = json.loads((folder / 'config.json').read_text())
    cfg = cfg.get('training_config', cfg)
    if 'split' not in cfg and (folder.parent / 'config.json').exists():
        cfg = json.loads((folder.parent / 'config.json').read_text())
    if 'split' not in cfg:
        return None
    output = folder / 'result_integrity_audit.json'
    fingerprints = {name: sha256(folder / name) for name in required +
                    ['valid_top10.npz', 'valid_metrics.json'] if (folder / name).exists()}
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get('source_sha256') == fingerprints:
            return previous
    data = load_bundle(cfg['split'], sid_length=cfg.get('sid_length', 3))
    n_items = len(data['item_tokens'])
    checked = []
    for partition in ['valid', 'test']:
        path = folder / f'{partition}_top10.npz'
        metric_path = folder / f'{partition}_metrics.json'
        if not path.exists() or not metric_path.exists():
            continue
        rows = json.loads(metric_path.read_text())
        with np.load(path) as packed:
            n = rows['raw']['n_users']
            fallback = (np.random.default_rng(7).permutation(len(data['valid_target']))[:n]
                        if partition == 'valid' else data['legacy_test_idx'][:n])
            indices = packed['indices'] if 'indices' in packed else fallback
            targets = data[partition + '_target'][indices]
            if len(np.unique(indices)) != n:
                raise ValueError('Repeated or missing users')
            if 'targets' in packed and not np.array_equal(packed['targets'], targets):
                raise ValueError('Stored targets disagree with split and user indices')
            head = data['head_mask'][targets]
            baseline = (packed['raw'] == targets[:, None]).any(1)
            for name, row in rows.items():
                top = packed[name]
                if top.shape != (n, 10) or np.any(top < 0) or np.any(top >= n_items):
                    raise ValueError(f'{partition}/{name}: invalid recommendation shape or item')
                if np.any(np.diff(np.sort(top, axis=1), axis=1) == 0):
                    raise ValueError(f'{partition}/{name}: duplicate recommendation')
                hit = (top == targets[:, None]).any(1)
                counts = np.bincount(top.ravel(), minlength=n_items)
                positive = counts[counts > 0] / counts.sum()
                cumulative = np.cumsum(np.sort(counts), dtype=np.float64) / counts.sum()
                values = {'hits': int(hit.sum()), 'R10': hit.mean(),
                    'head_users': int(head.sum()), 'tail_users': int((~head).sum()),
                    'headR10': hit[head].mean(), 'tailR10': hit[~head].mean(),
                    'cov10': np.count_nonzero(counts) / n_items,
                    'entropy': -np.sum(positive * np.log(positive)) / np.log(n_items),
                    'gini': 1 + 1 / n_items - 2 * cumulative.mean()}
                values['N10'] = sum(np.sum(top[:, k] == targets) / np.log2(k + 2)
                                   for k in range(10)) / n
                for key, value in values.items():
                    close(value, row[key], f'{partition}/{name}/{key}')
                for key, mask in [('paired_overall', np.ones(n, dtype=bool)), ('paired_tail', ~head)]:
                    if key not in row:
                        continue
                    a, b = baseline[mask], hit[mask]
                    expected = {'delta': b.mean() - a.mean(), 'n': len(a),
                        'gained_hits': int((b & ~a).sum()), 'lost_hits': int((a & ~b).sum())}
                    for field, value in expected.items():
                        close(value, row[key][field], f'{partition}/{name}/{key}/{field}')
                checked.append({'partition': partition, 'method': name, 'users': n})
    result = {'dataset': cfg['split'], 'passed': True, 'checked': checked,
        'source_sha256': fingerprints, 'finished_unix': time.time(),
        'scope': 'saved top10 identities, users, labels, point metrics and paired gain/loss counts; not a new uncertainty estimate'}
    save_json(output, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs')
    args = ap.parse_args()
    checked, failures = [], []
    for marker in sorted(Path(args.root).rglob('complete.json')):
        folder = marker.parent
        try:
            result = audit(folder)
            if result:
                checked.append({'run': str(folder), 'method_partitions': len(result['checked'])})
        except (ValueError, KeyError) as error:
            failures.append({'run': str(folder), 'error': str(error)})
    save_json(Path(args.root) / 'result_integrity_summary.json',
              {'checked': checked, 'failures': failures, 'finished_unix': time.time()})
    print(json.dumps({'checked': len(checked), 'failures': failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
