"""Verify saved width sweeps use identical users and reconstruct key metrics."""
import json
from pathlib import Path
import time
import numpy as np
from .common import load_bundle, save_json, sha256


def main():
    checked = []
    for folder in sorted(Path('runs').glob('*_widths')):
        if not (folder / 'complete.json').exists():
            continue
        cfg = json.loads((folder / 'config.json').read_text())
        primary = cfg['source_config']
        data = load_bundle(primary['split'], sid_length=primary['sid_length'])
        result = json.loads((folder / 'results.json').read_text())
        reference = None
        sources = {}
        for width, detail in result.items():
            path = folder / f'width_{width}_top10.npz'
            sources[path.name] = sha256(path)
            with np.load(path) as packed:
                indices, targets = packed['indices'], packed['targets']
                if reference is None:
                    reference = indices.copy()
                np.testing.assert_array_equal(indices, reference)
                np.testing.assert_array_equal(targets, data['test_target'][indices])
                np.testing.assert_array_equal(indices, data['legacy_test_idx'][:len(indices)])
                if len(np.unique(indices)) != len(indices):
                    raise ValueError('Duplicated users')
                tail = ~data['head_mask'][targets]
                raw_hits = (packed['raw'] == targets[:, None]).any(1)
                for method, values in detail['metrics'].items():
                    top = packed[method]
                    if top.shape != (len(indices), 10) or np.any(top < 0) or np.any(top >= len(data['pop_counts'])):
                        raise ValueError('Invalid recommendation identity or shape')
                    if np.any(np.diff(np.sort(top, axis=1), axis=1) == 0):
                        raise ValueError('Duplicated recommendation')
                    hits = (top == targets[:, None]).any(1)
                    actual = [hits.mean(), hits[tail].mean(), len(np.unique(top)) / len(data['pop_counts'])]
                    np.testing.assert_allclose(actual, [values[k] for k in ['R10', 'tailR10', 'cov10']], rtol=1e-12, atol=1e-12)
                    for key, mask in [('paired_overall', np.ones(len(hits), dtype=bool)), ('paired_tail', tail)]:
                        if key in values:
                            pair = values[key]
                            if pair['gained_hits'] != int((hits & ~raw_hits & mask).sum()) or pair['lost_hits'] != int((~hits & raw_hits & mask).sum()):
                                raise ValueError('Paired hit decomposition mismatch')
            for audit in detail['exact_audit']:
                if audit['top10_overlap'] > audit['candidate_inclusion'] + 1e-12:
                    raise ValueError('Overlap exceeds candidate inclusion')
        record = {'run': str(folder), 'passed': True, 'widths': list(result), 'users': len(reference),
            'sources': sources, 'results_sha256': sha256(folder / 'results.json'), 'finished_unix': time.time(),
            'scope': 'same users and targets across widths, recommendation identities, overall/tail recall, coverage, paired gained/lost hits, overlap bounded by inclusion'}
        save_json(folder / 'width_integrity_audit.json', record)
        checked.append(record)
    save_json(Path('runs/width_integrity_summary.json'), {'checked': checked, 'finished_unix': time.time()})
    print(json.dumps({'checked': len(checked), 'passed': True}))


if __name__ == '__main__':
    main()
