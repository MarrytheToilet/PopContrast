"""Measure popularity coupling of the actual prior used by each complete run."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from .common import ROOT, save_json, sha256


def main():
    results = []
    for marker in sorted(Path('runs').rglob('complete.json')):
        folder = marker.parent
        if any(p.startswith('exploratory_') for p in folder.parts):
            continue
        config_path = folder / 'config.json'
        if not config_path.exists() or not (folder / 'test_metrics.json').exists():
            continue
        cfg = json.loads(config_path.read_text())
        train = cfg.get('training_config', cfg)
        split = train.get('split')
        if split is None:
            continue
        if (folder / 'priors.npz').exists():
            source = folder / 'priors.npz'
            with np.load(source) as packed:
                if 'geometric' not in packed:
                    continue
                prior = packed['geometric'].copy()
        elif (folder / 'marginal.npy').exists():
            source = folder / 'marginal.npy'
            prior = np.load(source)
        else:
            continue
        with np.load(ROOT / 'results/benchmark/data' / f'{split}.npz') as packed:
            counts = packed['pop_counts']
        if prior.shape != counts.shape or not np.isfinite(prior).all():
            raise ValueError('Invalid prior: ' + str(folder))
        value = {'run': str(folder), 'dataset': split,
            'rho_actual_geometric_prior_train_count': float(spearmanr(prior, counts).statistic),
            'items': len(counts), 'prior_source': str(source), 'prior_source_sha256': sha256(source),
            'train_counts_array_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
            'validation_or_test_targets_used': False, 'finished_unix': time.time(),
            'scope': 'actual stored prior used by this evaluation; correlation is descriptive, not a causal explanation or an applicability threshold'}
        save_json(folder / 'prior_coupling.json', value)
        results.append(value)
    save_json(Path('runs/prior_coupling_summary.json'), results)
    print(json.dumps({'completed_priors': len(results)}))


if __name__ == '__main__':
    main()
