"""Verify published summaries, selected strengths, source hashes, and plot inputs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark.settings import PRIMARY_RUNS


def main():
    base = ROOT/'results/benchmark'
    rows = json.loads((base/'completed_results.json').read_text())
    assert len(rows)==5*len(PRIMARY_RUNS)==165
    assert set(r['run'] for r in rows)==set(PRIMARY_RUNS)
    sources = json.loads((base/'summary_sources.json').read_text())
    for name,sha in sources['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    identities = set()
    comparisons = 0
    for run in PRIMARY_RUNS:
        folder = base/'runs'/run
        metrics = json.loads((folder/'test_metrics.json').read_text())
        selection = json.loads((folder/'validation_selection.json').read_text())['geometric']
        for row in [r for r in rows if r['run']==run]:
            assert row['n_users']==5000
            if row['rule'] not in ['raw','fixed_beta_0.75']:
                assert row['method']==selection[row['rule']],run
            original=metrics[row['method']]
            for key in ['R10','N10','tailR10','headR10','cov10','entropy','gini']:
                assert row[key]==original[key],(run,key)
                comparisons+=1
            for metric in ['tail','overall']:
                pair=original.get('paired_'+metric,{})
                assert row[metric+'_ci95']==pair.get('ci95'),(run,metric)
            if row['rule']=='raw':
                identity=(row['dataset'],row['checkpoint_sha256'] or run)
                assert identity not in identities,run
                identities.add(identity)
    for file in ['figdata_beauty.npz','figdata_sports.npz','figdata_toys.npz','rankshift_beauty.npz']:
        assert (ROOT/'assets/plot_data'/file).is_file(),file
    print(f'Passed: {len(identities)} independent settings, {comparisons} metric comparisons, {len(sources["sources"])} source hashes, all selected strengths and intervals.')


if __name__=='__main__':
    main()
