"""Verify published summaries, selected strengths, source hashes, and plot inputs."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark.settings import PRIMARY_RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true',
                        help='Check that the separately retained analysis inputs are present.')
    args = parser.parse_args()
    base = ROOT/'results/benchmark'
    required = [base/'runs'/run/name for run in PRIMARY_RUNS
                for name in ['test_metrics.json', 'validation_selection.json', 'config.json', 'complete.json']]
    required += [ROOT/'assets/plot_data'/name for name in
                 ['figdata_beauty.npz', 'figdata_sports.npz', 'figdata_toys.npz', 'rankshift_beauty.npz']]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        parser.exit(1, 'Analysis inputs are stored separately from Git. Restore results/ and '
                    'assets/plot_data/ from your local analysis archive before this command.\n'
                    f'Missing {len(missing)} required files; first: {missing[0]}\n')
    if args.check_inputs:
        print('Local analysis inputs found.')
        return
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
