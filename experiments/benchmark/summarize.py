"""Rebuild publication summaries from completed, frozen benchmark records."""
import csv
import hashlib
import json
from pathlib import Path
from .settings import PRIMARY_RUNS, BASELINE_RUNS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/benchmark'
RUNS = BASE / 'runs'
SOURCES = {}
RULES = ('strict', 'budget_5pct', 'simultaneous_ci_budget_5pct')


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    records = []
    for run in PRIMARY_RUNS:
        folder = RUNS / run
        complete = read(folder / 'complete.json')
        metrics = read(folder / 'test_metrics.json')
        config = read(folder / 'config.json')
        training = read(folder.parent / 'config.json') if folder.name == 'evaluation' else config
        training = config.get('training_config', training)
        model = training.get('model', 'TIGER')
        if 'base_model' in training:
            model = Path(training['base_model']).name + '+LoRA(seqrec)'
        selected = read(folder / 'validation_selection.json')['geometric']
        methods = {'raw': 'raw', 'fixed_beta_0.75': 'geometric:0.75', **{r: selected[r] for r in RULES}}
        for rule, name in methods.items():
            value = metrics[name]
            assert value['n_users'] == 5000, (run, rule)
            pair = value.get('paired_tail', {})
            overall = value.get('paired_overall', {})
            checkpoint = config.get('checkpoint_sha256', complete.get('checkpoint_sha256'))
            if checkpoint is None and (folder/'prior_geometry.json').exists():
                checkpoint = read(folder/'prior_geometry.json').get('checkpoint_sha256')
            records.append(dict(run=run, dataset=training.get('split', config.get('split')),
                model=model, checkpoint_sha256=checkpoint,
                evaluation_precision=config.get('precision', 'float32' if 'base_model' not in training else 'bf16'),
                role='primary_or_baseline', seed=training.get('seed', 'unrecorded'),
                sid_length=training.get('sid_length', config.get('sid_length', 'n/a')),
                length_grouping=training.get('length_grouping', False) if model=='TIGER' and 'seed' in training else training.get('length_grouping'),
                rule=rule, method=name, n_users=value['n_users'],
                **{k:value.get(k) for k in ['R10','tailR10','headR10','N10','cov10','entropy','gini','hits','tail_users']},
                overall_delta=overall.get('delta', 0 if name=='raw' else None), overall_ci95=overall.get('ci95'),
                tail_delta=pair.get('delta', 0 if name=='raw' else None), tail_ci95=pair.get('ci95'),
                source=str((folder/'test_metrics.json').relative_to(ROOT))))
    write_json(BASE/'completed_results.json', records)
    write_csv(BASE/'completed_results.csv', records)
    baselines = []
    for run in BASELINE_RUNS:
        folder = RUNS/run
        metrics = read(folder/'test_metrics.json')
        selected = read(folder/'validation_selection.json')
        config = read(folder/'config.json')
        for family in ['raw','geometric','arithmetic','null','logcount','mmr','fusion']:
            for rule in RULES:
                name = 'raw' if family=='raw' else selected[family][rule]
                m = metrics[name]
                baselines.append(dict(run=run, dataset=config['split'], family=family, rule=rule,
                    method=name, n_users=m['n_users'], **{k:m[k] for k in ['R10','tailR10','cov10']},
                    source=str((folder/'test_metrics.json').relative_to(ROOT))))
    write_csv(BASE/'matched_baseline_comparisons.csv', baselines)
    full = []
    for seed in range(3):
        folder = RUNS/f'full_users_clothing_s{seed}'
        config = read(folder/'config.json')
        primary = RUNS/config['source'].removeprefix('runs/')
        selection = read(folder/'validation_selection.json')['geometric']
        methods = {**{r:selection[r] for r in RULES}, 'fixed_beta_0.75':'geometric:0.75'}
        for location, cohort in [(primary,'primary_5000'), (folder,'all_users')]:
            metrics = read(location/'test_metrics.json')
            for rule,name in methods.items():
                m = metrics[name]
                pair = m.get('paired_tail', dict(delta=0.,ci95=[0.,0.],gained_hits=0,lost_hits=0))
                full.append(dict(run=folder.name,cohort=cohort,users=m['n_users'],rule=rule,method=name,
                    **{k:m[k] for k in ['R10','tailR10','cov10']},tail_delta=pair['delta'],tail_ci95=pair['ci95'],
                    gained_tail_hits=pair['gained_hits'],lost_tail_hits=pair['lost_hits']))
    write_csv(BASE/'full_user_sensitivity.csv',full)
    write_json(BASE/'summary_sources.json',dict(preferred_runs=PRIMARY_RUNS,sources=SOURCES,
        precision_preference='Complete FP32 evaluations; independent of test recommendation performance',
        summary_source_sha256=hashlib.sha256((BASE/'completed_results.json').read_bytes()).hexdigest()))
    print(f'Rebuilt {len(PRIMARY_RUNS)} primary settings, {len(baselines)} matched baseline rows, and {len(full)} cohort rows.')


if __name__ == '__main__':
    main()
