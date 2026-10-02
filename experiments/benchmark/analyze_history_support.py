"""Explore history support and fixed-candidate recovery using archived outputs.

No new recommender or calibration rule is selected. Existing validation choices
are held fixed. Regressions describe target-conditioned candidate populations;
they do not identify harmful bias, niche preference, or a causal mechanism.
Run: python -m experiments.benchmark.analyze_history_support
"""
from pathlib import Path
import hashlib
import json
import csv

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/benchmark'
RUNS = {'clothing': 'clothing_s1_l3', 'ml1m': 'ml1m_s1_l4',
        'sports': 'sports_s1_l3', 'toys': 'toys_s1_l3'}
OUT = BASE / 'history_support_analysis'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def clustered_fit(x, y, groups):
    """OLS with target-item clustered sandwich intervals; descriptive only."""
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coef
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    rank = np.linalg.matrix_rank(x)
    assert n_groups > 2 and len(y) > rank
    sums = np.zeros((n_groups, x.shape[1]))
    np.add.at(sums, inverse, x * residual[:, None])
    bread = np.linalg.pinv(x.T @ x)
    cov = bread @ (sums.T @ sums) @ bread
    cov *= n_groups / (n_groups - 1) * (len(y) - 1) / (len(y) - rank)
    half = t.ppf(.975, n_groups - 1) * np.sqrt(np.maximum(np.diag(cov), 0))
    return coef, half, int(n_groups)


def paired_delta(a, b, mask):
    values = b[mask].astype(int) - a[mask].astype(int)
    n = len(values)
    counts = np.array([(values == v).sum() for v in (-1, 0, 1)])
    rng = np.random.default_rng(20261002)
    draw = rng.multinomial(n, counts / n, size=4000)
    lo, hi = np.quantile((draw[:, 2] - draw[:, 0]) / n * 100, [.025, .975])
    return dict(n=n, lost=int(counts[0]), gained=int(counts[2]),
                delta_pp=float(values.mean()*100), ci95_pp=[float(lo), float(hi)])


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(
        datasets=RUNS, excluded={'beauty': 'No matching complete candidate cache in local run.'},
        reference_bank='All 128 archived histories; match stored priors and checkpoint metadata.',
        selection='Archived per-family budget_5pct validation choices; no new beta/estimator selection.',
        primary_signal='log(mean probability) minus mean(log probability), in training-catalog SD units',
        sensitivity_signal='log(M * sum(history_probability_weights ** 2)), in catalog SD units',
        controls=['log1p(training count)', 'squared log1p(training count)',
                  'raw target rank groups: 1, 2-5, 6-10, 11-15, 16-20, >20',
                  'raw target score minus score of tenth candidate'],
        populations=['reachable targets', 'reachable tail targets', 'reachable head targets'],
        outcome='Arithmetic hit minus geometric hit, with separately validation-selected strengths.',
        inference='Exploratory OLS, target-item clustered 95% intervals, no multiple-test adjustment.',
        transfer='Fit on validation; test reduction in MSE from adding primary signal to controls.',
        groups='Catalog-derived gap terciles; report both raw-miss recovery and raw-hit loss.',
        caveat='These datasets and outcomes were previously studied; this is not confirmatory evaluation.')
    save(OUT/'protocol.json', protocol)  # Write design before opening outcomes.
    sources, results = {}, {}

    def record(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    record(Path(__file__).resolve())
    for ds, run in RUNS.items():
        folder = BASE/'runs'/run/'evaluation'
        cfg = json.loads(record(folder/'config.json').read_text())
        resampling = json.loads(record(BASE/'estimator_resampling'/ds/'config.json').read_text())
        bank_path = record(folder/'sampled_history_scores.npz')
        assert sources[str(bank_path.relative_to(ROOT))] == resampling['background_sha256']
        assert cfg['checkpoint_sha256'] == resampling['checkpoint_sha256']
        with np.load(bank_path) as z:
            scores = z['scores'].astype(np.float64)
        geometric = scores.mean(0)
        arithmetic = logsumexp(scores, axis=0) - np.log(len(scores))
        gap = arithmetic-geometric
        log_weights = scores - logsumexp(scores, axis=0)
        concentration = np.log(len(scores)*np.exp(2*log_weights).sum(0))
        np.testing.assert_allclose(gap, (-np.log(len(scores))-log_weights).mean(0), atol=1e-11)
        with np.load(record(folder/'priors.npz')) as z:
            stored_priors = {f: z[f].astype(np.float64) for f in ['geometric', 'arithmetic', 'null']}
            for family, ref in [('geometric', geometric), ('arithmetic', arithmetic)]:
                np.testing.assert_allclose(ref, z[family], atol=2e-5, rtol=2e-6)
        with np.load(record(folder/'test_item_hits.npz')) as z:
            counts, head = z['pop_counts'], z['head_mask'].astype(bool)
        selection = json.loads(record(folder/'validation_selection.json').read_text())
        selected = {f: selection[f]['budget_5pct'] for f in ['geometric', 'arithmetic', 'null']}
        standardized_priors = {f: (v-v.mean())/max(v.std(ddof=1),1e-8) for f,v in stored_priors.items()}
        thresholds = np.quantile(gap, [1/3, 2/3])
        signals = {'gap': (gap-gap.mean())/gap.std(),
                   'concentration': (concentration-concentration.mean())/concentration.std()}
        partitions = {}
        for part in ['valid', 'test']:
            with np.load(record(folder/f'{part}_candidates.npz')) as z:
                targets, indices = z['targets'], z['indices']
                offsets, items, raw = z['offsets'], z['items'], z['raw']
            rank = np.full(len(targets), np.inf)
            margin = np.zeros(len(targets))
            raw_top = []
            corrected_top = {f: [] for f in selected}
            for n, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                ids, values = items[start:end], raw[start:end]
                order = np.lexsort((ids, -values))
                raw_top.append(ids[order[:10]])
                for fam, name in selected.items():
                    beta = 0. if name == 'raw' else float(name.split(':')[1])
                    adjusted = values - beta * standardized_priors[fam][ids]
                    corrected_top[fam].append(ids[np.lexsort((ids,-adjusted))[:10]])
                match = np.flatnonzero(ids[order] == targets[n])
                if len(match):
                    rank[n] = match[0]+1
                    margin[n] = values[order[match[0]]] - values[order[9]]
            hits = {}
            with np.load(record(folder/f'{part}_top10.npz')) as z:
                np.testing.assert_array_equal(targets, z['targets'])
                np.testing.assert_array_equal(indices, z['indices'])
                np.testing.assert_array_equal(raw_top, z['raw'])
                for fam, name in {'raw': 'raw', **selected}.items():
                    if fam != 'raw':
                        np.testing.assert_array_equal(corrected_top[fam], z[name])
                    hits[fam] = (z[name] == targets[:, None]).any(1)
                    assert not np.any(hits[fam] & ~np.isfinite(rank))
            metrics = json.loads(record(folder/f'{part}_metrics.json').read_text())
            for fam, name in {'raw': 'raw', **selected}.items():
                assert np.isclose(hits[fam].mean(), metrics[name]['R10'])
                assert np.isclose(hits[fam][~head[targets]].mean(), metrics[name]['tailR10'])
            lp = np.log1p(counts[targets])
            rank_group = np.digitize(rank, [1, 5, 10, 15, 20], right=True)
            x = np.column_stack([np.ones(len(targets)), lp, lp**2,
                                 *(rank_group == k for k in range(1, 6)), margin])
            partitions[part] = dict(targets=targets, rank=rank, hits=hits, x=x,
                                    tail=~head[targets], reachable=np.isfinite(rank))
        result = dict(histories=len(scores), items=len(counts), selected=selected,
                      gap_tercile_thresholds=thresholds.tolist(), paired={}, adjusted={}, grouped={})
        test = partitions['test']
        for cohort, mask in [('all', np.ones(len(test['targets']), dtype=bool)), ('tail', test['tail'])]:
            result['paired'][cohort] = {f'{a}_to_{b}': paired_delta(test['hits'][a], test['hits'][b], mask)
                for a,b in [('raw','geometric'), ('raw','arithmetic'), ('geometric','arithmetic'), ('geometric','null')]}
        for cohort in ['all', 'tail', 'head']:
            masks = {p: d['reachable'] & (d['tail'] if cohort=='tail' else ~d['tail'] if cohort=='head' else True)
                     for p,d in partitions.items()}
            tests = {}
            for signal, per_item in signals.items():
                fitted = {}
                for p, d in partitions.items():
                    mask = masks[p]
                    x, ids = d['x'][mask], d['targets'][mask]
                    y = d['hits']['arithmetic'][mask].astype(int)-d['hits']['geometric'][mask].astype(int)
                    xx = np.column_stack([x, per_item[ids]])
                    coef, half, clusters = clustered_fit(xx,y,ids)
                    fitted[p] = (coef, np.linalg.lstsq(x,y,rcond=None)[0])
                    tests[f'{signal}_{p}'] = dict(n=int(mask.sum()), item_clusters=clusters,
                        coefficient_pp_per_catalog_sd=float(coef[-1]*100),
                        ci95_pp=[float((coef[-1]-half[-1])*100),float((coef[-1]+half[-1])*100)])
                if signal == 'gap':
                    mask=masks['test'];ids=test['targets'][mask];x=test['x'][mask]
                    y=test['hits']['arithmetic'][mask].astype(int)-test['hits']['geometric'][mask].astype(int)
                    new, base=fitted['valid']
                    prediction=np.column_stack([x,per_item[ids]]) @ new
                    loss_reduction=(y-x@base)**2-(y-prediction)**2
                    mean, half, _=clustered_fit(np.ones((len(y),1)),loss_reduction,ids)
                    tests['validation_to_test_mse_reduction']=dict(mean=float(mean[0]),
                        ci95=[float(mean[0]-half[0]),float(mean[0]+half[0])],
                        baseline_mse=float(np.mean((y-x@base)**2)), augmented_mse=float(np.mean((y-prediction)**2)))
            result['adjusted'][cohort]=tests
            bins=np.digitize(gap[test['targets']],thresholds)
            groups=[]
            for group in range(3):
                mask=masks['test'] & (bins==group)
                raw_hit=mask & test['hits']['raw'];raw_miss=mask & ~test['hits']['raw']
                row=dict(gap_group=group,n=int(mask.sum()),raw_hits=int(raw_hit.sum()),raw_misses=int(raw_miss.sum()))
                for fam in ['geometric','arithmetic','null']:
                    gained=int((raw_miss & test['hits'][fam]).sum());lost=int((raw_hit & ~test['hits'][fam]).sum())
                    row[fam]=dict(gained=gained,lost=lost,net=gained-lost,
                        recovery_rate=gained/int(raw_miss.sum()) if raw_miss.any() else None,
                        loss_rate=lost/int(raw_hit.sum()) if raw_hit.any() else None)
                groups.append(row)
            result['grouped'][cohort]=groups
        results[ds]=result
        print(ds, selected, result['adjusted']['all']['gap_test'], flush=True)
    save(OUT/'results.json',dict(protocol=protocol,sources=sources,results=results))
    with (OUT/'adjusted_associations.csv').open('w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['dataset','population','partition','reachable_users','target_items',
                         'gap_coefficient_pp','ci95_low','ci95_high'])
        for dataset, result in results.items():
            for cohort, stats in result['adjusted'].items():
                for partition in ['valid','test']:
                    row=stats[f'gap_{partition}']
                    writer.writerow([dataset,cohort,partition,row['n'],row['item_clusters'],
                                     row['coefficient_pp_per_catalog_sd'],*row['ci95_pp']])


if __name__ == '__main__':
    main()
