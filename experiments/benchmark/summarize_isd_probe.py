"""Audit and summarize every fixed ISD probe arm, including independent confirmation."""
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT, paired_ci, save_json, sha256, stable_topk
from .evaluate import standardize
from .isd_probe import BASES, choose, evaluate


def main():
    output = ROOT/'reports/isd_probe'
    output.mkdir(parents=True, exist_ok=True)
    rows, comparisons, sources = [], [], {}
    caches = {}
    for dataset in ['beauty', 'toys']:
        with np.load(ROOT/f'results/benchmark/data/{dataset}.npz') as z:
            data = {k: z[k] for k in ['head_mask', 'pop_counts', 'test_history', 'test_target', 'legacy_marginal']}
        prior = standardize(data['legacy_marginal'])
        for phase, folder in [('pilot', 'isd_probe'), ('confirmation', 'isd_confirmation')]:
            base = ROOT/f'results/benchmark/{folder}/{dataset}'
            complete = json.loads((base/'complete.json').read_text())
            for name, key in [('protocol.json','protocol_sha256'), ('selection.json','selection_sha256'),
                              ('test_results.json','test_results_sha256')]:
                assert sha256(base/name) == complete[key]
            report = json.loads((base/'test_results.json').read_text())
            selection = json.loads((base/'selection.json').read_text())
            if phase == 'pilot':
                validation = json.loads((base/'valid_results.json').read_text())
                assert choose(validation['metrics']) == selection['selected']
            else:
                original = ROOT/selection['frozen_source']
                assert sha256(original) == selection['frozen_source_sha256']
                assert json.loads(original.read_text())['selected'] == selection['selected']
            with np.load(base/'test_predictions.npz') as z:
                packed = {k:z[k] for k in z.files}
            idx, target = packed.pop('indices'), packed.pop('targets')
            np.testing.assert_array_equal(target, data['test_target'][idx])
            caches[dataset, phase] = (idx, target, packed)
            for name, top in packed.items():
                metrics, hit = evaluate(top, target, data['head_mask'], len(prior))
                for key, value in metrics.items():
                    assert np.isclose(value, report['metrics'][name][key]), (dataset, name, key)
                search = 'isd' if name.startswith('isd') else 'raw'
                with np.load(base/f'test_{search}_candidates.npz') as z:
                    ids, offsets, score = z['items'], z['offsets'], z['raw']
                for u, picks in enumerate(top):
                    pool = ids[offsets[u]:offsets[u+1]]
                    chosen = picks[picks>=0]
                    assert len(chosen) == len(np.unique(chosen))
                    assert set(chosen) <= set(pool)
                    assert not set(chosen) & set(data['test_history'][idx[u]])
                    if name in ['raw','isd_search_raw'] or name.startswith(('popcontrast:', 'isd_search_popcontrast:')):
                        scores = score[offsets[u]:offsets[u+1]]
                        eligible = ~np.isin(pool, data['test_history'][idx[u]])
                        beta = float(name.split(':')[1]) if ':' in name else 0.
                        expected = stable_topk(scores[eligible]-beta*prior[pool[eligible]], ids=pool[eligible])
                        np.testing.assert_array_equal(chosen, expected)
                rows.append({'dataset':dataset, 'phase':phase, 'method':name, **metrics,
                             'tail_candidate_recall':report['candidate_recall'][search]['tail']})
            for path in base.iterdir():
                if path.suffix in ['.json','.npz']: sources[str(path.relative_to(ROOT))] = sha256(path)
        a, b = caches[dataset,'pilot'], caches[dataset,'confirmation']
        assert not set(a[0]) & set(b[0]), 'Confirmation overlaps pilot'
        assert set(a[2]) == set(b[2]), 'Selected methods changed'
        target = np.r_[a[1],b[1]]
        merged = {k:np.concatenate([a[2][k],b[2][k]]) for k in a[2]}
        for name, top in merged.items():
            metrics, _ = evaluate(top,target,data['head_mask'],len(prior))
            rows.append({'dataset':dataset,'phase':'combined','method':name, **metrics,
                         'tail_candidate_recall': None})
        for phase in ['pilot','confirmation','combined']:
            if phase == 'combined': targets, packed = target, merged
            else: _, targets, packed = caches[dataset,phase]
            tail = ~data['head_mask'][targets]
            for family, baseline in BASES.items():
                name = next(k for k in packed if k.startswith(family+':'))
                before = (packed[baseline] == targets[:,None]).any(axis=1)
                after = (packed[name] == targets[:,None]).any(axis=1)
                comparisons.append({'dataset':dataset,'phase':phase,'method':name,'baseline':baseline,
                                    'overall':paired_ci(before,after), 'tail':paired_ci(before[tail],after[tail])})
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(output/'audit.json',{'status':'passed','test_users_per_dataset':10000,
                                'pilot_users':3000,'independent_confirmation_users':7000,
                                'validation_users':1000,'new_training_runs':0,
                                'prediction_replay':True,'candidate_membership_and_masks':True,
                                'selection_frozen':True,'disjoint_test_cohorts':True,
                                'sources':sources,'comparisons':comparisons})
    text=['# ISD 与 PopContrast 快速验证', '',
          '这是一轮独立实现的探索实验，不是 ISD 原论文的同数据复现。使用现有旧 TIGER 权重和几何参照（PopContrast-G），无新训练。每个数据集一个旧 checkpoint，原训练种子未记录，因此不是多种子证据。',
          'Beauty、Toys 各用1,000验证用户锁定校正强度，随后测试3,000用户，再在不重叠的7,000用户上确认；确认阶段未重新选参数。',
          'ISD 使用 J=50、κ=60；beam 从原文50适配为20。共现窗口原文未明确数值，本次固定10、对称计数。所有方法共同过滤已观察历史物品。',
          '因此这里的原始召回与覆盖率不能直接替换正文表4（checkpoint、用户集合与过滤协议不同）。',
          '组合“完整 ISD + PC”先校正生成器得分，再执行原有 RRF；“ISD搜索 + PC”用校正得分替代最终 RRF，是独立消融。', '',
          '## 不重叠确认样本：7,000用户/数据集', '',
          '|数据集|方法|总体 R@10 (%)|长尾 R@10 (%)|长尾命中|Coverage@10 (%)|',
          '|---|---|---:|---:|---:|---:|']
    names={'raw':'原始 TIGER','ordering_only':'仅 ISD 排序','isd_search_raw':'仅 ISD 搜索','isd':'完整 ISD',
           'popcontrast':'原始候选 + PC','isd_search_popcontrast':'ISD 搜索 + PC','isd_popcontrast':'完整 ISD + PC'}
    for row in rows:
        if row['phase']!='confirmation':continue
        text.append(f"|{row['dataset']}|{names[row['method'].split(':')[0]]}|{100*row['R10']:.3f}|{100*row['tailR10']:.3f}|{row['tail_hits']}|{100*row['cov10']:.2f}|")
    text+=['','## 校正的增量（确认样本）','','区间为逐用户配对 bootstrap 95% 区间，单位为百分点；未作多重比较校正。','',
           '|数据集|比较|总体变化 [95% CI]|长尾变化 [95% CI]|','|---|---|---|---|']
    for r in comparisons:
        if r['phase']!='confirmation':continue
        def fmt(k):
            v=r[k];return f"{100*v['delta']:+.3f} [{100*v['ci95'][0]:+.3f}, {100*v['ci95'][1]:+.3f}]"
        text.append(f"|{r['dataset']}|{names[r['method'].split(':')[0]]} 相对 {names[r['baseline']]}|{fmt('overall')}|{fmt('tail')}|")
    total_seconds = sum(json.loads((ROOT/f'results/benchmark/{folder}/{dataset}/{partition}_results.json').read_text())['wall_seconds']
                        for folder in ['isd_probe','isd_confirmation'] for dataset in ['beauty','toys']
                        for partition in (['valid','test'] if folder=='isd_probe' else ['test']))
    text+=['','## 资源与结论范围','',
           f'全部验证、首轮测试和确认评估循环累计 {total_seconds:.1f} 秒；使用本地3090、两个CPU线程，CUDA进程显存分配限制为设备总量的12%（约3GB）。无新模型训练。该耗时包括批处理与统计排序，不作为在线延迟对比。',
           '优先考察完整 ISD 与完整 ISD + PopContrast-G 的配对增量。搜索消融用于说明候选生成与排序的作用，不能把“仅 ISD 搜索”当作完整基线来宣称优于 ISD。',
           '结果支持校正与 ISD 的互补性，不支持 PopContrast 单独优于完整 ISD。确认样本与首轮样本不重叠，但共享同一模型和验证选参；不能替代跨训练种子验证。',
           '', '全部首轮、确认及合并结果保存在 metrics.csv；选参、逐用户预测、候选和校验散列保存在结果目录及 audit.json。',
           '原文：https://arxiv.org/html/2607.24995。该原文用于实现方法，不提供本实验数值。','']
    (output/'RESULTS_zh.md').write_text('\n'.join(text))
    print(json.dumps({'status':'passed','rows':len(rows),'output':str(output)},ensure_ascii=False))


if __name__ == '__main__':main()
