"""Paired estimator and user-segment audits at frozen validation choices.

Reads completed outputs only. This does not choose settings using test labels.
All intervals condition on one checkpoint; they are not training-seed intervals.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .common import load_bundle,paired_ci,save_json,sha256
from .evaluate import group_analysis


def audit(folder):
    out=folder/'selected_estimator_audit.json'
    required=['complete.json','test_top10.npz','test_metrics.json','validation_selection.json','config.json']
    if not all((folder/name).exists() for name in required):return False
    if out.exists() and json.loads(out.read_text()).get('audit_version',1)>=2:return False
    rows=json.loads((folder/'test_metrics.json').read_text())
    if rows.get('raw',{}).get('n_users')!=5000:return False
    cfg=json.loads((folder/'config.json').read_text())
    train=cfg.get('training_config',cfg)
    if 'split' not in train and (folder.parent/'config.json').exists():
        train=json.loads((folder.parent/'config.json').read_text())
    data=load_bundle(train['split'],sid_length=train.get('sid_length',3))
    selection=json.loads((folder/'validation_selection.json').read_text())
    if 'geometric' not in selection:return False
    with np.load(folder/'test_top10.npz') as f:
        indices=f['indices'] if 'indices' in f else data['legacy_test_idx'][:5000]
        targets=f['targets'] if 'targets' in f else data['test_target'][indices]
        if not np.array_equal(targets,data['test_target'][indices]):raise ValueError('Target/index mismatch')
        names={'raw'}
        for family in ['geometric','arithmetic','null','logcount','poprank','mmr','fusion']:
            for rule in ['strict','budget_5pct','simultaneous_ci_budget_5pct']:
                if family in selection:names.add(selection[family][rule])
        top={name:f[name] for name in names}
    hits={name:(values==targets[:,None]).any(1) for name,values in top.items()}
    for name,hit in hits.items():
        if int(hit.sum())!=rows[name]['hits']:raise ValueError('Hit-count mismatch: '+name)
    tail=~data['head_mask'][targets]
    result={}
    for rule in ['strict','budget_5pct','simultaneous_ci_budget_5pct']:
        reference=selection['geometric'][rule]
        comparisons={}
        for family in ['geometric','arithmetic','null','logcount','poprank','mmr','fusion']:
            if family not in selection:continue
            name=selection[family][rule]
            comparisons[family]={'method':name,'metrics':rows[name],
                'paired_vs_geometric_overall':paired_ci(hits[reference],hits[name]),
                'paired_vs_geometric_tail':paired_ci(hits[reference][tail],hits[name][tail])}
        result[rule]={'geometric_method':reference,'comparisons':comparisons,
                     'geometric_segments_vs_raw':group_analysis(data,data['test_history'][indices],
                                                                targets,hits['raw'],hits[reference])}
    save_json(out,{'audit_version':2,'source':str(folder),'dataset':train['split'],'n_test':len(targets),
                  'validation_selection_sha256':sha256(folder/'validation_selection.json'),
                  'test_top10_sha256':sha256(folder/'test_top10.npz'),
                  'rules':result,'new_test_selection':False,'interval_scope':'paired users, fixed checkpoint',
                  'finished_unix':time.time()})
    print(str(out),flush=True);return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs')
    args = ap.parse_args()
    for complete in sorted(Path(args.root).rglob('complete.json')):
        audit(complete.parent)


if __name__=='__main__':main()
