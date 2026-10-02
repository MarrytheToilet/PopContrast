"""Run a published training/evaluation setting from its recorded configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from .settings import PRIMARY_RUNS

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'results/benchmark/runs'
RECIPES = ROOT / 'configs/benchmark.json'


def read(path):
    return json.loads(path.read_text())


def options(config, names):
    args = []
    for name in names:
        value = config.get(name)
        if value is None or value is False:
            continue
        flag = '--' + name.replace('_', '-')
        args.append(flag)
        if value is True:
            continue
        args.extend(str(v) for v in value) if isinstance(value, list) else args.append(str(value))
    return args


def command(run, stage, output_root, model_root):
    recipe = read(RECIPES)[run]
    evaluation = recipe['evaluation']
    is_llm = run.startswith('lcrec_')
    is_classical = run.startswith(('hstu_', 'sasrec_'))
    training = recipe['training']
    output = output_root/recipe['output_name']
    cmd = [sys.executable, '-m']
    if stage == 'train':
        if run == 'legacy_beauty_full':
            raise ValueError('The diagnostic checkpoint has an unrecorded training seed; use the original TIGER training recipe and evaluate its supplied checkpoint.')
        budget = recipe.get('training_budget_epochs')
        if is_llm:
            cfg = {**training, 'base_model': str(model_root/Path(training['base_model']).name),
                   'output':str(output), 'stop_after_epochs':budget}
            names = ['base_model','split','seed','sid_length','epochs','stop_after_epochs','batch_size','accumulation','eval_n','eval_batch_size','length_grouping','checkpoint_steps','prompt_format','validate_steps','patience','output']
            return cmd+['experiments.benchmark.lcrec']+options(cfg,names)
        cfg = {**training,'output':str(output),'epochs':budget if budget is not None else training['epochs']}
        names = ['model','split','seed','epochs','patience','batch_size','output'] if is_classical else ['split','seed','sid_length','epochs','patience','batch_size','accumulation','eval_batch_size','valid_limit','length_grouping','snapshot_epochs','group_effective_batches','output']
        return cmd+['experiments.benchmark.classical' if is_classical else 'experiments.benchmark.train']+options(cfg,names)
    if is_classical:
        raise ValueError('Conventional scorers evaluate after training in the same command; use --stage train.')
    if is_llm:
        cfg = {**evaluation,'training_run':str(output),'output':str(output/Path(run).name),'precision':'fp32'}
        return cmd+['experiments.benchmark.lcrec_evaluate']+options(cfg,['training_run','output','marginal_m','chunk_size','batch_size','beam','n_valid','n_test','precision'])
    cfg = {**evaluation,'checkpoint':str(ROOT/'checkpoints/legacy/beauty.pt') if run=='legacy_beauty_full' else str(output/'best_model.pt'),
           'output':str(output if run=='legacy_beauty_full' else output/Path(run).name),'precision':'fp32'}
    return cmd+['experiments.benchmark.evaluate']+options(cfg,['split','checkpoint','output','sid_length','beam','eval_batch_size','n_test','n_valid','exact_n','marginal_m','chunk_size','cached_catalog_scoring','bound_betas','precision','legacy_prior'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--run',choices=PRIMARY_RUNS)
    ap.add_argument('--stage',choices=['train','evaluate'],default='evaluate')
    ap.add_argument('--output-root',type=Path,default=ROOT/'runs')
    ap.add_argument('--model-root',type=Path,default=ROOT/'models')
    ap.add_argument('--device',help='CUDA_VISIBLE_DEVICES for this command only')
    ap.add_argument('--print-command',action='store_true')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(PRIMARY_RUNS));return
    if args.run is None:
        ap.error('--run or --list is required')
    if args.output_root.resolve().is_relative_to(ARCHIVE.resolve()):
        ap.error('Use a runtime output directory outside the frozen result archive')
    cmd = command(args.run,args.stage,args.output_root,args.model_root)
    print(shlex.join(cmd),flush=True)
    if not args.print_command:
        env = os.environ.copy()
        if args.device is not None:env['CUDA_VISIBLE_DEVICES'] = args.device
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)


if __name__ == '__main__':
    main()
