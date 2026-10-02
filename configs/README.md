# Benchmark recipes

`benchmark.json` contains the training and evaluation inputs for the 33 primary
settings listed in `experiments/benchmark/settings.py`. The runner reads these
recipes without loading any experimental outputs:

```bash
python -m experiments.benchmark.run --list
python -m experiments.benchmark.run --run beauty_s1_l3_fast/evaluation --stage train --print-command
```

`training_budget_epochs` records the executed budget separately from the
configured scheduler horizon. In particular, Qwen1.5B uses four training epochs
with a six-epoch scheduler horizon. Output and model locations are supplied by
the runner's `--output-root` and `--model-root` options.

Metrics, selected correction strengths, score arrays, and training logs are
experimental outputs and belong in the ignored local result archive. The legacy
Beauty checkpoint has no recorded training seed; its recipe supports evaluation
of a locally supplied checkpoint but does not invent a training configuration.
