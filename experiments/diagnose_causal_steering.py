"""Causal validation of the 'not representation' leg (Arditi-style symmetric test).
Run from genrec/ root.

If the CAA popularity direction v_pop were the CAUSAL knob for popularity, then at a
decode layer:  +α·v_pop should push outputs MORE popular, −α·v_pop LESS popular.
We measure the mean item-popularity (log count) of the top-10 exactly-scored items
under {baseline, +α·v_pop (add), −α·v_pop (add negative), ablate}. If popularity of the
outputs barely moves, the residual direction is NOT causal for popularity → confirms
the bias is not representational (it's in the decoder marginal).
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger, num_decoder_layers
from popcontrast.hidden_states import collect_hidden, pop_intervention_hooks
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
SPLIT = os.environ.get("CAUSAL_SPLIT", "beauty")
LAYER = int(os.environ.get("CAUSAL_LAYER", 3))
N_EVAL = int(os.environ.get("CAUSAL_N", 1500))


def run():
    ds_tr = build_dataset(split=SPLIT, train_test_split="train")
    ds_te = build_dataset(split=SPLIT, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
    model = load_tiger(f"out/tiger/amazon/{SPLIT}/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]
    logpop = torch.tensor(np.log1p(pop.counts[:I]), dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(0)
    # CAA direction v_pop at LAYER, step 0
    coll = [ds_tr.samples[i] for i in rng.permutation(len(ds_tr.samples))[:15000]]
    hc = collect_hidden(model, coll, tables, pop, [LAYER], batch_size=256, device=DEVICE)
    X, y = hc.H[(LAYER, 0)], hc.y[(LAYER, 0)]
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    v_pop = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32, device=DEVICE)

    eval_s = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:N_EVAL]]

    def mean_top_pop(inject):
        vals = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in eval_s:
                enc, attn = encode_history(model, s["history"], tables, device=DEVICE)
                if inject is None:
                    sc = score_all_items(model, enc, attn, item_tok)
                else:
                    mode, a = inject
                    with pop_intervention_hooks(model, LAYER, v_pop, a, None, mode=mode):
                        sc = score_all_items(model, enc, attn, item_tok)
                top = sc.topk(10).indices
                vals.append(logpop[top].mean().item())
        return float(np.mean(vals))

    conds = {"baseline": None, "add +2·v": ("add", 2.0), "add -2·v": ("add", -2.0),
             "ablate v": ("ablate", 1.0), "steer -2·v": ("steer", 2.0)}
    out = {"split": SPLIT, "layer": LAYER, "n": len(eval_s),
           "mean_toppop": {k: mean_top_pop(v_) for k, v_ in conds.items()}}
    base = out["mean_toppop"]["baseline"]
    out["shift_vs_baseline"] = {k: out["mean_toppop"][k] - base for k in out["mean_toppop"]}
    with open(os.path.join(RES, "causal_steering.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print("\nINTERPRETATION: if |shift| for +v vs -v is tiny relative to head/tail logpop "
          "gap, the residual direction is NOT causal for popularity.", flush=True)


if __name__ == "__main__":
    run()
