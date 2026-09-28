"""Independent SASRec / HSTU backbone tests with validation-only checkpointing.

Both use next-item CE on the same training windows as TIGER; this is explicitly
a CE training variant, not a claimed reproduction of original BCE SASRec or
industrial HSTU. Item 0 is padding and all catalog IDs are offset by one.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import ROOT, load_bundle, metrics, paired_ci, save_json, seed_all, sha256, stable_topk
from .evaluate import BETAS, select_validation, standardize


def build(model_name, n_items):
    spec = importlib.util.spec_from_file_location("popcontrast_" + model_name,
                        ROOT / f"genrec/genrec/models/{model_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if model_name == "sasrec":
        return module.SASRec(n_items, embed_dim=128, num_heads=2, num_blocks=2,
                              ffn_dim=512, dropout=.2, loss_type="ce")
    return module.HSTU(n_items, embed_dim=128, num_heads=2, num_blocks=2,
                        dropout=.2, use_temporal_bias=False)


def hidden(model, name, ids):
    if name == "sasrec":
        return model._encode(ids)[:, -1]
    # Same HSTU operations as its original forward, projecting only last state.
    length = ids.shape[1]
    causal = torch.triu(torch.ones(length, length, device=ids.device), diagonal=1).bool()
    padding = ids == 0
    x = model.emb_dropout(model.item_embedding(ids))
    for layer in model.layers:
        x = layer(x, causal, padding, None)
    return model.final_norm(x)[:, -1]


@torch.no_grad()
def iter_scores(model, name, histories, batch_size=64):
    model.eval()
    for start in range(0, len(histories), batch_size):
        ids = torch.as_tensor(histories[start:start + batch_size] + 1, device="cuda", dtype=torch.long)
        yield (hidden(model, name, ids) @ model.item_embedding.weight[1:].T).float().cpu().numpy()


def all_top10(model, name, histories, priors=None):
    methods = {"raw": (0., None)}
    if priors:
        methods.update({f"{key}:{beta}": (beta, prior) for key, prior in priors.items() for beta in BETAS[1:]})
    output = {key: [] for key in methods}
    for scores in iter_scores(model, name, histories):
        for key, (beta, prior) in methods.items():
            for row in scores:
                output[key].append(stable_topk(row if prior is None else row - beta * prior))
    return {key: np.asarray(rows) for key, rows in output.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sasrec", "hstu"], required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    seed_all(args.seed)
    torch.set_num_threads(2)
    data = load_bundle(args.split)
    n_items = len(data["pop_counts"])
    model = build(args.model, n_items).cuda()
    ids = torch.as_tensor(data["train_history"] + 1, dtype=torch.long)
    targets = torch.as_tensor(data["train_target"], dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, targets), batch_size=args.batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, persistent_workers=True,
                         generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_idx = np.random.default_rng(7).permutation(len(data["valid_target"]))[:5000]
    val_tgt = data["valid_target"][val_idx]
    save_json(out / "config.json", {**vars(args), "n_items": n_items, "dim": 128,
               "objective": "last-item full CE" if n_items <= 100000 else "last-item sampled CE (2048 shared uniform negatives)",
               "HSTU_time_bias": False, "validation_n": len(val_idx), "item_indexing": "catalog ID + 1; zero padding excluded from logits",
               "data_sha256": sha256(ROOT / f"results/benchmark/data/{args.split}.npz")})
    best, stale, records = -1, 0, []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.
        for batch, target in loader:
            batch, target = batch.cuda(non_blocking=True), target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            state = hidden(model, args.model, batch)
            if n_items <= 100000:
                logits = state @ model.item_embedding.weight[1:].T
                loss = F.cross_entropy(logits, target)
            else:
                negatives = torch.randint(n_items, (2048,), device="cuda")
                pos = (state * model.item_embedding(target + 1)).sum(-1, keepdim=True)
                neg = state @ model.item_embedding(negatives + 1).T
                neg = neg.masked_fill(target[:, None] == negatives[None, :], float("-inf"))
                loss = F.cross_entropy(torch.cat([pos, neg], 1), torch.zeros_like(target))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach())
        # Score validation in chunks; do not hold a 500k x 5000 matrix on GPU.
        top = all_top10(model, args.model, data["valid_history"][val_idx])["raw"]
        recall = float((top == val_tgt[:, None]).any(1).mean())
        record = {"epoch": epoch, "loss": loss_sum / len(loader), "valid_R10": recall,
                  "seconds": time.perf_counter() - started}
        records.append(record)
        if recall > best:
            best, stale = recall, 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            stale += 1
        save_json(out / "history.json", records)
        print(json.dumps({**record, "best": best, "patience": stale}), flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(out / "best_model.pt", weights_only=True))
    # Expected logits and expected log-softmax differ by an item-independent constant.
    marginal = np.zeros(n_items, dtype=np.float64)
    for scores in iter_scores(model, args.model, data["train_history"][data["legacy_marginal_idx"]]):
        marginal += scores.astype(np.float64).sum(0)
    marginal /= len(data["legacy_marginal_idx"])
    priors = {"geometric": standardize(marginal), "logcount": standardize(np.log1p(data["pop_counts"]))}
    np.save(out / "marginal.npy", marginal)
    val_rows = val_hits = None
    for partition in ["valid", "test"]:
        idx = val_idx if partition == "valid" else data["legacy_test_idx"]
        tgt = data[partition + "_target"][idx]
        top_arrays = all_top10(model, args.model, data[partition + "_history"][idx], priors)
        rows, hits, tops = {}, {}, {"indices": idx, "targets": tgt}
        for name, beta in [("raw", 0)] + [(name, b) for name in priors for b in BETAS[1:]]:
            key = name if name == "raw" else f"{name}:{beta}"
            top = top_arrays[key]
            rows[key], hits[key] = metrics(top, tgt, data["head_mask"], n_items)
            tops[key] = top
            if name != "raw":
                rows[key]["paired_overall"] = paired_ci(hits["raw"], hits[key])
                tail = ~data["head_mask"][tgt]
                rows[key]["paired_tail"] = paired_ci(hits["raw"][tail], hits[key][tail])
        save_json(out / f"{partition}_metrics.json", rows)
        np.savez_compressed(out / f"{partition}_top10.npz", **tops)
        if partition == "valid":
            selection = select_validation(rows, hits, tgt, data["head_mask"])
            save_json(out / "validation_selection.json", selection)
        else:
            save_json(out / "selected_test.json", {name: {rule: {"method": choices[rule], "metrics": rows[choices[rule]]}
                 for rule in ["strict", "budget_5pct", "simultaneous_ci_budget_5pct"]} for name, choices in selection.items()})
    save_json(out / "complete.json", {"best_valid": best, "epochs": len(records), "finished_unix": time.time(),
              "checkpoint_sha256": sha256(out / "best_model.pt"), "marginal_sha256": sha256(out / "marginal.npy")})


if __name__ == "__main__":
    main()
