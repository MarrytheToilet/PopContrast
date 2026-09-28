"""Fresh seeded TIGER training, fixed tokenizer, validation-only checkpointing."""
import argparse
import json
import time
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler

from .common import ROOT, load_bundle, load_model, save_json, seed_all, sha256, tokens_for_histories


class LengthSampler(Sampler):
    def __init__(self, lengths, batch_size, seed, effective_batch_size=None):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0
        self.group_size=effective_batch_size or batch_size
        if self.group_size%batch_size:
            raise ValueError("Effective grouping must be divisible by microbatch size")

    def __len__(self):
        return (len(self.lengths)//self.group_size)*(self.group_size//self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(order), 50 * self.group_size):
            chunk = order[start:start + 50 * self.group_size]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            batches.extend(chunk[i:i+self.group_size].tolist() for i in range(0, len(chunk), self.group_size)
                           if len(chunk[i:i+self.group_size]) == self.group_size)
        for index in rng.permutation(len(batches)):
            for start in range(0,self.group_size,self.batch_size):
                yield batches[index][start:start+self.batch_size]


def trim_collate(examples):
    ids, mask, labels = [torch.stack(values) for values in zip(*examples)]
    length = max(1, int(mask.sum(1).max()))
    return ids[:, -length:], mask[:, -length:], labels


@torch.no_grad()
def validate(model, loader, sid_length, beams):
    model.eval()
    total, hits = 0, 0
    for ids, mask, labels in loader:
        ids, mask = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.generate(input_ids=ids, attention_mask=mask,
                                  max_length=sid_length + 1, num_beams=beams,
                                  num_return_sequences=beams)
        pred = pred[:, 1:].reshape(len(ids), beams, sid_length).cpu()
        matches = (pred[:, :10] == labels[:, None]).all(-1).any(-1)
        hits += int(matches.sum())
        total += len(ids)
    return {"R10_sid": hits / total, "hits": hits, "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sid-length", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accumulation", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=96)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--length-grouping", action="store_true")
    ap.add_argument("--snapshot-epochs", nargs="*", type=int, default=[])
    ap.add_argument("--group-effective-batches", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        print("Already complete:", output, flush=True)
        return
    seed_all(args.seed)
    torch.set_num_threads(4)
    data = load_bundle(args.split, sid_length=args.sid_length)
    loaders, training_sampler = {}, None
    for partition, bs in [("train", args.batch_size), ("valid", args.eval_batch_size)]:
        hist = data[partition + "_history"]
        target = data[partition + "_target"]
        if partition == "valid" and args.valid_limit:
            idx = np.random.default_rng(7).permutation(len(target))[:args.valid_limit]
            hist, target = hist[idx], target[idx]
        ids, mask = tokens_for_histories(hist, data["item_tokens"])
        labels = torch.as_tensor(data["item_tokens"][target], dtype=torch.long)
        if args.length_grouping:
            if partition == "valid":
                ordering = torch.argsort(mask.sum(1), stable=True)
                ids, mask, labels = ids[ordering], mask[ordering], labels[ordering]
            else:
                training_sampler = LengthSampler(mask.sum(1).numpy(), bs, args.seed,
                                     bs*args.accumulation if args.group_effective_batches else None)
        batching = ({"batch_sampler": training_sampler} if args.length_grouping and partition == "train"
                    else {"batch_size": bs, "shuffle": partition == "train", "drop_last": partition == "train"})
        loaders[partition] = DataLoader(TensorDataset(ids, mask, labels), **batching,
                collate_fn=trim_collate if args.length_grouping else None,
                num_workers=2, pin_memory=True, persistent_workers=True,
                generator=torch.Generator().manual_seed(args.seed))
    model = load_model(sid_length=args.sid_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    dataset_meta = json.loads((ROOT / f"results/benchmark/data/{args.split}.json").read_text())
    meta = {**vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "tokenizer": dataset_meta.get("tokenizer", "original fixed RQ-VAE") +
                         ({3: "", 4: " + one collision disambiguation token", 5: " + two base-256 disambiguation tokens",
                           6: " + lossless six base-16 digits for the original three codes"}[args.sid_length]),
            "checkpoint_selection": "validation SID Recall@10, native beam=30; no test evaluation during training",
            "training_precision": "float32", "validation_precision": "bfloat16 autocast, original generate",
            "data_sha256": sha256(ROOT / f"results/benchmark/data/{args.split}.npz"),
            "t5_source_sha256": sha256(ROOT / "genrec/genrec/modules/t5.py"),
            "started_unix": time.time(), "n_train": len(data["train_target"]),
            "n_valid": len(loaders["valid"].dataset)}
    save_json(output / "config.json", meta)
    print(json.dumps(meta), flush=True)
    best, stale, history = -1, 0, []
    for epoch in range(args.epochs):
        if training_sampler:
            training_sampler.epoch = epoch
        started = time.perf_counter()
        model.train()
        loss_sum, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_id, (ids, mask, labels) in enumerate(loaders["train"]):
            ids, mask, labels = [x.cuda(non_blocking=True) for x in (ids, mask, labels)]
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            group_start = (batch_id // args.accumulation) * args.accumulation
            divisor = min(args.accumulation, len(loaders["train"]) - group_start)
            (out.loss / divisor).backward()
            if (batch_id + 1) % args.accumulation == 0 or batch_id + 1 == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(out.loss.detach())
            n_batches += 1
            if n_batches % 1000 == 0:
                progress = {"epoch": epoch, "batch": n_batches, "batches_per_epoch": len(loaders["train"]),
                            "optimizer_steps_this_epoch": n_batches // args.accumulation,
                            "loss": loss_sum/n_batches, "elapsed_seconds": time.perf_counter()-started}
                save_json(output/"progress.json", progress)
                print(json.dumps({"training_progress": progress}), flush=True)
        val = validate(model, loaders["valid"], args.sid_length, 30)
        row = {"epoch": epoch, "loss": loss_sum / n_batches, "valid": val,
               "seconds": time.perf_counter() - started}
        history.append(row)
        if val["R10_sid"] > best:
            best, stale = val["R10_sid"], 0
            torch.save(model.state_dict(), output / "best_model.pt")
            save_json(output / "best.json", row)
        else:
            stale += 1
        save_json(output / "history.json", history)
        print(json.dumps({**row, "best": best, "patience": stale}), flush=True)
        if epoch+1 in args.snapshot_epochs:
            snapshot=output/"snapshots"/f"epoch{epoch+1}"
            snapshot.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(output/"best_model.pt",snapshot/"best_model.pt")
            shutil.copyfile(output/"best.json",snapshot/"best.json")
            save_json(snapshot/"config.json",{**meta,"epochs":epoch+1,"output":str(snapshot),
                      "snapshot_source":str(output),"snapshot_rule":"best validation checkpoint among first N complete epochs; constant learning rate and epoch sampler unaffected by later epoch cap"})
            save_json(snapshot/"history.json",history)
            save_json(snapshot/"complete.json",{"epochs":epoch+1,"best_valid":best,
                      "checkpoint_sha256":sha256(snapshot/"best_model.pt"),"finished_unix":time.time(),
                      "completion_reason":"prespecified training-exposure snapshot","converged":False})
        if stale >= args.patience:
            break
    save_json(output / "complete.json", {"epochs": len(history), "best_valid": best,
              "checkpoint_sha256": sha256(output / "best_model.pt"), "finished_unix": time.time()})


if __name__ == "__main__":
    main()
