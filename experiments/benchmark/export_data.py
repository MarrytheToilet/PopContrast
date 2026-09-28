"""Export the original item mapping, fixed tokenizer, and leave-one-out splits.

Run locally from the repository root with PYTHONPATH=.:genrec. Remote jobs then
need only NumPy and PyTorch, and never regenerate or reorder semantic IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from popcontrast.data_utils import build_dataset, build_sem_id_tables, compute_popularity


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["beauty", "clothing", "sports", "toys"])
    ap.add_argument("--output", default="results/benchmark/data")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for split in args.splits:
        ds = build_dataset(split=split, root=str(root / "genrec/dataset/amazon"),
                           rqvae_path=str(root / "genrec/out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt"),
                           max_seq_len=50)
        tables, pop = build_sem_id_tables(ds), compute_popularity(ds)
        data = {"item_tokens": np.asarray(tables.item_tokens, dtype=np.int32),
                "item_codes": np.asarray(tables.item_codes, dtype=np.int32),
                "pop_counts": pop.counts,
                "head_mask": pop.bucket == "head"}
        for partition in ["train", "valid", "test"]:
            histories, targets, users = [], [], []
            for uid, seq in enumerate(ds.sequences):
                positions = range(1, len(seq) - 2) if partition == "train" else [len(seq) - (2 if partition == "valid" else 1)]
                for pos in positions:
                    histories.append(seq[max(0, pos - 50):pos])
                    targets.append(seq[pos])
                    users.append(uid)
            padded = np.full((len(targets), 50), -1, dtype=np.int32)
            for row, history in enumerate(histories):
                padded[row, -len(history):] = history
            data[partition + "_history"] = padded
            data[partition + "_target"] = np.asarray(targets, dtype=np.int32)
            data[partition + "_user"] = np.asarray(users, dtype=np.int32)
        # The original main experiment's fixed subset, reproduced independently.
        rng = np.random.default_rng(0)
        data["legacy_marginal_idx"] = rng.permutation(len(data["train_target"]))[:512]
        data["legacy_test_idx"] = rng.permutation(len(data["test_target"]))[:5000]
        cache = torch.load(root / f"results/cache_scores_{split}.pt", map_location="cpu", weights_only=False)
        assert np.array_equal(cache["targets"].numpy(), data["test_target"][data["legacy_test_idx"]])
        assert np.array_equal(cache["seg_head"].numpy(), data["head_mask"][cache["targets"].numpy()])
        data["legacy_marginal"] = cache["marginal"].numpy()
        path = output / f"{split}.npz"
        np.savez_compressed(path, **data)
        # Embeddings are separate so training and inference do not load them.
        np.save(output / f"{split}_embeddings.npy", ds.item_embeddings.numpy())
        ckpt = root / f"genrec/out/tiger/amazon/{split}/best_model.pt"
        meta = {"split": split, "n_items": len(tables.item_tokens),
                "n_users": len(ds.sequences), "n_train": len(data["train_target"]),
                "sid_length": tables.sem_id_len, "bundle_sha256": digest(path),
                "checkpoint_sha256": digest(ckpt),
                "tokenizer": "original frozen RQ-VAE; no new tokenizer training",
                "legacy_cache_targets_verified": True,
                "history_items": 50,
                "original_training_seed": "not recorded in checkpoint; original trainer does not explicitly seed RNGs"}
        (output / f"{split}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
