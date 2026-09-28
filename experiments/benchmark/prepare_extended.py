"""Additional public benchmarks with explicit, reproducible ID construction.

MovieLens uses movie title/genre TF-IDF embeddings. Amazon 2023 uses train-only
collaborative SVD embeddings. Both use 3-stage residual k-means IDs. These are
different tokenizer settings, not reproductions of the original RQ-VAE setup.
"""
import argparse
import gzip
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .common import ROOT, save_json, sha256

SOURCES = {
    "ml1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "games2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
    "arts2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Arts_Crafts_and_Sewing.csv.gz",
    "books2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz",
}


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
            partial.replace(path)
            print("downloaded", path, path.stat().st_size, flush=True)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES))
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    from scipy.sparse import coo_matrix
    import pandas as pd
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    raw_dir, output = ROOT / "results/benchmark/raw", ROOT / "results/benchmark/data"
    output.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        url = SOURCES[name]
        raw = raw_dir / url.rsplit("/", 1)[-1]
        download(url, raw)
        if args.download_only:
            continue
        if (output / f"{name}.json").exists():
            print("already prepared", name, flush=True)
            continue
        if name == "ml1m":
            with zipfile.ZipFile(raw) as archive:
                with archive.open("ml-1m/ratings.dat") as f:
                    frame = pd.read_csv(f, sep="::", engine="python", names=["user", "item", "rating", "time"])
                with archive.open("ml-1m/movies.dat") as f:
                    movies = pd.read_csv(f, sep="::", engine="python", names=["item", "title", "genres"], encoding="latin1")
        else:
            frame = pd.read_csv(raw, usecols=["user_id", "parent_asin", "timestamp"])
            frame.columns = ["user", "item", "time"]
        # No rating threshold: observed interaction next-item prediction.
        before = len(frame)
        frame = frame.sort_values(["user", "time"], kind="stable").drop_duplicates(["user", "item"], keep="first")
        lengths = frame.groupby("user").size()
        frame = frame[frame.user.isin(lengths[lengths >= 5].index)]
        item_names = sorted(frame.item.unique())
        mapping = {item: index for index, item in enumerate(item_names)}
        frame["item_idx"] = frame.item.map(mapping)
        sequences = [group.item_idx.to_numpy(dtype=np.int32) for _, group in frame.groupby("user", sort=True)]
        n_items = len(item_names)
        user_rows, item_cols = [], []
        for uid, seq in enumerate(sequences):
            user_rows.extend([uid] * (len(seq) - 2))
            item_cols.extend(seq[:-2])
        counts = np.bincount(item_cols, minlength=n_items)
        if name == "ml1m":
            texts = movies.set_index("item").reindex(item_names).fillna("")
            x = TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1, 2)).fit_transform(
                (texts.title + " " + texts.genres.str.replace("|", " ", regex=False)).tolist())
            embeddings = TruncatedSVD(n_components=64, random_state=0).fit_transform(x)
            embedding_kind = "title and genre TF-IDF + 64-dimensional SVD; no interaction labels"
        else:
            matrix = coo_matrix((np.ones(len(item_cols), dtype=np.float32), (item_cols, user_rows)),
                                shape=(n_items, len(sequences))).tocsr()
            matrix = normalize(matrix, norm="l2", axis=1)
            embeddings = TruncatedSVD(n_components=64, n_iter=5, random_state=0).fit_transform(matrix)
            embedding_kind = "64-dimensional SVD of train-only item-user incidence; held-out interactions excluded"
        embeddings = normalize(embeddings).astype(np.float32)
        residual = embeddings.copy()
        codes = []
        for level in range(3):
            quantizer = MiniBatchKMeans(n_clusters=256, batch_size=8192, max_iter=100,
                                        n_init=3, random_state=level, reassignment_ratio=.01)
            labels = quantizer.fit_predict(residual)
            codes.append(labels)
            residual -= quantizer.cluster_centers_[labels]
        item_codes = np.stack(codes, axis=1).astype(np.int32)
        data = {"item_codes": item_codes, "item_tokens": item_codes + np.arange(3) * 256 + 1,
                "pop_counts": counts, "head_mask": np.zeros(n_items, dtype=bool)}
        data["head_mask"][np.argsort(-counts, kind="stable")[:round(.2 * n_items)]] = True
        for partition in ["train", "valid", "test"]:
            n = sum(len(seq) - 3 for seq in sequences) if partition == "train" else len(sequences)
            hist = np.full((n, 50), -1, dtype=np.int32)
            targets = np.empty(n, dtype=np.int32)
            users = np.empty(n, dtype=np.int32)
            row = 0
            for uid, seq in enumerate(sequences):
                positions = range(1, len(seq) - 2) if partition == "train" else [len(seq) - (2 if partition == "valid" else 1)]
                for pos in positions:
                    h = seq[max(0, pos - 50):pos]
                    hist[row, -len(h):] = h
                    targets[row], users[row] = seq[pos], uid
                    row += 1
            data[partition + "_history"], data[partition + "_target"], data[partition + "_user"] = hist, targets, users
        rng = np.random.default_rng(0)
        data["legacy_marginal_idx"] = rng.permutation(len(data["train_target"]))[:512]
        data["legacy_test_idx"] = rng.permutation(len(data["test_target"]))[:5000]
        np.savez_compressed(output / f"{name}.npz", **data)
        np.save(output / f"{name}_embeddings.npy", embeddings)
        save_json(output / f"{name}.json", {"split": name, "source_url": url, "source_sha256": sha256(raw),
                  "n_items": n_items, "n_users": len(sequences), "n_interactions": len(frame),
                  "n_rows_before_filtering": before, "n_train": len(data["train_target"]),
                  "n_zero_train_items": int((counts == 0).sum()), "sid_length": 3,
                  "tokenizer": "residual MiniBatchKMeans, 3 x 256; collisions retained; distinct from original RQ-VAE",
                  "embedding_kind": embedding_kind, "history_items": 50,
                  "split_protocol": "chronological per-user leave-last-two-out; all observed ratings; stable time ties",
                  "bundle_sha256": sha256(output / f"{name}.npz")})
        print("prepared", name, n_items, len(sequences), flush=True)


if __name__ == "__main__":
    main()
