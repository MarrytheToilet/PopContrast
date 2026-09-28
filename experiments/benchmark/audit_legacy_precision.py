"""Re-execute historical beam code while isolating encoder/log-softmax precision.

Exports candidates for a local join against the original exact-score cache.
Only the type-only SemIdTables import is replaced to avoid optional packages.
The second module changes exactly the historical log-softmax cast.
"""
import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, load_bundle, load_model, save_json, seed_all, sha256, tokens_for_histories
from .search import encode, make_trie


def decoder(float_logsoftmax=False):
    path = ROOT / "popcontrast/decoding.py"
    source = path.read_text().replace("from popcontrast.data_utils import SemIdTables", "SemIdTables = object")
    if float_logsoftmax:
        old = "logp = F.log_softmax(logits, dim=-1)"
        if source.count(old) != 1:
            raise ValueError("Historical source changed")
        source = source.replace(old, "logp = F.log_softmax(logits.float(), dim=-1)")
    name = "legacy_precision_audit_" + str(float_logsoftmax)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.trie_beam_search


@torch.no_grad()
def main():
    seed_all(20260927)
    torch.set_num_threads(2)
    data = load_bundle("beauty")
    model = load_model("checkpoints/legacy/beauty.pt").eval()
    wrapper = types.SimpleNamespace(model=model)
    trie = make_trie(data["item_tokens"], np.zeros(len(data["pop_counts"])))
    tables = types.SimpleNamespace(sem_id_len=3, tokens_to_items=trie.leaves,
                                  allowed_next=lambda p: list(trie.children.get(p, ())))
    original, cast = decoder(False), decoder(True)
    out = ROOT / "runs/legacy_precision_audit"
    out.mkdir(parents=True, exist_ok=True)
    indices = data["legacy_test_idx"][:1000]
    modes = {"historical": (False, True, original), "float_logsoftmax": (False, True, cast),
             "aligned_bf16_encoder": (True, True, cast), "all_fp32": (False, False, cast)}
    save_json(out / "config.json", {"n": len(indices), "beams": [20, 100],
              "users": "original cache order, independently target-verified during export",
              "original_decoder_sha256": sha256(ROOT / "popcontrast/decoding.py"),
              "checkpoint_sha256": sha256("checkpoints/legacy/beauty.pt"),
              "modes": {key: {"encoder_bf16": x[0], "decoder_bf16": x[1], "float_logsoftmax": key != "historical"} for key, x in modes.items()},
              "hardware_library_difference": "historical source on current RTX5090/torch2.12.1; exact bitwise old-run reproduction not assumed"})
    for mode, (encoder_bf16, decoder_bf16, function) in modes.items():
        path = out / (mode + ".npz")
        if path.exists():
            continue
        cache = {width: {"items": [], "raw": [], "offsets": [0]} for width in [20, 100]}
        for row, idx in enumerate(indices):
            ids, mask = tokens_for_histories(data["test_history"][[idx]], data["item_tokens"], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16) if encoder_bf16 else contextlib.nullcontext():
                enc = encode(model, ids, mask)
            for width in cache:
                with torch.autocast("cuda", dtype=torch.bfloat16) if decoder_bf16 else contextlib.nullcontext():
                    items, scores, _ = function(wrapper, enc, mask, tables, num_beams=width, device="cuda")
                cache[width]["items"].extend(items)
                cache[width]["raw"].extend(scores)
                cache[width]["offsets"].append(len(cache[width]["items"]))
            if (row + 1) % 200 == 0:
                print(mode, row + 1, flush=True)
        np.savez_compressed(path, indices=indices, targets=data["test_target"][indices],
                 **{f"B{width}_{key}": np.asarray(value) for width, records in cache.items() for key, value in records.items()})
    save_json(out / "complete.json", {"n": len(indices), "note": "join exported candidates against historical full-score cache locally"})


if __name__ == "__main__":
    main()
