"""Select best-vs-final saved LLM adapters under explicit likelihood decoding.

Needed for the initial Qwen3B run because its training validation inherited a
chat repetition penalty (1.05). Training loss/weights are unaffected. Selection
uses only a fixed validation subset, before estimating or opening test results.
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import load_bundle, save_json, seed_all, sha256
from .lcrec import Collate, Examples, init_model, tokenizer_and_codes, trie_lookup, validate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-run",required=True)
    ap.add_argument("--users",type=int,default=1000)
    ap.add_argument("--batch-size",type=int,default=8)
    args=ap.parse_args()
    source=Path(args.training_run)
    if (source/"adapter_selection.json").exists():
        return
    if not (source/"complete.json").exists():
        raise ValueError("Finish training before final adapter selection")
    config=json.loads((source/"config.json").read_text())
    data=load_bundle(config["split"],sid_length=config["sid_length"])
    tokenizer,codes=tokenizer_and_codes(config["base_model"],config["sid_length"])
    indices=np.random.default_rng(7).permutation(len(data["valid_target"]))[:args.users]
    ds=Examples(data,"valid",tokenizer,codes,indices,config.get("prompt_format","legacy"))
    loader=DataLoader(ds,batch_size=args.batch_size,num_workers=2,collate_fn=Collate(tokenizer.pad_token_id,False))
    children,leaves=trie_lookup(ds.item_ids)
    rows={}
    for name in ["best_adapter","latest_adapter"]:
        seed_all(20260927)
        torch.set_num_threads(4)
        folder=source/name
        model=init_model(config["base_model"],tokenizer,codes,str(folder)).eval()
        result=validate(model,loader,children,leaves,SimpleNamespace(sid_length=config["sid_length"]),tokenizer,data)
        rows[name]={"sha256":sha256(folder/"adapter_model.safetensors"),"validation":result}
        print(name,json.dumps(result),flush=True)
        del model
        gc.collect();torch.cuda.empty_cache()
    selected=max(rows,key=lambda key:(rows[key]["validation"]["R10"],rows[key]["validation"]["N10"],key=="best_adapter"))
    destination=source/"evaluation_adapter"
    if destination.exists():
        raise ValueError("Unfinished selection destination exists; inspect before overwrite")
    shutil.copytree(source/selected,destination)
    save_json(source/"adapter_selection.json",{"selected":selected,"candidates":rows,"users":args.users,
              "repetition_penalty":1.,"no_repeat_ngram_size":0,"beam":20,"finished_unix":time.time(),
              "training_validation_note":"Initial Qwen3B training selection used inherited chat repetition_penalty=1.05; this final likelihood-only selection uses no held-out test labels"})


if __name__=="__main__":
    main()
