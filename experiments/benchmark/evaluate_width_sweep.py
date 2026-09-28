"""Matched B=20/50/100 quality after a completed long-ID/catalog evaluation.

The B20 candidate cache and exact-reference top lists are reused. All widths
use identical test users, checkpoint, priors and frozen validation choices.
Shared-GPU elapsed time is throughput metadata, never online latency.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import load_bundle,load_model,metrics,paired_ci,save_json,seed_all,sha256,stable_topk,tokens_for_histories
from .evaluate import standardize
from .search import make_trie,encode,beam_search_batch


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--users",type=int,default=1000)
    ap.add_argument("--batch-size",type=int,default=8)
    args=ap.parse_args();source,out=Path(args.source),Path(args.output)
    if (out/"complete.json").exists():return
    if not (source/"complete.json").exists():raise ValueError("Complete primary evaluation first")
    cfg=json.loads((source/"config.json").read_text())
    if cfg.get("beam")!=20 or cfg.get("precision","fp32")!="fp32":
        raise ValueError("This sweep requires a FP32 B20 primary evaluation")
    checkpoint=Path(cfg["checkpoint"])
    if sha256(checkpoint)!=cfg["checkpoint_sha256"]:raise ValueError("Frozen checkpoint changed")
    selected=json.loads((source/"validation_selection.json").read_text())["geometric"]["budget_5pct"]
    selected_beta=0. if selected=="raw" else float(selected.split(":")[1])
    betas=sorted({.75,1.5,selected_beta})
    save_json(out/"config.json",{**vars(args),"source_config":cfg,"widths":[20,50,100],"betas":betas,
              "frozen_validation_choice":selected,"gpu_shared":True,
              "priors_sha256":sha256(source/"priors.npz"),"users_unchanged_across_widths":True})
    seed_all(20260927);torch.set_num_threads(4)
    data=load_bundle(cfg["split"],sid_length=cfg["sid_length"])
    with np.load(source/"priors.npz") as f:prior=standardize(f["geometric"])
    with np.load(source/"test_candidates.npz") as f:cache={k:f[k] for k in f.files}
    indices=cache["indices"][:args.users];targets=cache["targets"][:args.users]
    exact={row["sample_index"]:row["comparisons"] for row in json.loads((source/"test_exact_audit.json").read_text())}
    model=load_model(checkpoint,sid_length=cfg["sid_length"]).eval()
    trie=make_trie(data["item_tokens"],prior)
    result={}
    for width in [20,50,100]:
        tops={"raw":[]};tops.update({f"geometric:{b}":[] for b in betas})
        audits=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for start in range(0,len(indices),args.batch_size):
            batch_indices=indices[start:start+args.batch_size]
            if width==20:
                candidates=[]
                for row in range(start,start+len(batch_indices)):
                    a,b=cache["offsets"][row:row+2]
                    candidates.append({"items":cache["items"][a:b],"raw_scores":cache["raw"][a:b]})
            else:
                ids,masks=tokens_for_histories(data["test_history"][batch_indices],data["item_tokens"],"cuda")
                encoded=encode(model,ids,masks)
                candidates=beam_search_batch(model,encoded,masks,trie,prior,width)
            for index,candidate in zip(batch_indices,candidates):
                items,raw=candidate["items"],candidate["raw_scores"]
                tops["raw"].append(stable_topk(raw,ids=items))
                for beta in betas:
                    top=stable_topk(raw-beta*prior[items],ids=items)
                    tops[f"geometric:{beta}"].append(top)
                    reference=exact.get(int(index),{}).get(str(beta))
                    if reference is not None:
                        wanted=np.array(reference["leaf_exact_top"])
                        audits.append({"user_index":int(index),"beta":beta,
                            "candidate_inclusion":float(np.isin(wanted,items).mean()),
                            "top10_overlap":float(np.isin(wanted,top).mean())})
            if (start//200)!=(start+len(batch_indices))//200:
                print("beam",width,start+len(batch_indices),len(indices),flush=True)
        rows={};hits={};tail=~data["head_mask"][targets]
        for name,top in tops.items():
            rows[name],hits[name]=metrics(top,targets,data["head_mask"],len(prior))
            if name!="raw":
                rows[name]["paired_overall"]=paired_ci(hits["raw"],hits[name])
                rows[name]["paired_tail"]=paired_ci(hits["raw"][tail],hits[name][tail])
        row={"metrics":rows,"exact_audit":audits,"elapsed_seconds_shared_gpu":time.perf_counter()-started,
             "b20_cache_reused":width==20,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        result[str(width)]=row;save_json(out/f"width_{width}.json",row)
        np.savez_compressed(out/f"width_{width}_top10.npz",indices=indices,targets=targets,**tops)
        print(json.dumps({"width":width,"raw":rows["raw"],"corrected_selected":rows[f"geometric:{selected_beta}"]}),flush=True)
    save_json(out/"results.json",result);save_json(out/"complete.json",{"finished_unix":time.time()})


if __name__=="__main__":main()
