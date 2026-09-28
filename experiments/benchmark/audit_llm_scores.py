"""Diagnose beam/cache/full-teacher-forcing differences before test reporting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import load_bundle, save_json, seed_all, sha256, stable_topk
from .lcrec import Collate, Examples, init_model, recommend, tokenizer_and_codes, trie_lookup
from .lcrec_evaluate import cached_item_scores


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-run",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--audit-batch-size",type=int,default=8)
    ap.add_argument("--direct-batch-size",type=int,default=4)
    args=ap.parse_args()
    run=Path(args.training_run)
    config=json.loads((run/"config.json").read_text())
    seed_all(20260927);torch.set_num_threads(4)
    data=load_bundle(config["split"],sid_length=config["sid_length"])
    tokenizer,codes=tokenizer_and_codes(config["base_model"],config["sid_length"])
    selected=np.random.default_rng(7).permutation(len(data["valid_target"]))[:8]
    ds=Examples(data,"valid",tokenizer,codes,selected,config.get("prompt_format","legacy"))
    ids,mask,_=Collate(tokenizer.pad_token_id,False)([ds[i] for i in range(len(ds))])
    ids,mask=ids.cuda(),mask.cuda();children,leaves=trie_lookup(ds.item_ids)
    adapter=run/("evaluation_adapter" if (run/"evaluation_adapter").exists() else "best_adapter")
    model=init_model(config["base_model"],tokenizer,codes,str(adapter)).eval()
    all_tokens=torch.tensor(ds.item_ids,device="cuda")
    audit=[]
    for precision in ["bf16","fp32"]:
        if precision=="fp32":model.float()
        batch=[]
        for start in range(0,len(ids),args.audit_batch_size):
            batch.extend(recommend(model,ids[start:start+args.audit_batch_size],mask[start:start+args.audit_batch_size],
                                   children,leaves,config["sid_length"],20,tokenizer))
        for row,(items,raw) in enumerate(batch):
            prompt=ids[row][mask[row].bool()]
            tokens=all_tokens[items]
            cached=cached_item_scores(model,prompt,tokens,32)
            direct=[]
            for start in range(0,len(tokens),args.direct_batch_size):
                part=tokens[start:start+args.direct_batch_size]
                full=torch.cat([prompt[None].expand(len(part),-1),part],1)
                logp=model(input_ids=full).logits[:,len(prompt)-1:-1].float().log_softmax(-1)
                direct.append(logp.gather(2,part[:,:,None]).squeeze(-1).sum(-1).cpu().numpy())
                del logp,full
            direct=np.concatenate(direct)
            single_items,single_raw=recommend(model,prompt[None],torch.ones_like(prompt[None]),children,leaves,
                       config["sid_length"],20,tokenizer)[0]
            lookup=dict(zip(single_items,single_raw))
            shared=[i for i,item in enumerate(items) if item in lookup]
            beam_batch_error=[abs(float(raw[i])-lookup[items[i]]) for i in shared]
            record={"precision":precision,"user":int(selected[row]),"prompt_tokens":len(prompt),
                "candidate_items":len(items),"beam_cache_max_error":float(np.max(np.abs(raw-cached))),
                "beam_direct_max_error":float(np.max(np.abs(raw-direct))),
                "cache_direct_max_error":float(np.max(np.abs(cached-direct))),
                "audit_batch_size":args.audit_batch_size,"direct_batch_size":args.direct_batch_size,
                "batch_vs_single_common_score_max_error":float(max(beam_batch_error,default=0.)),
                "beam_top10":stable_topk(raw,ids=items).tolist(),
                "raw_score_range":[float(raw.min()),float(raw.max())]}
            audit.append(record);print(json.dumps(record),flush=True)
    save_json(args.output,{"training_run":str(run),"adapter_sha256":sha256(adapter/"adapter_model.safetensors"),
                          "validation_only":True,"rows":audit,"finished_unix":time.time()})


if __name__=="__main__":main()
