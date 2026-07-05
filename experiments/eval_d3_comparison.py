"""Head-to-head vs D3 ("Decoding Matters", EMNLP24): does PopContrast (NO external
model) match/beat D3 (TIGER + external SASRec fusion) on tail/coverage? Run from genrec/.

Note: D3's other half — removing length-normalization — is N/A here: TIGER semantic
IDs are FIXED-LENGTH (3 tokens), so amplification bias doesn't arise. The comparable
part is D3's SASRec logit fusion, adapted to item level:
    D3:        rank by  α·z(logp_TIGER) + (1−α)·z(score_SASRec)
    PopContrast: rank by  logp_TIGER − β·z(marginal)         (no external model)
    D3+ours:   fuse then subtract marginal (orthogonality check)

Alignment: SASRec item ids are 1-indexed (id = TIGER_0indexed + 1); we map with +1.
VERIFIED (2026-07-02): our SASRec full-catalog R@10 (~0.017-0.02 Beauty) reproduces via
genrec's OWN dataset+collate+eval — so our scoring is correct and CONSISTENT with the
TIGER exact eval. (genrec's trainer log reports ~0.084 but that number does NOT reproduce
with its own eval loop — anomalous; we trust our consistent full-catalog framework.)
Compares on each split -> results/d3_comparison.json.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from popcontrast.data_utils import build_dataset, compute_popularity, build_sem_id_tables
from popcontrast.model_utils import load_tiger
from popcontrast.oracle import encode_history, score_all_items, _item_tokens_tensor
from genrec.models.sasrec import SASRec

from popcontrast import RESULTS_DIR as RES
DEVICE = "cuda"
SPLITS = os.environ.get("D3_SPLITS", "beauty").split(",")
N_EVAL = int(os.environ.get("D3_N", 3000))
MAX_SEQ = 50


def load_sasrec(split):
    sd = torch.load(f"out/sasrec/amazon/{split}/best_model.pt", map_location=DEVICE)
    n_sas = sd["item_embedding.weight"].shape[0] - 1     # infer num_items from checkpoint
    m = SASRec(num_items=n_sas, max_seq_len=MAX_SEQ, embed_dim=64,
               num_heads=1, num_blocks=2, ffn_dim=256, dropout=0.5).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m, n_sas


def zscore(t):
    return (t - t.mean()) / t.std().clamp_min(1e-6)


@torch.no_grad()
def sasrec_scores(model, hist_0idx, n_sas, I):
    """SASRec scores aligned to TIGER 0-indexed items 0..I-1 (index i -> sasrec logit i+1).
    SASRec ids are 1-indexed (+1). Feed the raw history UNPADDED (length=len, positions
    0..len-1, most-recent at last position) — matches a batch-of-one eval; padding to a
    fixed 50 shifts position embeddings and corrupts scores."""
    ids = [i + 1 for i in hist_0idx if 0 <= i < n_sas][-MAX_SEQ:]
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)   # (1, len)
    logits, _ = model(x)
    last = logits[0, -1, :].clone(); last[0] = float("-inf")
    out = torch.full((I,), float("-inf"), device=DEVICE)
    k = min(I, n_sas)
    out[:k] = last[1:k + 1]                     # TIGER item i <- sasrec logit i+1
    return out


def metrics(tops, targets, seg_head, n_items):
    hit = (tops == targets[:, None])
    hit10 = hit.any(1).float()
    tail = ~seg_head
    seg = lambda v, m: float(v[m].mean()) if m.any() else 0.0
    cnt = torch.bincount(tops[:, :10].reshape(-1), minlength=n_items).float().cpu().numpy()
    nz = cnt[cnt > 0]; p = nz/nz.sum(); ent = float(-(p*np.log(p)).sum()/np.log(n_items))
    return {"R10": float(hit10.mean()), "headR10": seg(hit10, seg_head),
            "tailR10": seg(hit10, tail),
            "cov10": float(torch.unique(tops[:, :10]).numel()/n_items), "ent": ent}


def run_split(split):
    ds_tr = build_dataset(split=split, train_test_split="train")
    ds_te = build_dataset(split=split, train_test_split="test", max_seq_len=50)
    pop = compute_popularity(ds_tr); tables = build_sem_id_tables(ds_tr)
    tiger = load_tiger(f"out/tiger/amazon/{split}/best_model.pt", device=DEVICE)
    item_tok = _item_tokens_tensor(tables, DEVICE); I = item_tok.shape[0]
    sas, n_sas = load_sasrec(split)
    print(f"  [{split}] TIGER items={I}  SASRec items={n_sas}", flush=True)
    marg = torch.load(os.path.join(RES, f"cache_scores_{split}.pt"))["marginal"].to(DEVICE)
    z_marg = zscore(marg)

    rng = np.random.default_rng(0)
    eval_s = [ds_te.samples[i] for i in rng.permutation(len(ds_te.samples))[:N_EVAL]]
    targets = torch.tensor([s["target"] for s in eval_s], device=DEVICE)
    seg_head = torch.tensor([pop.bucket[s["target"]] == "head" for s in eval_s], device=DEVICE)

    methods = {k: torch.empty(len(eval_s), 10, dtype=torch.long, device=DEVICE)
               for k in ["baseline", "ours_b0.75", "d3_a0.7", "d3_a0.5", "d3+ours"]}
    sas_only = torch.empty(len(eval_s), 10, dtype=torch.long, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, s in enumerate(eval_s):
            enc, attn = encode_history(tiger, s["history"], tables, device=DEVICE)
            tg = score_all_items(tiger, enc, attn, item_tok)      # (I,) logp
            ss = sasrec_scores(sas, s["history"], n_sas, I).float()   # (I,)
            ztg, zss = zscore(tg), zscore(ss)
            methods["baseline"][i] = tg.topk(10).indices
            methods["ours_b0.75"][i] = (tg - 0.75 * z_marg).topk(10).indices
            methods["d3_a0.7"][i] = (0.7*ztg + 0.3*zss).topk(10).indices
            methods["d3_a0.5"][i] = (0.5*ztg + 0.5*zss).topk(10).indices
            methods["d3+ours"][i] = (0.7*ztg + 0.3*zss - 0.75*z_marg).topk(10).indices
            sas_only[i] = ss.topk(10).indices
    out = {k: metrics(v, targets, seg_head, I) for k, v in methods.items()}
    out["_sasrec_only"] = metrics(sas_only, targets, seg_head, I)  # SANITY: compare to train log
    out["_meta"] = {"n": len(eval_s), "items": I,
                    "head": int(seg_head.sum()), "tail": int((~seg_head).sum())}
    return out


if __name__ == "__main__":
    allres = {}
    for sp in SPLITS:
        print(f"=== {sp} ===", flush=True)
        allres[sp] = run_split(sp)
        for k, m in allres[sp].items():
            if k.startswith("_meta"): continue
            print(f"  {k:14s} R10={m['R10']:.4f} head={m['headR10']:.4f} "
                  f"tail={m['tailR10']:.4f} cov={m['cov10']:.4f} ent={m['ent']:.3f}", flush=True)
    with open(os.path.join(RES, "d3_comparison.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print("saved results/d3_comparison.json", flush=True)
