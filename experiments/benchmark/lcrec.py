"""Recover the Qwen2.5-1.5B + LoRA SID recommendation route.

Controlled seqrec-only LC-Rec-style backbone experiment: the fixed TIGER IDs
isolate architecture from tokenizer changes. This is not a reproduction of
LC-Rec's full multi-task training recipe. Trainable token rows are explicitly
saved by PEFT; target tokens are never truncated; generation follows the item
trie, not just independent position masks.
"""
import argparse
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .common import load_bundle, metrics, save_json, seed_all, stable_topk, sha256


class Examples(Dataset):
    def __init__(self, data, partition, tokenizer, code_ids, indices=None, prompt_format="legacy"):
        self.history = data[partition + "_history"]
        self.target = data[partition + "_target"]
        self.indices = np.arange(len(self.target)) if indices is None else indices
        self.item_ids = code_ids[np.asarray(data["item_tokens"])]
        self.eos = tokenizer.eos_token_id
        self.prefix = tokenizer.encode("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nUser interaction history: ", add_special_tokens=False)
        self.suffix = tokenizer.encode("\nPredict the next item:\n\n### Response:", add_special_tokens=False)
        self.separator = tokenizer.encode(", ", add_special_tokens=False)
        if prompt_format == "compact":
            self.prefix = tokenizer.encode("History:", add_special_tokens=False)
            self.suffix = tokenizer.encode("\nNext:", add_special_tokens=False)
            self.separator = []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        prompt = list(self.prefix)
        history = self.history[index]
        for number, item in enumerate(history[history >= 0]):
            if number:
                prompt.extend(self.separator)
            prompt.extend(self.item_ids[item].tolist())
        prompt.extend(self.suffix)
        target = int(self.target[index])
        response = self.item_ids[target].tolist() + [self.eos]
        return prompt, response, target


class Collate:
    def __init__(self, pad_id, training):
        self.pad_id, self.training = pad_id, training

    def __call__(self, examples):
        sequences = [p + r if self.training else p for p, r, _ in examples]
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, -len(sequence):] = torch.tensor(sequence)
            mask[row, -len(sequence):] = 1
        return ids, mask, torch.tensor([t for _, _, t in examples])


class LengthBatches(Sampler):
    """Shuffle examples, then group similar lengths inside random megabatches."""
    def __init__(self, dataset, batch_size, seed):
        self.lengths = (dataset.history[dataset.indices] >= 0).sum(1)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(order), 50 * self.batch_size):
            chunk = order[start:start + 50 * self.batch_size]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            batches.extend(chunk[i:i+self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size))
        ordering = rng.permutation(len(batches))
        # Exercise the largest batch first, so insufficient memory fails early.
        longest = max(range(len(batches)), key=lambda i: self.lengths[batches[i][0]])
        where = int(np.flatnonzero(ordering == longest)[0])
        ordering[0], ordering[where] = ordering[where], ordering[0]
        for index in ordering:
            yield batches[index]


def tokenizer_and_codes(path, length):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    words = [f"<C{pos}_{code}>" for pos in range(length) for code in range(256)]
    tokenizer.add_special_tokens({"additional_special_tokens": words})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    code_ids = np.array([tokenizer.pad_token_id] + tokenizer.convert_tokens_to_ids(words))
    if len(set(code_ids[1:].tolist())) != 256 * length:
        raise RuntimeError("SID token IDs must be unique")
    for word, index in zip(words, code_ids[1:]):
        if tokenizer.encode(word, add_special_tokens=False) != [int(index)]:
            raise RuntimeError("SID token is not atomic")
    return tokenizer, code_ids


def init_model(base_path, tokenizer, code_ids, adapter=None, trainable=False):
    from transformers import AutoModelForCausalLM, GenerationConfig
    from peft import LoraConfig, PeftModel, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True)
    # Instruction-chat presets can contain repetition penalties, which would
    # silently change SID log-scores relative to teacher-forced marginals.
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.resize_token_embeddings(len(tokenizer))
    if not model.config.tie_word_embeddings:
        raise ValueError("This controlled implementation requires tied input/output embeddings for trainable SID rows")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    else:
        cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         trainable_token_indices={"embed_tokens": code_ids[1:].tolist()},
                         ensure_weight_tying=True)
        model = get_peft_model(model, cfg)
    if not adapter or trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.cuda()


def trie_lookup(item_ids):
    children, leaves = {}, {}
    for item, row in enumerate(item_ids):
        row = tuple(int(t) for t in row)
        leaves.setdefault(row, []).append(item)
        for depth in range(len(row)):
            children.setdefault(row[:depth], set()).add(row[depth])
    return {key: sorted(value) for key, value in children.items()}, leaves


def merge_sid_rows_for_inference(model):
    """Materialize saved SID token rows once; leave every LoRA matrix unmerged.

    PEFT's unmerged TrainableTokensLayer builds a complete embedding matrix
    with index_copy on each forward. Its merge is an exact row replacement,
    unlike adding low-rank weights in BF16. Use only on a frozen eval model.
    """
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
    if model.training:
        raise ValueError("SID row materialization is restricted to eval models")
    layers = [module for module in model.modules() if isinstance(module, TrainableTokensLayer)]
    for layer in layers:
        layer.merge(safe_merge=True)
    return layers


@torch.no_grad()
def recommend(model, ids, mask, children, leaves, sid_length, beams, tokenizer, prefill_once=False, vectorized_trie=True):
    prompt_length = ids.shape[1]
    def allowed(batch_id, sequence):
        prefix = tuple(sequence[prompt_length:].tolist())
        return children.get(prefix, [tokenizer.eos_token_id])
    model.eval()
    cache_args={}
    if vectorized_trie:
        from transformers import LogitsProcessor,LogitsProcessorList
        class TrieMask(LogitsProcessor):
            def __call__(self,input_ids,scores):
                # One device-to-host transfer and one scatter, replacing a
                # synchronization and mask write for every individual beam.
                prefixes=input_ids[:,prompt_length:].detach().cpu().tolist()
                vocab=scores.shape[1]
                flat=[row*vocab+token for row,prefix in enumerate(prefixes)
                      for token in children.get(tuple(prefix),[tokenizer.eos_token_id])]
                legal=torch.zeros_like(scores,dtype=torch.bool)
                legal.view(-1)[torch.tensor(flat,device=scores.device,dtype=torch.long)]=True
                return scores.masked_fill(~legal,float("-inf"))
        cache_args["logits_processor"]=LogitsProcessorList([TrieMask()])
    else:
        cache_args["prefix_allowed_tokens_fn"]=allowed
    if prefill_once and prompt_length>1:
        positions=mask[:,:-1].long().cumsum(-1)-1
        positions.masked_fill_(mask[:,:-1]==0,1)
        first=model(input_ids=ids[:,:-1],attention_mask=mask[:,:-1],position_ids=positions,
                    use_cache=True,logits_to_keep=1)
        cache=first.past_key_values
        cache.batch_repeat_interleave(beams)
        cache_args["past_key_values"]=cache
    generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=sid_length,
                 num_beams=beams, num_return_sequences=beams, do_sample=False,
                 repetition_penalty=1., no_repeat_ngram_size=0,
                 renormalize_logits=False,
                 length_penalty=0., pad_token_id=tokenizer.pad_token_id,
                 eos_token_id=tokenizer.eos_token_id, use_cache=True,
                 return_dict_in_generate=True, output_scores=True, **cache_args)
    sequences = generated.sequences[:, prompt_length:].reshape(len(ids), beams, sid_length).cpu().numpy()
    scores = generated.sequences_scores.reshape(len(ids), beams).float().cpu().numpy()
    result = []
    for seqs, values in zip(sequences, scores):
        item_scores = {}
        for seq, value in zip(seqs, values):
            for item in leaves[tuple(seq)]:
                item_scores[item] = float(value)
        item = np.asarray(list(item_scores))
        raw = np.asarray([item_scores[i] for i in item])
        result.append((item, raw))
    return result


@torch.no_grad()
def validate(model, loader, children, leaves, args, tokenizer, data):
    tops, targets = [], []
    for ids, mask, target in loader:
        candidates = recommend(model, ids.cuda(), mask.cuda(), children, leaves,
                               args.sid_length, 20, tokenizer)
        tops.extend(stable_topk(raw, ids=item) for item, raw in candidates)
        targets.extend(target.tolist())
    return metrics(np.asarray(tops), np.asarray(targets), data["head_mask"], len(data["pop_counts"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sid-length", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--stop-after-epochs", type=int,
                    help="Complete this many epochs while retaining the original LR schedule horizon")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulation", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--length-grouping", action="store_true")
    ap.add_argument("--checkpoint-steps", type=int, default=250)
    ap.add_argument("--prompt-format", choices=["compact", "legacy"], default="compact")
    ap.add_argument("--validate-steps", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    seed_all(args.seed)
    torch.set_num_threads(4)
    data = load_bundle(args.split, sid_length=args.sid_length)
    tokenizer, code_ids = tokenizer_and_codes(args.base_model, args.sid_length)
    tokenizer.save_pretrained(out / "tokenizer")
    resume_state = None
    if args.resume:
        old = json.loads((out / "config.json").read_text())
        for key in ["base_model", "split", "seed", "sid_length", "batch_size", "accumulation", "length_grouping", "prompt_format", "epochs"]:
            if old[key] != getattr(args, key):
                raise ValueError(f"Resume must preserve training semantics: {key}")
        if not args.length_grouping:
            raise ValueError("Exact sampler resume requires length grouping")
        resume_state = torch.load(out / "latest_optimizer.pt", map_location="cpu", weights_only=False)
        if resume_state.get("adapter_sha256") and sha256(out / "latest_adapter/adapter_model.safetensors") != resume_state["adapter_sha256"]:
            raise RuntimeError("Adapter and optimizer snapshots disagree")
        events_path = out / "resume_events.json"
        events = json.loads(events_path.read_text()) if events_path.exists() else []
        events.append({"unix": time.time(), "from_step": resume_state["steps"],
                       "previous_settings": {k: v for k, v in old.items() if k in vars(args)}, "new_settings": vars(args)})
        save_json(events_path, events)
    model = init_model(args.base_model, tokenizer, code_ids,
                       str(out / "latest_adapter") if args.resume else None, trainable=args.resume)
    train_ds = Examples(data, "train", tokenizer, code_ids, prompt_format=args.prompt_format)
    valid_indices = np.random.default_rng(7).permutation(len(data["valid_target"]))[:args.eval_n]
    valid_ds = Examples(data, "valid", tokenizer, code_ids, valid_indices, args.prompt_format)
    sampler = LengthBatches(train_ds, args.batch_size, args.seed) if args.length_grouping else None
    batching = {"batch_sampler": sampler} if sampler else {"batch_size": args.batch_size, "shuffle": True}
    train_loader = DataLoader(train_ds, **batching, num_workers=2,
                     collate_fn=Collate(tokenizer.pad_token_id, True),
                     generator=torch.Generator().manual_seed(args.seed), pin_memory=True,
                     persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, num_workers=2,
                     collate_fn=Collate(tokenizer.pad_token_id, False), pin_memory=True)
    children, leaves = trie_lookup(train_ds.item_ids)
    token_params, lora_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (token_params if "trainable_tokens" in name else lora_params).append(param)
    if not token_params or not lora_params:
        raise RuntimeError("Both token rows and LoRA parameters must be trainable")
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": 2e-4},
                                  {"params": token_params, "lr": 1e-3}], weight_decay=.01)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.accumulation)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                 lambda step: min(1., (step + 1) / 100) * .5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    config = {**vars(args), "training_variant": "LC-Rec-style seqrec-only; dataset's frozen IDs (see data metadata)",
              "n_train": len(train_ds), "n_valid": len(valid_ds), "num_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "new_token_rows_trainable": True, "checkpointing": "non-reentrant + enable_input_require_grads",
              "token_truncation": "none; at most 50 history items; target+EOS retained",
              "validation_generation": {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                                        "length_penalty": 0.0, "renormalize_logits": False},
              "valid_seed": 7, "beam_width": 20, "sid_codes": code_ids.tolist()}
    save_json(out / "config.json", config)
    print(json.dumps({k: v for k, v in config.items() if k != "sid_codes"}), flush=True)
    best, steps, history, start_epoch, skip_batches, stale = -1, 0, [], 0, 0, 0
    validation_events = []
    if args.resume:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["torch_rng"])
        torch.cuda.set_rng_state(resume_state["cuda_rng"])
        start_epoch, skip_batches, steps = resume_state["epoch"], resume_state["batch"], resume_state["steps"]
        best = json.loads((out / "best.json").read_text())["valid"]["R10"] if (out / "best.json").exists() else -1
        history = json.loads((out / "history.json").read_text()) if (out / "history.json").exists() else []
        validation_events = json.loads((out / "validation_events.json").read_text()) if (out / "validation_events.json").exists() else []
        stale = resume_state.get("stale", 0)
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def checkpoint(epoch, batch):
        model.save_pretrained(out / "latest_adapter", save_embedding_layers=False)
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "batch": batch, "steps": steps, "stale": stale,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "adapter_sha256": sha256(out / "latest_adapter/adapter_model.safetensors")}
        torch.save(state, out / "latest_optimizer.pt.tmp")
        (out / "latest_optimizer.pt.tmp").replace(out / "latest_optimizer.pt")
    def validate_and_record(epoch):
        nonlocal best, stale
        val = validate(model, valid_loader, children, leaves, args, tokenizer, data)
        record = {"epoch": epoch, "step": steps, "valid": val}
        if val["R10"] > best:
            best, stale = val["R10"], 0
            model.save_pretrained(out / "best_adapter", save_embedding_layers=False)
            save_json(out / "best.json", record)
        else:
            stale += 1
        record["stale_validation_checks"] = stale
        validation_events.append(record)
        save_json(out / "validation_events.json", validation_events)
        print(json.dumps(record), flush=True)
        model.train()
        return val
    def finish(reason):
        save_json(out / "complete.json", {"best_valid_R10": best, "steps": steps, "finished_unix": time.time(),
                  "completion_reason": reason, "next_step": "freeze best adapter; estimate marginal; matched corrected beam evaluation"})
    epoch_limit = min(args.epochs, args.stop_after_epochs or args.epochs)
    for epoch in range(start_epoch, epoch_limit):
        if sampler:
            sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen_batches = 0., 0
        started = time.perf_counter()
        for batch_id, (ids, mask, _) in enumerate(train_loader):
            if epoch == start_epoch and batch_id < skip_batches:
                continue
            ids, mask = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True)
            # Only project the response positions through the large language head.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(input_ids=ids, attention_mask=mask, use_cache=False,
                                   logits_to_keep=args.sid_length + 2)
                logits = prediction.logits[:, :-1].float()
                target = ids[:, -(args.sid_length + 1):]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            if not loss.requires_grad or not torch.isfinite(loss):
                raise RuntimeError("Missing gradients or nonfinite LC-Rec loss")
            (loss / args.accumulation).backward()
            loss_sum += float(loss.detach())
            seen_batches += 1
            if batch_id == 0 and epoch == 0:
                checks = {"lora_grad_norm": sum(float(p.grad.float().norm()) for p in lora_params if p.grad is not None),
                          "token_grad_norm": sum(float(p.grad.float().norm()) for p in token_params if p.grad is not None),
                          "target_tokens_per_example": int(target.shape[1]), "first_loss": float(loss.detach())}
                save_json(out / "gradient_check.json", checks)
                if min(checks["lora_grad_norm"], checks["token_grad_norm"]) <= 0:
                    raise RuntimeError("LC-Rec gradient health check failed")
                print("gradient_check", json.dumps(checks), flush=True)
            if (batch_id + 1) % args.accumulation == 0 or batch_id + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps % 20 == 0:
                    print(json.dumps({"epoch": epoch, "step": steps, "mean_loss": loss_sum / seen_batches,
                          "epoch_elapsed_seconds": time.perf_counter() - started,
                          "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
                    save_json(out / "progress.json", {"epoch": epoch, "step": steps,
                              "batch": batch_id + 1, "batches_per_epoch": len(train_loader),
                              "loss": loss_sum / seen_batches, "elapsed_seconds": time.perf_counter() - started})
                if args.checkpoint_steps and steps % args.checkpoint_steps == 0:
                    checkpoint(epoch, batch_id + 1)
                if args.validate_steps and steps % args.validate_steps == 0:
                    validate_and_record(epoch)
                    if args.patience and stale >= args.patience:
                        finish("validation_early_stopping")
                        return
                if stop["requested"]:
                    checkpoint(epoch, batch_id + 1)
                    save_json(out / "interrupted.json", {"epoch": epoch, "batch": batch_id + 1, "steps": steps,
                              "note": "Optimizer and RNG saved; use --resume. Training is not marked complete."})
                    return
                if args.smoke_steps and steps >= args.smoke_steps:
                    model.save_pretrained(out / "smoke_adapter", save_embedding_layers=False)
                    save_json(out / "smoke_complete.json", {"steps": steps, "last_loss": float(loss.detach()),
                              "elapsed_seconds": time.perf_counter() - started,
                              "microbatches": batch_id + 1, "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20})
                    return
        val = validate_and_record(epoch)
        row = {"epoch": epoch, "loss": loss_sum / max(seen_batches, 1), "valid": val,
               "loss_scope": "resumed segment" if epoch == start_epoch and skip_batches else "full epoch",
               "seconds": time.perf_counter() - started}
        history.append(row)
        save_json(out / "history.json", history)
        print(json.dumps(row), flush=True)
        if args.patience and stale >= args.patience:
            finish("validation_early_stopping")
            return
    finish("epoch_budget")


if __name__ == "__main__":
    main()
