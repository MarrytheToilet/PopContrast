"""CPU tests of LLM cache scoring and saved trainable SID rows."""
import tempfile
import unittest
import copy

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

from .common import stable_topk
from .lcrec_evaluate import cached_item_scores


def small_model():
    torch.manual_seed(9)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            tie_word_embeddings=True, attention_dropout=0.)).eval()


class LLMTests(unittest.TestCase):
    def test_materialized_sid_rows_preserve_logits_and_scores(self):
        from .lcrec import merge_sid_rows_for_inference
        for dtype in [torch.float32, torch.bfloat16]:
            model = get_peft_model(small_model().to(dtype), LoraConfig(task_type="CAUSAL_LM", r=2,
                    target_modules=["q_proj", "v_proj"],
                    trainable_token_indices={"embed_tokens": [31, 32, 33]}, ensure_weight_tying=True))
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "trainable_tokens_delta" in name or "lora_B" in name:
                        param.add_(torch.randn_like(param) * .02)
            model.eval()
            prompt = torch.tensor([1, 4, 6, 9])
            items = torch.tensor([[31, 32, 33], [32, 34, 35], [31, 32, 36]])
            before = model(input_ids=prompt[None]).logits.detach().clone()
            score_before = cached_item_scores(model, prompt, items, 2)
            layers = merge_sid_rows_for_inference(model)
            self.assertGreaterEqual(len(layers), 2)
            torch.testing.assert_close(before, model(input_ids=prompt[None]).logits, rtol=0., atol=0.)
            np.testing.assert_array_equal(score_before, cached_item_scores(model, prompt, items, 2))
            for name, module in model.named_modules():
                if hasattr(module, "lora_A"):
                    self.assertFalse(module.merged, name)
            for layer in layers:
                layer.unmerge()
            torch.testing.assert_close(before, model(input_ids=prompt[None]).logits, rtol=0., atol=0.)

    def test_generation_scores_ignore_chat_repetition_presets(self):
        from types import SimpleNamespace
        from .lcrec import recommend, trie_lookup
        model = small_model().eval()
        model.generation_config.repetition_penalty = 1.4
        tokens = np.asarray([[31,32,33],[31,34,35],[32,34,35]])
        children, leaves = trie_lookup(tokens)
        prompt = torch.tensor([[5,31,5,31]])
        candidates = recommend(model,prompt,torch.ones_like(prompt),children,leaves,3,3,
                               SimpleNamespace(pad_token_id=0,eos_token_id=2))
        items,raw = candidates[0]
        full = torch.cat([prompt.expand(len(tokens),-1),torch.tensor(tokens)],1)
        with torch.no_grad():
            logp = model(input_ids=full).logits[:,prompt.shape[1]-1:-1].float().log_softmax(-1)
        expected = logp.gather(2,torch.tensor(tokens)[:,:,None]).squeeze(-1).sum(-1).numpy()
        np.testing.assert_allclose(raw,expected[items],atol=2e-6)

    def test_cache_matches_full_teacher_forcing(self):
        model = small_model()
        prompt = torch.tensor([1, 4, 6, 9])
        items = torch.tensor([[31, 32, 33], [32, 34, 35], [31, 32, 36]])
        cached = cached_item_scores(model, prompt, items, 2)
        full = torch.cat([prompt.expand(len(items), -1), items], 1)
        with torch.no_grad():
            logp = model(input_ids=full).logits[:, len(prompt)-1:-1].float().log_softmax(-1)
        expected = logp.gather(2, items[:, :, None]).squeeze(-1).sum(-1).numpy()
        np.testing.assert_allclose(cached, expected, atol=2e-6)

    def test_new_token_gradient_and_adapter_roundtrip(self):
        model = get_peft_model(small_model(), LoraConfig(task_type="CAUSAL_LM", r=2,
                lora_alpha=4, target_modules=["q_proj", "v_proj"],
                trainable_token_indices={"embed_tokens": [31, 32, 33]}, ensure_weight_tying=True))
        ids = torch.tensor([[1, 31, 32, 33]])
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        for family in ["lora_B", "trainable_tokens"]:
            self.assertGreater(sum(float(p.grad.norm()) for name, p in model.named_parameters()
                                   if family in name and p.grad is not None), 0.)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        optimizer.step()
        model.eval()
        before = model(input_ids=ids).logits.detach()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            after = PeftModel.from_pretrained(small_model(), folder).eval()(input_ids=ids).logits.detach()
        torch.testing.assert_close(before, after, rtol=0., atol=0.)

    def test_shared_prefix_scores_across_lengths_and_chunks(self):
        model = small_model()
        prompt = torch.tensor([1, 4, 6, 9])
        for tokens in [[[31],[32],[31]], [[31,32],[31,33],[32,33],[31,32]],
                       [[31,32,33,34],[31,32,33,35],[31,33,34,35],[31,32,33,34]]]:
            items = torch.tensor(tokens)
            full = torch.cat([prompt.expand(len(items), -1), items], 1)
            with torch.no_grad():
                logp = model(input_ids=full).logits[:,len(prompt)-1:-1].float().log_softmax(-1)
            expected = logp.gather(2,items[:,:,None]).squeeze(-1).sum(-1).numpy()
            for chunk in [1,2,16]:
                np.testing.assert_allclose(cached_item_scores(model,prompt,items,chunk),expected,atol=2e-6)

    def test_shared_prompt_prefill_matches_expanded_beam(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig,LlamaForCausalLM
        from .lcrec import recommend,trie_lookup
        llama=LlamaForCausalLM(LlamaConfig(vocab_size=40,hidden_size=32,intermediate_size=64,
              num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True)).eval()
        tokens=np.asarray([[31,32,33],[31,34,35],[32,34,35]])
        children,leaves=trie_lookup(tokens)
        ids=torch.tensor([[0,0,5,31],[5,31,5,31]])
        mask=(ids!=0).long();tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2)
        for model in [small_model(),llama]:
            reference=recommend(model,ids,mask,children,leaves,3,3,tokenizer,prefill_once=False,vectorized_trie=False)
            cached=recommend(model,ids,mask,children,leaves,3,3,tokenizer,prefill_once=True,vectorized_trie=False)
            for (items,scores),(other_items,other_scores) in zip(reference,cached):
                np.testing.assert_array_equal(items,other_items)
                np.testing.assert_allclose(scores,other_scores,atol=3e-6)
            vectorized=recommend(model,ids,mask,children,leaves,3,3,tokenizer,vectorized_trie=True)
            for (items,scores),(other_items,other_scores) in zip(reference,vectorized):
                np.testing.assert_array_equal(items,other_items)
                np.testing.assert_array_equal(scores,other_scores)

    def test_partition_topk_boundary_ties(self):
        rng = np.random.default_rng(5)
        values = rng.integers(0, 5, 2000).astype(float)
        ids = rng.permutation(len(values))
        expected = ids[np.lexsort((ids, -values))[:10]]
        np.testing.assert_array_equal(stable_topk(values, ids=ids), expected)

    def test_optimizer_and_rng_resume(self):
        config = LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=.1,
                           target_modules=["q_proj", "v_proj"], trainable_token_indices={"embed_tokens": [31, 32, 33]})
        model = get_peft_model(small_model(), config).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
        ids = torch.tensor([[1, 31, 32, 33]])
        def update(m, opt):
            opt.zero_grad()
            m(input_ids=ids, labels=ids).loss.backward()
            opt.step()
        update(model, optimizer)
        saved_rng = torch.get_rng_state()
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, save_embedding_layers=False)
            update(model, optimizer)
            expected = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            resumed = PeftModel.from_pretrained(small_model(), folder, is_trainable=True).train()
            resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.002)
            resumed_optimizer.load_state_dict(saved_optimizer)
            torch.set_rng_state(saved_rng)
            update(resumed, resumed_optimizer)
        for name, param in resumed.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(param, expected[name], rtol=0., atol=0.)

    def test_output_sid_rows_are_trainable(self):
        model = get_peft_model(small_model(), LoraConfig(task_type="CAUSAL_LM", r=2,
                 target_modules=["q_proj", "v_proj"], trainable_token_indices={"embed_tokens": [31, 32, 33]}))
        model.get_output_embeddings()(torch.randn(2, 32))[:, 31:34].sum().backward()
        grad = sum(float(p.grad.norm()) for name, p in model.named_parameters()
                   if "trainable_tokens" in name and p.grad is not None)
        self.assertGreater(grad, 0.)


if __name__ == "__main__":
    unittest.main()
