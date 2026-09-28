"""Correctness checks for mathematical claims used by the new decoder."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .common import stable_topk
from .search import beam_search, beam_search_batch, certified_search, make_trie, score_items, marginal_potential, batched_best_first


class ToyModel:
    """Normalized autoregressive distribution with deliberate prefix pruning."""
    def __call__(self, encoder_outputs, attention_mask, decoder_input_ids=None,
                 labels=None, use_cache=False):
        if labels is not None:
            decoder_input_ids = torch.cat([torch.zeros((len(labels), 1), dtype=torch.long), labels[:, :-1]], 1)
        logits = torch.zeros((len(decoder_input_ids), decoder_input_ids.shape[1], 6))
        for b, row in enumerate(decoder_input_ids):
            for pos in range(len(row)):
                logits[b, pos] = torch.tensor([-20., 3., 1., 0., -1., -2.]) if pos == 0 else torch.tensor([-20., -20., -20., 2. * float(row[pos] == 1), 1., 0.])
        return SimpleNamespace(logits=logits)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.tensor([[1, 3], [1, 4], [2, 3], [2, 4], [2, 4]])
        self.prior = np.array([1., 1., -3., -2., -2.])
        self.trie = make_trie(self.tokens.tolist(), self.prior)
        self.enc, self.mask = torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.long)
        self.model = ToyModel()

    def test_upper_bound_for_every_completion(self):
        scores = score_items(self.model, self.enc, self.mask, self.tokens).numpy()
        for beta in [0, .75, 1.5]:
            for prefix, prior_min in self.trie.min_prior.items():
                if not prefix:
                    prefix_score = 0.
                else:
                    prefix_score = float(score_items(self.model, self.enc, self.mask, torch.tensor([prefix]))[0])
                bound = prefix_score - beta * prior_min
                for item, tokens in enumerate(self.tokens.tolist()):
                    if tuple(tokens[:len(prefix)]) == prefix:
                        self.assertLessEqual(scores[item] - beta * self.prior[item], bound + 1e-6)

    def test_wide_beam_and_certificate_match_exhaustive(self):
        for beta in [0, .75, 1.5]:
            scores = score_items(self.model, self.enc, self.mask, self.tokens).numpy() - beta * self.prior
            expected = stable_topk(scores)
            for mode in ["raw", "bound"]:
                got = beam_search(self.model, self.enc, self.mask, self.trie, self.prior, 10, beta, mode)
                np.testing.assert_array_equal(got["top"], expected)
            got = certified_search(self.model, self.enc, self.mask, self.trie, self.prior, beta, k=3)
            self.assertTrue(got["certified"])
            np.testing.assert_array_equal(got["top"], expected[:3])

    def test_budget_stop_is_not_certified(self):
        got = certified_search(self.model, self.enc, self.mask, self.trie, self.prior, .75, k=3, max_expansions=0)
        self.assertFalse(got["certified"])

    def test_batched_best_first_certificate_and_budget(self):
        seed=beam_search(self.model,self.enc,self.mask,self.trie,self.prior,1)
        for beta in [0,.75,1.5]:
            exact=score_items(self.model,self.enc,self.mask,self.tokens).numpy()-beta*self.prior
            for batch in [1,2,32]:
                got=batched_best_first(self.model,self.enc,self.mask,self.trie,self.prior,beta,
                                     seed_candidates=seed,k=3,max_expansions=100,batch_size=batch)
                self.assertTrue(got["certified"])
                np.testing.assert_array_equal(got["top"],stable_topk(exact,k=3))
            stopped=batched_best_first(self.model,self.enc,self.mask,self.trie,self.prior,beta,
                                      seed_candidates=seed,k=3,max_expansions=0)
            self.assertFalse(stopped["certified"])
            self.assertEqual(stopped["expanded_prefixes"],0)
            np.testing.assert_array_equal(stopped["top"],stable_topk(seed["raw_scores"]-beta*self.prior[seed["items"]],k=3,ids=seed["items"]))

    def test_narrow_beam_can_recover_pruned_item(self):
        raw = beam_search(self.model, self.enc, self.mask, self.trie, self.prior, 1, 1.5, "raw")
        bound = beam_search(self.model, self.enc, self.mask, self.trie, self.prior, 1, 1.5, "bound")
        self.assertNotIn(2, raw["items"])
        self.assertIn(2, bound["items"])

    def test_common_score_replay_agrees_under_full_inclusion(self):
        scores = np.array([3., 2., 2., 0.])
        exact = stable_topk(scores, 2)
        candidates = np.array([3, 2, 1, 0])
        replay = stable_topk(scores[candidates], 2, ids=candidates)
        np.testing.assert_array_equal(exact, replay)

    def test_batched_search_preserves_separate_beams(self):
        for mode in ["raw", "bound", "terminal", "potential"]:
            self.trie.potential = marginal_potential(self.tokens, -np.arange(5.), self.prior, .75)
            for width in [1, 3, 10]:
                expected = beam_search(self.model, self.enc, self.mask, self.trie, self.prior, width, .75, mode)
                batch = beam_search_batch(self.model, self.enc.expand(3,-1,-1), self.mask.expand(3,-1),
                                          self.trie, self.prior, width, .75, mode)
                for actual in batch:
                    np.testing.assert_array_equal(actual["items"], expected["items"])
                    np.testing.assert_allclose(actual["raw_scores"], expected["raw_scores"])
                    np.testing.assert_array_equal(actual["top"], expected["top"])

    def test_mass_potential_zero_strength_and_leaf_identity(self):
        marginal = np.array([-1., -2., -4., -5., -5.])
        for beta in [0., .75]:
            values = marginal_potential(self.tokens, marginal, self.prior, beta)
            if beta == 0:
                self.assertTrue(all(value == 0 for value in values.values()))
            for item, row in enumerate(self.tokens):
                self.assertAlmostEqual(values[tuple(row.tolist())], -beta * self.prior[item], places=6)
            self.trie.potential = values
            result = beam_search(self.model, self.enc, self.mask, self.trie, self.prior, 10, beta, "potential")
            scores = score_items(self.model, self.enc, self.mask, self.tokens).numpy() - beta * self.prior
            np.testing.assert_array_equal(result["top"], stable_topk(scores))

    def test_potential_is_expected_tilt_not_extreme_leaf(self):
        marginal = np.array([-1., -2., -4., -5., -5.])
        values = marginal_potential(self.tokens, marginal, self.prior, .75)
        weights = np.exp(marginal) / np.exp(marginal).sum()
        expected = np.log((weights * np.exp(-.75 * self.prior)).sum())
        self.assertAlmostEqual(values[()], expected, places=6)
        self.assertLess(values[()], -.75 * self.prior.min())


if __name__ == "__main__":
    unittest.main()
