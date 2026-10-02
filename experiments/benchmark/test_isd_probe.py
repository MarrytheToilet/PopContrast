"""Small correctness tests for the independently implemented ISD probe."""
import unittest

import numpy as np

from .isd_probe import BASES, build_statistics, choose, evaluate, rerank
from .evaluate import BETAS


class ISDProbeTests(unittest.TestCase):
    def test_overlapping_histories_count_each_event_once(self):
        data = {'pop_counts': np.zeros(3), 'train_history': np.array([[-1, 0], [0, 1]]),
                'train_target': np.array([1, 2]), 'train_user': np.array([0, 0])}
        transition, co, count = build_statistics(data, window=2)
        np.testing.assert_array_equal(count, [1, 1, 1])
        self.assertEqual(transition.nnz, 2)
        self.assertAlmostEqual(transition[0, 1], np.log(2))
        np.testing.assert_allclose(co.toarray(), np.ones((3, 3)) - np.eye(3))

    def test_padding_does_not_enter_coverage(self):
        top = rerank(np.array([0, 1, 2]), np.array([3., 2., 1.]),
                     np.array([np.inf, 1., 2.]), np.array([False, True, True]),
                     np.zeros(3), fusion=True)
        np.testing.assert_array_equal(top, [1, 2] + [-1] * 8)
        rows, hits = evaluate(np.stack([top, top]), np.array([1, 0]),
                              np.array([True, False, False]), 3)
        self.assertEqual(rows['cov10'], 2 / 3)
        np.testing.assert_array_equal(hits, [True, False])

    def test_validation_budget_and_zero_strength_tie(self):
        rows = {}
        for family, base in BASES.items():
            rows[base] = {'R10': .1, 'tailR10': .02}
            for beta in BETAS:
                rows[f'{family}:{beta}'] = {'R10': .1, 'tailR10': .02}
            rows[f'{family}:2.0'] = {'R10': .09, 'tailR10': .9}
        self.assertTrue(all(name.endswith(':0.0') for name in choose(rows).values()))


if __name__ == '__main__':
    unittest.main()
