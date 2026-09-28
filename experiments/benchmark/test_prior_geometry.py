import unittest
import numpy as np
from .prior_geometry import personalization_decomposition


class GeometryTests(unittest.TestCase):
    def test_context_constants_do_not_change_diagnostics(self):
        rng=np.random.default_rng(7)
        score=rng.normal(size=(8,30))
        first=personalization_decomposition(score)
        second=personalization_decomposition(score+rng.normal(size=(8,1))*10)
        for key in first:
            if key!="scope":self.assertAlmostEqual(first[key],second[key],places=12)

    def test_no_personalization(self):
        score=np.tile(np.arange(10,dtype=float),(8,1))
        row=personalization_decomposition(score)
        self.assertEqual(row["item_main_effect_fraction"],1.)
        self.assertEqual(row["history_item_residual_variance"],0.)
        self.assertAlmostEqual(row["catalog_softmax_history_JS_nats"],0.,places=12)


if __name__=="__main__":unittest.main()
