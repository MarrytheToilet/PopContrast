"""Check the vectorized ranking against the production stable tie rule."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from .common import stable_topk
from .estimator_resampling import CandidateTable


class CandidateTableTests(unittest.TestCase):
    def test_ragged_scores_and_boundary_ties(self):
        rng=np.random.default_rng(42)
        ids=[rng.choice(100,n,replace=False) for n in [10,17,34]]
        scores=[rng.integers(-6,0,len(x)).astype(float) for x in ids]
        prior=rng.integers(0,4,100).astype(float)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"candidates.npz"
            np.savez(p,indices=np.arange(3),targets=np.zeros(3,dtype=int),items=np.concatenate(ids),
                     raw=np.concatenate(scores),offsets=np.r_[0,np.cumsum([len(x) for x in ids])])
            table=CandidateTable(p)
            for beta in [0.,.5,2.]:
                expected=np.array([stable_topk(s-beta*prior[i],ids=i) for i,s in zip(ids,scores)])
                np.testing.assert_array_equal(table.rank(prior,beta),expected)


if __name__=="__main__":unittest.main()
