import unittest
import numpy as np
from .train import LengthSampler


class SamplerTests(unittest.TestCase):
    def test_same_optimizer_batches_across_sid_length_and_microbatch(self):
        lengths=np.random.default_rng(18).integers(1,51,131413)
        baseline=LengthSampler(lengths*3,256,1)
        longer=LengthSampler(lengths*6,64,1,effective_batch_size=256)
        for epoch in [0,1,19]:
            baseline.epoch=longer.epoch=epoch
            full=list(baseline);micro=list(longer)
            self.assertEqual(len(micro),4*len(full))
            self.assertEqual(len(micro),len(longer))
            for i,batch in enumerate(full):
                self.assertEqual(batch,sum(micro[4*i:4*i+4],[]))


if __name__=="__main__":
    unittest.main()
