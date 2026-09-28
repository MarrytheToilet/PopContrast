"""Actual vendored T5 cache path versus original teacher-forced scorer."""
import unittest
import torch
from .common import load_model
from .search import encode, score_items, score_items_cached


class CachedT5Tests(unittest.TestCase):
    def test_cache_scores_match_multiple_lengths_and_chunks(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        for length in [3,4,5,6]:
            model=load_model(sid_length=length,device="cpu").eval()
            ids=torch.tensor([[0,0,1,2,3,4,5,6]])
            mask=(ids!=0).long()
            enc=encode(model,ids,mask)
            tokens=torch.randint(1,256*length,(7,length))
            tokens[-1]=tokens[0]
            reference=score_items(model,enc,mask,tokens,3)
            for chunk in [1,3,20]:
                cached=score_items_cached(model,enc,mask,tokens,chunk)
                torch.testing.assert_close(cached,reference,rtol=0,atol=1e-5)
                torch.testing.assert_close(cached[-1],cached[0],rtol=0,atol=1e-5)


if __name__=="__main__":
    unittest.main()
