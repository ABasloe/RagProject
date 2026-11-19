import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from run_scaling import nested_subsets, stage


class ScalingTests(unittest.TestCase):
    def test_subsets_are_nested_reproducible_and_preserve_relevance(self):
        ids = [str(i) for i in range(100)]
        required = {'3','60','99'}
        subsets = nested_subsets(ids, required, [10,30,100], 42)
        self.assertEqual(subsets, nested_subsets(ids, required, [10,30,100], 42))
        previous = set()
        for size, indices in subsets.items():
            self.assertEqual(len(indices), size)
            self.assertEqual(indices, sorted(indices))
            self.assertTrue(previous.issubset(indices))
            self.assertTrue(required.issubset({ids[i] for i in indices}))
            previous = set(indices)
        self.assertEqual(subsets[100], list(range(100)))

    def test_insufficient_size_or_missing_relevant_document_fails(self):
        with self.assertRaises(ValueError): nested_subsets(['a','b'], {'a','b'}, [1,2], 42)
        with self.assertRaises(ValueError): nested_subsets(['a','b'], {'c'}, [1,2], 42)

    def test_real_faiss_stage_and_checkpoint_reuse(self):
        faiss.omp_set_num_threads(1)
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(5000,8)).astype('float32')
        faiss.normalize_L2(vectors)
        ids = [str(i) for i in range(len(vectors))]
        queries = {'qids':['q1','q2'], 'vectors':vectors[:2].copy(), 'qrels':{'q1':{'0':1},'q2':{'1':1}}}
        with tempfile.TemporaryDirectory() as directory, patch('run_scaling.OUT',Path(directory)):
            result = stage(5000,'test',101,ids,vectors,queries,{'ef_search':32,'nprobe':128})
            for row in result:
                self.assertEqual(row['evidence_recall'],1)
                if row['method'] in ('flat','ivf_flat'): self.assertEqual(row['ann_recall'],1)
            with patch('run_scaling.build_index',side_effect=AssertionError('Must reuse completed checkpoints')):
                self.assertEqual(result,stage(5000,'test',101,ids,vectors,queries,{'ef_search':32,'nprobe':128}))


if __name__=='__main__': unittest.main()
