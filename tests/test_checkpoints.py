import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark import encode_checkpointed
from run_fiqa import select_settings


class FakeModel:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError('interrupted')
        return np.array([[float(text), 1.0] for text in texts])


class CheckpointTests(unittest.TestCase):
    def test_interruption_reuses_only_completed_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            texts = ['1', '2', '3', '4', '5']
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                encode_checkpointed(FakeModel(fail_on=2), texts, root, 'revision', 2, chunk_size=2)
            model = FakeModel()
            vectors = encode_checkpointed(model, texts, root, 'revision', 2, chunk_size=2)
            self.assertEqual(model.calls, 2)
            np.testing.assert_array_equal(vectors[:, 0], [1, 2, 3, 4, 5])
            cached = FakeModel(fail_on=1)
            np.testing.assert_array_equal(encode_checkpointed(cached, texts, root, 'revision', 2, chunk_size=2), vectors)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                encode_checkpointed(cached, texts, root, 'different-revision', 2, chunk_size=2)

    def test_selection_excludes_fast_low_recall_settings(self):
        rows = [dict(method='hnsw', parameter=16, median_ms=.01, ann_recall=.9),
                dict(method='hnsw', parameter=32, median_ms=.02, ann_recall=.97),
                dict(method='hnsw', parameter=64, median_ms=.04, ann_recall=.99),
                dict(method='ivf_flat', parameter=16, median_ms=.03, ann_recall=.96)]
        self.assertEqual(select_settings(rows), {'ef_search': 32, 'nprobe': 16})
        with self.assertRaises(RuntimeError):
            select_settings(rows, target=1.0)


if __name__ == '__main__':
    unittest.main()
