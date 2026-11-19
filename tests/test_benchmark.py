import unittest
from benchmark import metrics, select_queries


class PartitionTests(unittest.TestCase):
    def test_holdout_is_disjoint_complete_and_reproducible(self):
        ids = [str(i) for i in range(20)]
        texts = ["query " + qid for qid in ids]
        dev, dev_texts = select_queries(ids, texts, "dev", 0.2, 42)
        heldout, _ = select_queries(ids, texts, "heldout", 0.2, 42)
        self.assertEqual(len(dev), 4)
        self.assertFalse(set(dev) & set(heldout))
        self.assertEqual(set(dev + heldout), set(ids))
        self.assertEqual(dev_texts, ["query " + qid for qid in dev])
        self.assertEqual((dev, dev_texts), select_queries(ids, texts, "dev", 0.2, 42))
        reversed_dev, _ = select_queries(ids[::-1], texts[::-1], "dev", 0.2, 42)
        self.assertEqual(set(dev), set(reversed_dev))

    def test_small_holdout(self):
        self.assertEqual(len(select_queries(["a", "b"], ["A", "B"], "dev", 0.2, 42)[0]), 1)
        with self.assertRaises(ValueError):
            select_queries(["a"], ["A"], "dev", 0.2, 42)


class MetricsTests(unittest.TestCase):
    def test_distinguishes_ann_and_relevance(self):
        ann, recall, ndcg = metrics(["a", "c"], ["a", "b"], {"c": 1}, 2)
        self.assertEqual(ann, 0.5)
        self.assertEqual(recall, 1.0)
        self.assertAlmostEqual(ndcg, 1 / __import__("math").log2(3))

    def test_missing_results(self):
        self.assertEqual(metrics(["a", None], ["a", "b"], {"b": 1}, 2), (0.5, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
