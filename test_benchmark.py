import unittest
from benchmark import metrics


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
