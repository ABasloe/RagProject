import unittest
from analyze_benchmarks import bootstrap, uncertainty


class AnalysisTests(unittest.TestCase):
    def test_constant_difference_has_exact_interval(self):
        result = bootstrap([-.25] * 10, draws=200)
        self.assertEqual(result, {'mean': -.25, 'low': -.25, 'high': -.25})

    def test_pairing_uses_query_ids_not_row_order(self):
        rows = []
        for method, ids in [('flat', ['a','b']), ('hnsw', ['b','a'])]:
            for qid in ids:
                value = {'a': .8, 'b': .3}[qid] - (.1 if method == 'hnsw' else 0)
                rows.append(dict(method=method, query_id=qid, ann_recall=value, evidence_recall=value, ndcg=value))
        result = uncertainty({'queries': rows})['hnsw']['evidence_recall_minus_exact']
        for value in result.values():
            self.assertAlmostEqual(value, -.1)
        with self.assertRaises(ValueError):
            uncertainty({'queries': rows[:-1]})


if __name__ == '__main__':
    unittest.main()
