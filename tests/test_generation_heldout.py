import unittest
from generation_heldout import validate_partition, evaluation_report, CONDITIONS

class HeldoutTests(unittest.TestCase):
    def test_partition_rejects_leakage_duplicates_and_subsets(self):
        dev = [str(i) for i in range(60)]
        held = [str(i) for i in range(60,300)]
        validate_partition(dev,held)
        for bad in (held[:-1], held[:-1]+[held[0]], held[:-1]+[dev[0]]):
            with self.assertRaises(ValueError): validate_partition(dev,bad)

    def test_paired_report_and_invalid_predictions(self):
        rows = [dict(query_id=str(i),condition=c,gold_verdict='SUPPORTED',
                     predicted_verdict='INVALID' if c=='ivf_flat' else 'SUPPORTED')
                for i in range(3) for c in CONDITIONS]
        report = evaluation_report(rows)
        self.assertEqual(report['paired_accuracy_difference_vs_flat']['hnsw']['ci95'],[0,0])
        self.assertEqual(report['paired_accuracy_difference_vs_flat']['ivf_flat']['ci95'],[-1,-1])
        self.assertEqual(report['confusion']['ivf_flat']['SUPPORTED']['INVALID'],3)
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): evaluation_report(bad)
