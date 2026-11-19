import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generation_comparison import source_rows, summarize, finish_reason, normalized_response, ALL_CONDITIONS, ROOT


class ComparisonTests(unittest.TestCase):
    def test_normalization_does_not_repair_missing_content(self):
        value = dict(verdict='SUPPORTED', explanation='Evidence supports it.', citations=['Document 123',456])
        self.assertEqual(normalized_response(json.dumps(value))['citations'],['123','456'])
        del value['explanation']
        self.assertIsNone(normalized_response(json.dumps(value)))
        value['explanation']='Evidence supports it.'
        for bad in (True, 123.5, None):
            value['citations']=[bad]
            self.assertIsNone(normalized_response(json.dumps(value)))
        value['citations']=['Document 123 and 456']
        self.assertEqual(normalized_response(json.dumps(value))['citations'],['Document 123 and 456'])

    def test_all_five_conditions_and_separate_scoring(self):
        row = dict(condition='hnsw',gold_verdict='SUPPORTED',predicted_verdict='INVALID',parsed=None,
                   hit_output_limit=False,citations_outside_context=[],generation_ms=1,included_ids=['123'],
                   raw_output=json.dumps(dict(verdict='SUPPORTED',explanation='Supported.',citations=[123])))
        result = summarize([row],ALL_CONDITIONS)[0]
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['normalized_accuracy'],1)
        self.assertEqual(result['normalized_citations_outside_context'],0)

    def test_eos_at_budget_is_not_truncation(self):
        self.assertEqual(finish_reason([4,5,9],3,[8,9]),'eos')
        self.assertEqual(finish_reason([4,5,6],3,[8,9]),'length')
        self.assertEqual(finish_reason([4,9],3,9),'eos')
        self.assertEqual(finish_reason([4,5],3,None),'other')

    def test_only_complete_development_pairs_are_allowed(self):
        ids = [str(i) for i in range(24)]
        source = {'manifest': {'selected_ids': ids}, 'queries': [
            {'query_id': qid, 'condition': condition} for qid in ids for condition in ALL_CONDITIONS]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root/'results/scifact-dev/results.json'
            dev.parent.mkdir(parents=True)
            dev.write_text(json.dumps({'metadata': {'query_ids': ids}}))
            path = Path(directory)/'source.json'
            path.write_text(json.dumps(source))
            with patch('generation_comparison.ROOT', root):
                _, rows = source_rows(path)
                self.assertEqual(len(rows), 72)
                self.assertEqual(len(source_rows(path, ALL_CONDITIONS)[1]), 120)
                broken = copy.deepcopy(source)
                broken['queries'].append(rows[0])
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError): source_rows(path)
                broken = copy.deepcopy(source)
                broken['manifest']['selected_ids'][0] = 'not-a-development-query'
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError): source_rows(path)

    def test_invalid_predictions_penalize_accuracy_and_f1(self):
        rows = [dict(condition='flat', gold_verdict=label, predicted_verdict=prediction,
                     parsed={} if prediction!='INVALID' else None, hit_output_limit=False,
                     citations_outside_context=[],generation_ms=10)
                for label,prediction in [('SUPPORTED','SUPPORTED'),('REFUTED','INVALID'),('INSUFFICIENT_EVIDENCE','INSUFFICIENT_EVIDENCE')]]
        score = summarize(rows)[0]
        self.assertAlmostEqual(score['accuracy'],2/3)
        self.assertAlmostEqual(score['macro_f1'],2/3)
        self.assertAlmostEqual(score['valid_json_rate'],2/3)


if __name__ == '__main__': unittest.main()
