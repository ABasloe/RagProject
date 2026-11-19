import unittest
from generation_pilot import choose_claims, gold_label, parse_response, pack_context


class CharTokenizer:
    def encode(self, text, **kwargs): return list(text)
    def decode(self, tokens, **kwargs): return ''.join(tokens)


class GenerationTests(unittest.TestCase):
    def test_gold_labels_are_not_inferred_from_cited_documents(self):
        self.assertEqual(gold_label({'id':1,'evidence':{},'cited_doc_ids':[23]}),'INSUFFICIENT_EVIDENCE')
        self.assertEqual(gold_label({'id':1,'evidence':{'23':[{'label':'CONTRADICT'}]}}),'REFUTED')
        with self.assertRaises(ValueError): gold_label({'id':1,'evidence':{'23':[{'label':'SUPPORT'},{'label':'CONTRADICT'}]}})

    def test_balanced_selection_is_confined_to_development(self):
        claims={str(i):{'id':i,'evidence':{} if i%3==0 else {'1':[{'label':'SUPPORT' if i%3==1 else 'CONTRADICT'}]}} for i in range(30)}
        allowed=[str(i) for i in range(15)]
        chosen=choose_claims(claims,allowed,9)
        self.assertTrue(set(chosen)<=set(allowed))
        self.assertEqual(chosen,choose_claims(claims,allowed,9))
        self.assertEqual(len(set(chosen)),9)
        self.assertEqual([sum(gold_label(claims[q])==l for q in chosen) for l in ['SUPPORTED','REFUTED','INSUFFICIENT_EVIDENCE']],[3,3,3])

    def test_invalid_format_is_not_silently_counted_as_abstention(self):
        self.assertIsNone(parse_response('The claim is supported.'))
        self.assertIsNone(parse_response('{"verdict":"SUPPORTED","explanation":"Yes","citations":[23]}'))
        self.assertIsNone(parse_response('[]'))
        valid='{"verdict":"SUPPORTED","explanation":"Evidence supports it.","citations":["23"]}'
        self.assertEqual(parse_response(valid)['verdict'],'SUPPORTED')
        self.assertEqual(parse_response('```json\n'+valid+'\n```'),parse_response(valid))

    def test_context_budget_and_document_order(self):
        text,ids,truncated=pack_context(CharTokenizer(),['a','b'],{'a':'abcdef','b':'ghijkl'},budget=21,per_doc=3)
        self.assertLessEqual(len(text),21)
        self.assertEqual(ids,['a'])
        self.assertEqual(truncated,['a'])
        self.assertIn('abc',text)


if __name__=='__main__': unittest.main()
