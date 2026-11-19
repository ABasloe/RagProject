import json,csv,re
from pathlib import Path
p=json.load(open('results/generation-algorithms-3b-256/results.json'))
claims={str(c['id']):c for c in map(json.loads,open('data/scifact-original/claims_dev.jsonl'))}
corpus={str(c['doc_id']):c for c in map(json.loads,open('data/scifact-original/corpus.jsonl'))}
notes={
'212':'Direction reversal with gold context: reduced drift/younger age is called support for higher age. Retrieval also invents an expansion of CR.',
'42':'Gold response reverses protection against anemia into support for increased vulnerability. Retrieval abstains despite the protective title.',
'1221':'Gold response overlooks shared versus non-shared mutation signatures and parallel evolution. Retrieval context must be assessed separately.',
'911':'Gold context ends before knockout findings; response also treats claim spelling/isoform differences literally.',
'742':'Gold abstract is cut during results before the macrolide finding.',
'338':'Gold abstract ends before bleeding outcomes. All retrieval responses omit the explanation field, independently of verdict correctness.',
'1144':'Gold abstract is cut before diabetes estimates; response requests before/after implementation observations although the source is a modeling study.',
'1359':'Gold abstract ends before abstinence results; claim wording also conflates combination pharmacotherapies with nicotine replacement.',
'533':'Gold abstract is cut during results before the full fibrinogen/patency finding. Numeric gold citation is normalization-only, verdict remains wrong.',
'1339':'Gold abstract is cut before traumatic-procedure results. Numeric gold citation is normalization-only, verdict remains wrong.',
'163':'Correct label but explanation infers positive mental-health impact from study aims, without visible outcome evidence.',
'1259':'Correct label but supplied text primarily establishes metabolism mechanism and study design, not the outcome association.',
'756':'Correct label; explanation describes lysine acetylation, but does not explicitly establish the breadth asserted by many proteins in human cells.',
'985':'Correct label with visible microRNA competition mechanism and PTENP1 regulation.',
'386':'Correct label supported by visible bolus and multiple-step error findings.',
'674':'Correct gold refutation directly supported by visible causal-role sentence.'}
fields=['query_id','condition','gold_verdict','predicted_verdict','strict_valid','normalized_valid','gold_document_in_context','complete_rationale_sets_visible','total_rationale_sets','review_note','raw_output','supplied_passages']
rows=[]
for r in p['queries']:
 if r['condition']!='gold_evidence' and r['normalized_predicted_verdict']==r['gold_verdict']: continue
 norm=lambda x:' '.join(x.split())
 visible=total=0
 for doc, evidence in claims[r['query_id']]['evidence'].items():
  for item in evidence:
   total+=1
   visible+=all(norm(corpus[doc]['abstract'][i]) in norm(r['supplied_passages']) for i in item['sentences'])
 row={k:r[k] for k in ['query_id','condition','gold_verdict','raw_output','supplied_passages']}
 row.update(predicted_verdict=r['normalized_predicted_verdict'],strict_valid=r['parsed'] is not None,normalized_valid=r['normalized_parsed'] is not None,gold_document_in_context=bool(set(r['gold_evidence_ids'])&set(r['included_ids'])),complete_rationale_sets_visible=visible,total_rationale_sets=total,review_note=notes.get(r['query_id'],'No passages: correct evidence-only abstention.' if r['gold_verdict']=='INSUFFICIENT_EVIDENCE' else 'No-context abstention is expected instruction following, but incorrect against the dataset label.'))
 rows.append(row)
with open('docs/generation-development-audit.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
for r in rows:
 if r['condition']=='gold_evidence': print(r['query_id'],r['complete_rationale_sets_visible'],r['total_rationale_sets'])
print('Reviewed rows:',len(rows))
