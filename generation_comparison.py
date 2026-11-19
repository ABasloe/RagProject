"""Replay development pilot prompts with a pinned larger generator (no retrieval timing)."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from benchmark import atomic_json
from generation_pilot import ROOT, LABELS, parse_response

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
CONDITIONS = ('no_context', 'flat', 'gold_evidence')
ALL_CONDITIONS = ('no_context', 'flat', 'hnsw', 'ivf_flat', 'gold_evidence')


def normalized_response(text):
    """Normalize only unambiguous citation-ID formatting; never repair content."""
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get('citations'), list):
            return None
        citations = []
        for citation in value['citations']:
            if type(citation) is int and citation >= 0:
                citation = str(citation)
            elif isinstance(citation, str):
                citation = citation.strip()
                match = re.fullmatch(r'Document\s+([0-9]+)', citation)
                if match:
                    citation = match.group(1)
            else:
                return None
            citations.append(citation)
        value['citations'] = citations
        return parse_response(json.dumps(value))
    except (ValueError, TypeError):
        return None


def finish_reason(token_ids, max_new_tokens, eos_token_ids):
    """Distinguish natural stopping from an output cut off by the budget."""
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    if token_ids and token_ids[-1] in (eos_token_ids or []):
        return 'eos'
    return 'length' if len(token_ids) >= max_new_tokens else 'other'


def source_rows(source, conditions=CONDITIONS):
    data = json.loads(source.read_text())
    dev = json.loads((ROOT/'results/scifact-dev/results.json').read_text())
    allowed = set(dev['metadata']['query_ids'])
    rows = [r for r in data['queries'] if r['condition'] in conditions]
    selected = data['manifest']['selected_ids']
    expected = {(q, c) for q in selected for c in conditions}
    actual = [(r['query_id'], r['condition']) for r in rows]
    if not set(selected) <= allowed or len(set(selected)) != 24:
        raise ValueError('Expected 24 unique development claims')
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Missing or duplicate source outputs')
    return data, rows


def summarize(rows, conditions=CONDITIONS):
    import numpy as np
    result = []
    for condition in conditions:
        group = [r for r in rows if r['condition'] == condition]
        if not group:
            continue
        f1 = []
        for label in LABELS:
            tp = sum(r['gold_verdict'] == label and r['predicted_verdict'] == label for r in group)
            fp = sum(r['gold_verdict'] != label and r['predicted_verdict'] == label for r in group)
            fn = sum(r['gold_verdict'] == label and r['predicted_verdict'] != label for r in group)
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        result.append(dict(condition=condition, count=len(group),
            accuracy=sum(r['gold_verdict']==r['predicted_verdict'] for r in group)/len(group),
            macro_f1=sum(f1)/3, valid_json_rate=sum(r['parsed'] is not None for r in group)/len(group),
            output_limit_hits=sum(r['hit_output_limit'] for r in group),
            nonempty_citations=sum(bool(r['parsed'] and r['parsed']['citations']) for r in group),
            citations_outside_context=sum(bool(r['citations_outside_context']) for r in group),
            median_generation_ms=float(np.median([r['generation_ms'] for r in group]))))
        normalized = [normalized_response(r['raw_output']) for r in group] if all('raw_output' in r for r in group) else None
        if normalized is not None:
            predictions = [r['verdict'] if r else 'INVALID' for r in normalized]
            nf1 = []
            for label in LABELS:
                tp = sum(r['gold_verdict']==label and p==label for r,p in zip(group,predictions))
                fp = sum(r['gold_verdict']!=label and p==label for r,p in zip(group,predictions))
                fn = sum(r['gold_verdict']==label and p!=label for r,p in zip(group,predictions))
                nf1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
            result[-1].update(normalized_accuracy=sum(r['gold_verdict']==p for r,p in zip(group,predictions))/len(group),
                              normalized_macro_f1=sum(nf1)/3,
                              normalized_schema_valid_rate=sum(p is not None for p in normalized)/len(group),
                              normalized_citations_outside_context=sum(bool(set(p['citations'])-set(r['included_ids'])) for r,p in zip(group,normalized) if p is not None))
    return result


def worker(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = args.output
    source, rows = source_rows(args.source, args.conditions)
    revision = (ROOT/'data/generator-3b-revision.txt').read_text().strip()
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Expected immutable model revision')
    manifest = dict(generator=MODEL, revision=revision, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['generation_comparison.py','generation_pilot.py','benchmark.py']},
        selected_ids=source['manifest']['selected_ids'], conditions=args.conditions,
        citation_normalization='v1: trim strings; strip Document prefix from digit IDs; convert nonnegative integer IDs; strict and normalized scoring retained separately',
        protocol='Exact saved prompt replay; generation timing only; no online retrieval',
        max_new_tokens=args.max_new_tokens, dtype='float32', threads=1, decoding='greedy', seed=2026,
        torch=torch.__version__, transformers=transformers.__version__, python=sys.version)
    # Normalize tuples to the JSON representation before comparing on resume.
    manifest = json.loads(json.dumps(manifest))
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Comparison inputs changed; use a new output directory')
    atomic_json(path, manifest)
    atomic_json(out/'status.json',dict(stage='loading_model',total=len(rows),pid=os.getpid()))
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=revision,cache_dir=args.cache_dir,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,cache_dir=args.cache_dir,local_files_only=True,dtype=torch.float32).eval()
    with torch.inference_mode():
        model.generate(**tokenizer('Return a short greeting.',return_tensors='pt'),max_new_tokens=4,do_sample=False)
    checkpoints = out/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    outputs = []
    for source_row in rows:
        qid, condition = source_row['query_id'], source_row['condition']
        path = checkpoints/f'{qid}-{condition}.json'
        if path.exists():
            outputs.append(json.loads(path.read_text()))
            continue
        atomic_json(out/'status.json',dict(stage='running',completed=len(outputs),total=len(rows),query_id=qid,condition=condition,pid=os.getpid()))
        print(f'Generating {len(outputs)+1}/{len(rows)}: {qid} {condition}',flush=True)
        encoded = tokenizer(source_row['prompt'],return_tensors='pt',add_special_tokens=False)
        n = int(encoded['input_ids'].shape[1])
        if n != source_row['input_tokens'] or n > 2048:
            raise ValueError('Tokenization differs from baseline; comparison requires identical input budgets')
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        elapsed = (time.perf_counter()-start)*1000
        generated = output[0,n:]
        text = tokenizer.decode(generated,skip_special_tokens=True)
        parsed = parse_response(text)
        normalized = normalized_response(text)
        citations = set(parsed['citations']) if parsed else set()
        reason = finish_reason(generated.tolist(), args.max_new_tokens, model.generation_config.eos_token_id)
        row = {k:source_row[k] for k in ['query_id','condition','claim','gold_verdict','prompt','supplied_passages','included_ids','gold_evidence_ids','truncated_ids']}
        row.update(predicted_verdict=parsed['verdict'] if parsed else 'INVALID',raw_output=text,parsed=parsed,
            input_tokens=n,output_tokens=len(generated),hit_output_limit=reason=='length',generation_ms=elapsed,
            max_new_tokens=args.max_new_tokens,finish_reason=reason,reached_token_budget=len(generated)>=args.max_new_tokens,
            citations_outside_context=sorted(citations-set(row['included_ids'])),
            citations_matching_gold_documents=sorted(citations & set(row['gold_evidence_ids'])))
        normalized_citations = set(normalized['citations']) if normalized else set()
        row.update(normalized_parsed=normalized,
                   normalized_predicted_verdict=normalized['verdict'] if normalized else 'INVALID',
                   normalized_citations_outside_context=sorted(normalized_citations-set(row['included_ids'])),
                   normalized_citations_matching_gold_documents=sorted(normalized_citations & set(row['gold_evidence_ids'])))
        atomic_json(path,row)
        outputs.append(row)
        atomic_json(out/'summary.json',summarize(outputs,args.conditions))
    summary = summarize(outputs,args.conditions)
    atomic_json(out/'results.json',dict(manifest=manifest,summary=summary,queries=outputs,baseline_summary=summarize(rows,args.conditions)))
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    if not (out/'manual-review.csv').exists():
        shuffled = outputs.copy(); random.Random(2026).shuffle(shuffled)
        with (out/'manual-review.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['review_id','claim','supplied_passages','response','explanation_supported','citations_support_explanation','notes'])
            writer.writeheader()
            for i,r in enumerate(shuffled):
                writer.writerow(dict(review_id=i,claim=r['claim'],supplied_passages=r['supplied_passages'],response=r['raw_output']))
        atomic_json(out/'review-key.json',{str(i):dict(query_id=r['query_id'],condition=r['condition']) for i,r in enumerate(shuffled)})
    atomic_json(out/'status.json',dict(stage='complete',completed=len(outputs),total=len(rows)))
    print('Development comparison complete.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/generation-pilot/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/generation-comparison-3b')
    parser.add_argument('--cache-dir',default='/tmp/ragproject-model-cache')
    parser.add_argument('--max-new-tokens',type=int,default=96,help='Output token budget; use a new output directory when changing it')
    parser.add_argument('--conditions',nargs='+',choices=ALL_CONDITIONS,default=list(CONDITIONS))
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    if len(set(args.conditions)) != len(args.conditions):
        parser.error('conditions must be unique')
    args.output = args.output.resolve(); args.source = args.source.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    lock = os.fdopen(args.lock_fd,'a') if args.lock_fd is not None else (args.output/'run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1,'Comparison already running.\n')
    if args.background:
        with (args.output/'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--source',str(args.source),'--output',str(args.output),'--cache-dir',args.cache_dir,'--max-new-tokens',str(args.max_new_tokens),'--conditions',*args.conditions,'--lock-fd',str(lock.fileno())],
                cwd=ROOT,env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        (args.output/'run.pid').write_text(str(child.pid)+'\n')
        print(f'Comparison PID {child.pid}; log {args.output / "run.log"}')
        return
    try:
        worker(args)
    except Exception as exc:
        atomic_json(args.output/'status.json',dict(stage='failed',error=str(exc)))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
