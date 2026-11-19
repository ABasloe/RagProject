"""Frozen SciFact held-out generation evaluation."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from generation_comparison import normalized_response, finish_reason, summarize
from benchmark import MODEL, QUERY_PREFIX, atomic_json, load_data, read_jsonl

ROOT = Path(__file__).resolve().parent
GENERATOR = 'Qwen/Qwen2.5-3B-Instruct'
from generation_pilot import LABELS, CONDITIONS, SYSTEM, gold_label, parse_response, pack_context


def final_summary(rows):
    from generation_pilot import score_rows
    timings = {r['condition']:r for r in score_rows(rows)}
    return [dict(timings[r['condition']], **r) for r in summarize(rows, CONDITIONS)]


def evaluation_report(rows):
    import numpy as np
    qids = sorted({r['query_id'] for r in rows})
    lookup = {(r['query_id'], r['condition']):r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {(q,c) for q in qids for c in CONDITIONS}:
        raise ValueError('Report requires unique complete claim/condition pairs')
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, len(qids), size=(2000, len(qids)))
    scores = {c:np.array([lookup[q,c]['predicted_verdict']==lookup[q,c]['gold_verdict'] for q in qids], dtype=float) for c in CONDITIONS}
    report = {'claims':len(qids), 'bootstrap_seed':2026, 'bootstrap_draws':2000,
              'gold_class_counts':{label:sum(lookup[q,'flat']['gold_verdict']==label for q in qids) for label in LABELS},
              'accuracy_95_percentile_intervals':{}, 'paired_accuracy_difference_vs_flat':{}, 'confusion':{}}
    for c in CONDITIONS:
        report['accuracy_95_percentile_intervals'][c] = np.percentile(scores[c][draws].mean(axis=1), [2.5,97.5]).tolist()
        report['confusion'][c] = {gold:{pred:sum(lookup[q,c]['gold_verdict']==gold and lookup[q,c]['predicted_verdict']==pred for q in qids) for pred in (*LABELS,'INVALID')} for gold in LABELS}
        if c in ('hnsw','ivf_flat'):
            delta = scores[c]-scores['flat']
            report['paired_accuracy_difference_vs_flat'][c] = {'difference':float(delta.mean()), 'ci95':np.percentile(delta[draws].mean(axis=1), [2.5,97.5]).tolist()}
    return report


def verify_protocol():
    protocol = json.loads((ROOT/'docs/generation-final-protocol.json').read_text())
    for name, expected in protocol['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen protocol input changed: {name}')
    return protocol


def validate_partition(dev_ids, heldout_ids):
    if len(dev_ids) != 60 or len(set(dev_ids)) != 60:
        raise ValueError('Expected 60 unique development IDs')
    if len(heldout_ids) != 240 or len(set(heldout_ids)) != 240:
        raise ValueError('Expected 240 unique held-out IDs')
    if set(dev_ids) & set(heldout_ids):
        raise ValueError('Development/held-out overlap')


def prepare():
    verify_protocol()
    dev = json.loads((ROOT/'results/scifact-dev/results.json').read_text())
    heldout = json.loads((ROOT/'results/scifact-heldout/results.json').read_text())
    selected = heldout['metadata']['query_ids'].copy()
    validate_partition(dev['metadata']['query_ids'], selected)
    random.Random(2026).shuffle(selected)
    claims = {str(c['id']):c for c in read_jsonl(ROOT/'data/scifact-original/claims_dev.jsonl')}
    ids, docs, qids, queries, qrels = load_data(ROOT/'data/scifact','test')
    query_map = dict(zip(qids, queries))
    for qid in selected:
        if claims[qid]['claim'] != query_map[qid]: raise ValueError('Claim text mismatch')
        if not set(claims[qid]['evidence']) <= set(ids): raise ValueError('Missing evidence documents')
    key = hashlib.sha256(json.dumps([MODEL,dev['metadata']['revision'],ids,docs,qids,queries,QUERY_PREFIX],ensure_ascii=False).encode()).hexdigest()
    if key != dev['metadata']['cache_key']: raise ValueError('Embedding cache no longer matches data')
    return dev, claims, ids, dict(zip(ids,docs)), selected, ROOT/'data/cache'/f'{key}.npz'


def worker(args, out):
    import numpy as np
    import torch
    import faiss
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    dev, claims, ids, documents, selected, cache = prepare()
    revision = (ROOT/'data/generator-3b-revision.txt').read_text().strip()
    # Select only from the original development results, never the test metrics.
    from run_fiqa import select_settings
    selected_settings = select_settings(dev['summary'])
    files = ['generation_heldout.py','generation_comparison.py','docs/generation-final-protocol.json','generation_pilot.py','benchmark.py','run_fiqa.py','data/scifact-original/claims_dev.jsonl',
             'data/scifact/corpus.jsonl','data/scifact/queries.jsonl','data/scifact/qrels/test.tsv',
             'results/scifact-dev/results.json','results/scifact-heldout/results.json']
    manifest = {'generator':GENERATOR, 'generator_revision':revision, 'embedding_model':MODEL,
                'embedding_revision':dev['metadata']['revision'], 'selected_ids':selected,
                'labels':{qid:gold_label(claims[qid]) for qid in selected}, 'settings':selected_settings,
                'k':10,'nlist':64,'hnsw_m':16,'ef_construction':200,'construction_seed':42,'context_tokens':1536,'per_document_tokens':256,'max_new_tokens':256,'max_input_tokens':2048,
                'decoding':'greedy', 'threads':1,'dtype':'float32','seed':2026,'system_prompt':SYSTEM,
                'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'faiss':faiss.__version__,
                'python':sys.version, 'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
                'files':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest: raise ValueError('Held-out inputs changed; choose a new output directory')
    atomic_json(path,manifest)
    atomic_json(out/'status.json',dict(stage='loading_model',total=len(selected)*len(CONDITIONS),pid=os.getpid()))
    torch.set_num_threads(1); faiss.omp_set_num_threads(1); torch.manual_seed(2026)
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR,revision=revision,local_files_only=True,cache_dir='/tmp/ragproject-model-cache')
    generator = AutoModelForCausalLM.from_pretrained(GENERATOR,revision=revision,local_files_only=True,cache_dir='/tmp/ragproject-model-cache',dtype=torch.float32).eval()
    encoder = SentenceTransformer(MODEL,revision=dev['metadata']['revision'],device='cpu',local_files_only=True)
    with np.load(cache,allow_pickle=False) as saved: vectors = saved['vectors']
    indexes = {'flat':faiss.IndexFlatIP(vectors.shape[1]),
               'hnsw':faiss.IndexHNSWFlat(vectors.shape[1],16,faiss.METRIC_INNER_PRODUCT),
               'ivf_flat':faiss.IndexIVFFlat(faiss.IndexFlatIP(vectors.shape[1]),vectors.shape[1],64,faiss.METRIC_INNER_PRODUCT)}
    indexes['hnsw'].hnsw.efConstruction=200
    indexes['hnsw'].hnsw.rng=faiss.RandomGenerator(42)
    indexes['hnsw'].hnsw.efSearch=selected_settings['ef_search']
    indexes['ivf_flat'].cp.seed=42
    indexes['ivf_flat'].nprobe=selected_settings['nprobe']
    build_times={}
    for name,index in indexes.items():
        started=time.perf_counter(); index.train(vectors); index.add(vectors)
        build_times[name]=time.perf_counter()-started
    atomic_json(out/'index-build-times.json',build_times)
    # Warm-up is excluded; no study claims are used to warm the generator.
    warm = tokenizer('Return a short greeting.',return_tensors='pt')
    with torch.inference_mode(): generator.generate(**warm,max_new_tokens=4,do_sample=False)
    warm_query=encoder.encode([QUERY_PREFIX+'A warm-up query.'],normalize_embeddings=True,convert_to_numpy=True)
    for index in indexes.values(): index.search(warm_query,10)
    checkpoints=out/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    rows=[]
    for qid in selected:
        conditions=list(CONDITIONS); random.Random('2026:'+qid).shuffle(conditions)
        for condition in conditions:
            path=checkpoints/f'{qid}-{condition}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['query_id'] != qid or saved['condition'] != condition or saved['gold_verdict'] != gold_label(claims[qid]):
                    raise ValueError('Checkpoint identity mismatch')
                rows.append(saved); continue
            atomic_json(out/'status.json',{'stage':'running','query_id':qid,'condition':condition,'completed':len(rows),'total':len(selected)*len(CONDITIONS),'pid':os.getpid()})
            print(f'Generating {len(rows)+1}/{len(selected)*len(CONDITIONS)}: {qid} {condition}',flush=True)
            started=time.perf_counter()
            embed_ms=search_ms=0.0
            if condition in indexes:
                t=time.perf_counter(); query=encoder.encode([QUERY_PREFIX+claims[qid]['claim']],normalize_embeddings=True,convert_to_numpy=True)
                embed_ms=(time.perf_counter()-t)*1000
                t=time.perf_counter(); result=indexes[condition].search(query,10)[1][0]
                search_ms=(time.perf_counter()-t)*1000
                retrieved=[ids[i] for i in result if i>=0]
            elif condition=='gold_evidence': retrieved=sorted(claims[qid]['evidence'])
            else: retrieved=[]
            t=time.perf_counter()
            context,included,truncated=pack_context(tokenizer,retrieved,documents)
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':f"Claim: {claims[qid]['claim']}\n\nPassages:\n{context}"}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
            input_tokens=int(encoded['input_ids'].shape[1])
            if input_tokens>2048: raise ValueError('Prompt exceeds frozen input budget')
            packing_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            with torch.inference_mode(): output=generator.generate(**encoded,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            generation_ms=(time.perf_counter()-t)*1000
            generated=output[0,input_tokens:]
            text=tokenizer.decode(generated,skip_special_tokens=True)
            end_to_end_ms=(time.perf_counter()-started)*1000
            parsed=parse_response(text)
            citations=set(parsed['citations']) if parsed else set()
            gold_docs=set(claims[qid]['evidence'])
            row={'query_id':qid,'condition':condition,'claim':claims[qid]['claim'],'gold_verdict':gold_label(claims[qid]),
                 'predicted_verdict':parsed['verdict'] if parsed else 'INVALID','raw_output':text,'parsed':parsed,
                 'retrieved_ids':retrieved,'included_ids':included,'truncated_ids':truncated,'gold_evidence_ids':sorted(gold_docs),
                 'citations_in_context':sorted(citations & set(included)), 'citations_outside_context':sorted(citations-set(included)),
                 'citations_matching_gold_documents':sorted(citations & gold_docs), 'gold_document_in_context':bool(gold_docs & set(included)),
                 'input_tokens':input_tokens,'output_tokens':len(generated),'hit_output_limit':finish_reason(generated.tolist(),256,generator.generation_config.eos_token_id)=='length',
                 'query_embedding_ms':embed_ms,'search_ms':search_ms,'context_and_tokenization_ms':packing_ms,
                 'generation_ms':generation_ms,'end_to_end_ms':end_to_end_ms,'prompt':prompt,'supplied_passages':context,
                 'condition_order':conditions}
            normalized = normalized_response(text)
            nc = set(normalized['citations']) if normalized else set()
            row.update(normalized_parsed=normalized,
                       normalized_predicted_verdict=normalized['verdict'] if normalized else 'INVALID',
                       normalized_citations_outside_context=sorted(nc-set(included)),
                       normalized_citations_matching_gold_documents=sorted(nc & gold_docs),
                       max_new_tokens=256, reached_token_budget=len(generated)>=256,
                       finish_reason=finish_reason(generated.tolist(),256,generator.generation_config.eos_token_id))
            atomic_json(path,row); rows.append(row)
            atomic_json(out/'summary.json',final_summary(rows))
    summaries=final_summary(rows)
    atomic_json(out/'results.json',{'manifest':manifest,'summary':summaries,'queries':rows})
    atomic_json(out/'report.json',evaluation_report(rows))
    with (out/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    review=out/'manual-review.csv'
    if not review.exists():
        blinded=rows.copy(); random.Random(2026).shuffle(blinded)
        fields=['review_id','claim','supplied_passages','response','explanation_supported','citations_support_explanation','notes']
        with review.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for i,row in enumerate(blinded): writer.writerow({'review_id':i,'claim':row['claim'],'supplied_passages':row['supplied_passages'],'response':row['raw_output']})
        atomic_json(out/'review-key.json',{str(i):{'query_id':r['query_id'],'condition':r['condition'],'included_ids':r['included_ids']} for i,r in enumerate(blinded)})
    atomic_json(out/'status.json',{'stage':'complete','completed':len(rows),'total':len(rows)})
    print('Held-out evaluation complete.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/generation-heldout-3b-256')
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a') if args.lock_fd is not None else (out/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: parser.exit(1,'Held-out generation already running.\n')
    if args.background:
        env=dict(os.environ,HF_HOME=str(Path.home()/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        with (out/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--lock-fd',str(lock.fileno())],
                                   cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        (out/'run.pid').write_text(str(child.pid)+'\n'); print(f'Held-out generation PID {child.pid}; log {out / "run.log"}'); return
    try: worker(args,out)
    except Exception as exc:
        atomic_json(out/'status.json',{'stage':'failed','error':str(exc)}); raise
    finally: lock.close()


if __name__=='__main__': main()
