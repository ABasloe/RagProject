"""Controlled FiQA corpus scaling with frozen queries and checkpointed stages."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import faiss
import numpy as np
from benchmark import MODEL, QUERY_PREFIX, atomic_json, load_data, metrics
from run_fiqa import select_settings

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/fiqa-scaling'
SIZES = (5000, 20000, 57638)
BUILD_SEEDS = (101, 202, 303, 404, 505)
EF_SEARCH = (16, 32, 64, 128, 256, 512)
NPROBE = (1, 4, 16, 32, 64, 128)


def status(stage, **details):
    value = {'stage': stage, 'pid': os.getpid(), 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details}
    atomic_json(OUT/'status.json', value)
    print(json.dumps(value), flush=True)


def nested_subsets(ids, required, sizes, seed):
    """Keep relevant documents and add a single seeded order of distractors."""
    if len(ids) != len(set(ids)) or not required.issubset(set(ids)):
        raise ValueError('Duplicate corpus IDs or missing relevant documents')
    if sorted(set(sizes)) != list(sizes) or sizes[0] < len(required) or sizes[-1] > len(ids):
        raise ValueError('Sizes must increase and accommodate all relevant documents')
    rng = np.random.default_rng(seed)
    remaining = [doc for doc in ids if doc not in required]
    rng.shuffle(remaining)
    subsets = {}
    for size in sizes:
        selected = required | set(remaining[:size-len(required)])
        # Preserve original insertion order, including the full-corpus condition.
        subsets[size] = [i for i, doc in enumerate(ids) if doc in selected]
    return subsets


def load_inputs():
    inputs, hashes = {}, {}
    corpus_vectors = None
    for split in ('dev', 'test'):
        source = ROOT/f'results/fiqa-run/{split}/results.json'
        old = json.loads(source.read_text())['metadata']
        ids, docs, qids, queries, qrels = load_data(ROOT/'data/fiqa', split)
        key = hashlib.sha256(json.dumps([MODEL, old['revision'], ids, docs, qids, queries, QUERY_PREFIX], ensure_ascii=False).encode()).hexdigest()
        if key != old['cache_key'] or old['query_ids'] != qids:
            raise ValueError('Data/query IDs differ from original FiQA embedding run')
        cache = ROOT/'data/cache'/f'{key}.npz'
        hashes[str(cache.relative_to(ROOT))] = hashlib.sha256(cache.read_bytes()).hexdigest()
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(cache, allow_pickle=False) as saved:
            if corpus_vectors is None:
                corpus_vectors = saved['vectors']
                corpus_ids = ids
                revision = old['revision']
            elif ids != corpus_ids or revision != old['revision'] or not np.array_equal(corpus_vectors, saved['vectors']):
                raise ValueError('Dev/test corpus vectors or revisions differ')
            qvectors = saved['queries']
        if len(qvectors) != len(qids) or len(corpus_vectors) != len(ids):
            raise ValueError('Embedding shape mismatch')
        inputs[split] = {'qids': qids, 'qrels': qrels, 'vectors': np.ascontiguousarray(qvectors)}
    if set(inputs['dev']['qids']) & set(inputs['test']['qids']):
        raise ValueError('Development/test overlap')
    return corpus_ids, corpus_vectors, inputs, revision, hashes


def build_index(method, vectors, seed):
    dimension = vectors.shape[1]
    if method == 'flat':
        index = faiss.IndexFlatIP(dimension)
    elif method == 'hnsw':
        index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.rng = faiss.RandomGenerator(seed)
    else:
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, 128, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = seed
    started = time.perf_counter()
    index.train(vectors)
    training = time.perf_counter()-started
    started = time.perf_counter()
    index.add(vectors)
    adding = time.perf_counter()-started
    return index, training, adding


def evaluate(index, method, parameter, ids, query_set, truth, order, train, add):
    if method == 'hnsw':
        index.hnsw.efSearch = parameter
    elif method == 'ivf_flat':
        index.nprobe = parameter
    index.search(query_set['vectors'][order[:10]], 10)
    rows, times = [], []
    for qi in order:
        query = query_set['vectors'][qi:qi+1]
        samples = []
        for _ in range(3):
            started = time.perf_counter_ns()
            found = index.search(query, 10)[1][0]
            samples.append((time.perf_counter_ns()-started)/1e6)
        found_ids = [ids[i] if i >= 0 else None for i in found]
        exact_ids = [ids[i] for i in truth[qi]]
        values = metrics(found_ids, exact_ids, query_set['qrels'][query_set['qids'][qi]], 10)
        rows.append({'query_id': query_set['qids'][qi], 'ann_recall': values[0], 'evidence_recall': values[1],
                     'ndcg': values[2], 'latency_ms': samples, 'result_ids': found_ids, 'exact_ids': exact_ids})
        times.extend(samples)
    summary = {'method': method, 'parameter': parameter, 'train_seconds': train, 'add_seconds': add,
               'build_seconds': train+add, 'serialized_bytes': len(faiss.serialize_index(index)),
               'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times,95))}
    summary.update({key: float(np.mean([r[key] for r in rows])) for key in ('ann_recall','evidence_recall','ndcg')})
    return {'summary': summary, 'queries': rows}


def stage(size, split, seed, ids, vectors, query_set, settings=None):
    folder = OUT/str(size)/split/str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    configurations = {'flat':[None], 'hnsw':list(EF_SEARCH) if settings is None else [settings['ef_search']],
                      'ivf_flat':list(NPROBE) if settings is None else [settings['nprobe']]}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(query_set['qids']))
    methods = list(configurations)
    rng.shuffle(methods)
    # Exact neighbors are recomputed for each corpus size and query split.
    truth_index = faiss.IndexFlatIP(vectors.shape[1])
    truth_index.add(vectors)
    truth = truth_index.search(query_set['vectors'], 10)[1]
    del truth_index
    summaries = []
    for method in methods:
        parameters = configurations[method]
        rng.shuffle(parameters)
        index = None
        for parameter in parameters:
            path = folder/f'{method}-{parameter}.json'
            if not path.exists():
                status('running', corpus_size=size, split=split, seed=seed, method=method, parameter=parameter)
                if index is None:
                    index, training, adding = build_index(method, vectors, seed)
                result = evaluate(index, method, parameter, ids, query_set, truth, order, training, adding)
                result['metadata'] = {'corpus_size':size, 'split':split, 'seed':seed, 'query_order':[query_set['qids'][i] for i in order],
                                      'method_order':methods, 'parameter_order':parameters, 'k':10, 'nlist':128, 'threads':1, 'repeats':3}
                atomic_json(path, result)
            summaries.append(json.loads(path.read_text())['summary'])
        del index
    return summaries


def outputs(test_rows):
    with (OUT/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader(); writer.writerows(test_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    measures = [('median_ms','Median latency (ms)'),('p95_ms','p95 latency (ms)'),('build_seconds','Build time (s)'),
                ('serialized_bytes','Serialized index MB'),('ann_recall','ANN recall@10'),('evidence_recall','Evidence recall@10')]
    for ax, (key,label) in zip(axes.flat, measures):
        for method in ('flat','hnsw','ivf_flat'):
            points = [[r[key] for r in test_rows if r['corpus_size']==size and r['method']==method] for size in SIZES]
            factor = 1e6 if key=='serialized_bytes' else 1
            means=np.array([np.mean(p)/factor for p in points])
            lows=np.array([min(p)/factor for p in points]); highs=np.array([max(p)/factor for p in points])
            ax.errorbar(SIZES, means, yerr=[means-lows,highs-means], marker='o', capsize=3, label=method)
        ax.set_xlabel('Corpus documents'); ax.set_ylabel(label); ax.set_xscale('log')
        if key in ('median_ms','p95_ms','build_seconds'): ax.set_yscale('log')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('FiQA controlled scaling: means and observed ranges across five builds')
    for extension in ('png','pdf'): fig.savefig(OUT/f'scaling.{extension}',dpi=160)
    plt.close(fig)
    lines=['# FiQA controlled scaling', '', 'Nested relevance-preserving subsets; the same 500 development and 648 test queries at every size. All settings were selected on development data before any test evaluation.', '',
           '| Documents | Method | Search parameter | Median latency mean (range), ms | ANN recall mean | Evidence recall mean | Build time mean, s |', '|---|---|---|---|---|---|---|']
    for size in SIZES:
        for method in ('flat','hnsw','ivf_flat'):
            rows=[r for r in test_rows if r['corpus_size']==size and r['method']==method]
            values=[r['median_ms'] for r in rows]
            lines.append(f"| {size} | {method} | {rows[0]['parameter']} | {np.mean(values):.4f} ({min(values):.4f}–{max(values):.4f}) | {np.mean([r['ann_recall'] for r in rows]):.4f} | {np.mean([r['evidence_recall'] for r in rows]):.4f} | {np.mean([r['build_seconds'] for r in rows]):.4f} |")
    lines += ['', 'The search parameter is efSearch for HNSW and nprobe for IVF-Flat. k=10, nlist=128, HNSW M=16 and efConstruction=200 remain fixed. Corpus order follows the original document order. Search effort is tuned separately at each size to target development ANN recall >=0.95; test recall is measured, not guaranteed.', '',
              'Error bars are observed ranges across construction seeds 101/202/303/404/505, not confidence intervals. The same query order is used across methods within each seed; method/parameter timing order is randomized. Training plus insertion is build time. Embedding and generation are excluded.', '',
              'All 2,944 documents judged relevant to either query split are retained before random distractors are added with subset seed 2026. Test judgments are used solely for predefined corpus membership, never parameter selection. These small corpora are relevance-enriched and not random samples of FiQA. Only one nested subset sequence is tested; subset-sampling uncertainty is not estimated. Smaller corpora have fewer distractors, so absolute evidence metrics across sizes describe different retrieval problems. Full-corpus results remain the main relevance comparison.', '']
    (OUT/'report.md').write_text('\n'.join(lines))


def worker():
    faiss.omp_set_num_threads(1)
    ids, vectors, inputs, revision, fingerprints = load_inputs()
    for filename in ('run_scaling.py','benchmark.py','run_fiqa.py','data/fiqa/corpus.jsonl','data/fiqa/queries.jsonl','data/fiqa/qrels/dev.tsv','data/fiqa/qrels/test.tsv'):
        fingerprints[filename]=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    manifest = {'fingerprints':fingerprints, 'revision':revision, 'model':MODEL, 'sizes':list(SIZES), 'subset_seed':2026,
                'dev_seed':42, 'build_seeds':list(BUILD_SEEDS), 'ef_search':list(EF_SEARCH), 'nprobe':list(NPROBE),
                'nlist':128, 'hnsw_m':16, 'ef_construction':200, 'k':10, 'repeats':3, 'target':.95,
                'faiss':faiss.__version__, 'numpy':np.__version__, 'python':sys.version, 'platform':platform.platform()}
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Scaling code, inputs, or environment changed; archive results/fiqa-scaling before a new experiment')
    atomic_json(path,manifest)
    required={doc for split in inputs.values() for rels in split['qrels'].values() for doc in rels}
    subsets=nested_subsets(ids, required, SIZES, 2026)
    atomic_json(OUT/'subsets.json', {'required_document_ids':sorted(required), 'corpus_ids':{str(n):[ids[i] for i in positions] for n,positions in subsets.items()},
                                  'query_ids':{split:values['qids'] for split,values in inputs.items()}})
    selections={}
    # Freeze all development selections before evaluating any test condition.
    for size, positions in subsets.items():
        summaries=stage(size,'dev',42,[ids[i] for i in positions],np.ascontiguousarray(vectors[positions]),inputs['dev'])
        selections[str(size)]=select_settings(summaries)
    atomic_json(OUT/'selected-settings.json',selections)
    test_rows=[]
    for seed in BUILD_SEEDS:
        size_order=list(SIZES)
        np.random.default_rng(seed).shuffle(size_order)
        for size in size_order:
            positions=subsets[size]
            summaries=stage(size,'test',seed,[ids[i] for i in positions],np.ascontiguousarray(vectors[positions]),inputs['test'],selections[str(size)])
            test_rows.extend({'corpus_size':size,'seed':seed,**row} for row in summaries)
    outputs(test_rows)
    status('complete', summary='results/fiqa-scaling/report.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,'a') if args.lock_fd is not None else (OUT/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: parser.exit(1,'Scaling is already running.\n')
    if args.background:
        env=dict(os.environ,OMP_NUM_THREADS='1',MPLCONFIGDIR=str(OUT/'matplotlib-cache'))
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--lock-fd',str(lock.fileno())],cwd=ROOT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                   pass_fds=(lock.fileno(),),env=env)
        (OUT/'run.pid').write_text(str(child.pid)+'\n')
        print(f'Scaling launched: PID {child.pid}; log {OUT / "run.log"}')
        return
    try:
        status('starting')
        worker()
    except Exception as exc:
        status('failed',error=str(exc))
        raise
    finally: lock.close()


if __name__=='__main__': main()
