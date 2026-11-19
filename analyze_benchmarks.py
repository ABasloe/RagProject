"""Checkpointed uncertainty, truncation, and repeated-build study of frozen settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from benchmark import MODEL, QUERY_PREFIX, atomic_json, load_data, metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/analysis'
SOURCES = {
    'scifact': ('results/scifact-dev/results.json', 'results/scifact-heldout/results.json'),
    'fiqa': ('results/fiqa-run/dev/results.json', 'results/fiqa-run/test/results.json'),
}
METRICS = ('ann_recall', 'evidence_recall', 'ndcg')
SEEDS = (101, 202, 303, 404, 505)


def bootstrap(values, draws=5000, seed=2026):
    """Percentile interval for a query mean; accepts paired differences directly."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 100):
        indices = rng.integers(0, len(values), size=(min(100, draws-start), len(values)))
        means[start:start+len(indices)] = values[indices].mean(axis=1)
    return {'mean': float(values.mean()), 'low': float(np.quantile(means, .025)),
            'high': float(np.quantile(means, .975))}


def uncertainty(result):
    groups = {}
    for row in result['queries']:
        groups.setdefault(row['method'], {})[row['query_id']] = row
    qids = sorted(groups['flat'])
    output = {}
    for method, rows in groups.items():
        if set(rows) != set(qids):
            raise ValueError('Unpaired query results')
        output[method] = {}
        for metric in METRICS:
            values = np.array([rows[q][metric] for q in qids])
            exact = np.array([groups['flat'][q][metric] for q in qids])
            output[method][metric] = bootstrap(values)
            output[method][metric + '_minus_exact'] = bootstrap(values - exact)
    return output


def truncation(tokenizer, texts, limit):
    lengths = []
    for start in range(0, len(texts), 256):
        # SentenceTransformer's Transformer module strips and lowercases for this model.
        batch = [text.strip().lower() for text in texts[start:start+256]]
        encoded = tokenizer(batch, truncation=False, padding=False, add_special_tokens=True,
                            return_attention_mask=False, return_token_type_ids=False)
        lengths.extend(map(len, encoded['input_ids']))
    lengths = np.array(lengths)
    excess = np.maximum(lengths-limit, 0)
    return {'count': len(texts), 'max_seq_length': limit, 'over_limit_count': int((excess > 0).sum()),
            'over_limit_fraction': float((excess > 0).mean()), 'median_tokens': float(np.median(lengths)),
            'p95_tokens': float(np.percentile(lengths, 95)), 'max_tokens': int(lengths.max()),
            'tokens_beyond_limit': int(excess.sum()), 'fraction_tokens_beyond_limit': float(excess.sum()/lengths.sum()),
            'note': 'Lengths include special tokens; excess measures positions beyond the sequence budget, not a reconstruction of omitted content.'}


def repeated_build(result, seed):
    import faiss
    faiss.omp_set_num_threads(1)
    meta = result['metadata']
    dataset = ROOT / 'data' / Path(meta['dataset']).name
    ids, docs, all_qids, queries, qrels = load_data(dataset, meta['split'])
    cache_key = hashlib.sha256(json.dumps([MODEL, meta['revision'], ids, docs, all_qids, queries, QUERY_PREFIX], ensure_ascii=False).encode()).hexdigest()
    if cache_key != meta['cache_key']:
        raise ValueError('Dataset differs from the original embedding cache')
    with np.load(ROOT / 'data/cache' / (cache_key + '.npz')) as saved:
        vectors, all_queries = saved['vectors'], saved['queries']
    positions = {qid: i for i, qid in enumerate(all_qids)}
    qids = meta['query_ids']
    qvectors = np.ascontiguousarray(all_queries[[positions[q] for q in qids]])
    dimension = vectors.shape[1]
    exact = faiss.IndexFlatIP(dimension)
    exact.add(vectors)
    truth = exact.search(qvectors, meta['k'])[1]
    del exact
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(qids))
    configs = list(result['summary'])
    rng.shuffle(configs)
    summaries, per_query = [], []
    for config in configs:
        name, parameter = config['method'], config['parameter']
        if name == 'flat':
            index = faiss.IndexFlatIP(dimension)
        elif name == 'hnsw':
            index = faiss.IndexHNSWFlat(dimension, meta['hnsw_m'], faiss.METRIC_INNER_PRODUCT)
            index.hnsw.rng = faiss.RandomGenerator(seed)
            index.hnsw.efConstruction = meta['ef_construction']
            index.hnsw.efSearch = parameter
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, meta['nlist'], faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = seed
            index.nprobe = parameter
        started = time.perf_counter()
        index.train(vectors)
        train = time.perf_counter() - started
        started = time.perf_counter()
        index.add(vectors)
        add = time.perf_counter() - started
        size = len(faiss.serialize_index(index))
        index.search(qvectors[order[:min(10, len(order))]], meta['k'])
        times, scores = [], []
        for qi in order:
            query = qvectors[qi:qi+1]
            query_times = []
            for _ in range(meta['repeats']):
                started = time.perf_counter_ns()
                found = index.search(query, meta['k'])[1][0]
                query_times.append((time.perf_counter_ns()-started)/1e6)
            found_ids = [ids[i] if i >= 0 else None for i in found]
            values = metrics(found_ids, [ids[i] for i in truth[qi]], qrels[qids[qi]], meta['k'])
            scores.append(values)
            times.extend(query_times)
            per_query.append({'method': name, 'query_id': qids[qi], 'latency_ms': query_times, **dict(zip(METRICS, values))})
        summaries.append({'method': name, 'parameter': parameter, 'train_seconds': train, 'add_seconds': add,
                          'build_seconds': train+add, 'serialized_bytes': size, 'median_ms': float(np.median(times)),
                          'p95_ms': float(np.percentile(times,95)), **dict(zip(METRICS, np.mean(scores,axis=0)))})
        del index
    return {'seed': seed, 'query_order': [qids[i] for i in order], 'method_order': [c['method'] for c in configs],
            'summary': summaries, 'queries': per_query}


def plots(data, intervals, builds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'flat': '#555555', 'hnsw': '#0072B2', 'ivf_flat': '#D55E00'}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (dataset, (dev, test)) in enumerate(data.items()):
        for method in colors:
            sweep = [s for s in dev['summary'] if s['method'] == method]
            for col, metric in enumerate(['median_ms', 'p95_ms']):
                axes[row,col].plot([s[metric] for s in sweep], [s['ann_recall'] for s in sweep], 'o-', color=colors[method], label=method)
                axes[row,col].set_xscale('log')
                axes[row,col].set_xlabel(metric.replace('_',' ') + ' (log scale)')
                axes[row,col].set_ylabel('ANN recall@10')
                axes[row,col].set_title(dataset + ' development sweep')
            axes[row,2].scatter([s['serialized_bytes']/1e6 for s in sweep], [s['ann_recall'] for s in sweep], color=colors[method], label=method)
        axes[row,2].set_xlabel('Serialized index MB (decimal)')
        axes[row,2].set_ylabel('ANN recall@10')
        axes[row,2].set_title(dataset + ' development size/recall')
        for ax in axes[row]:
            ax.legend(); ax.grid(alpha=.2)
    for ext in ['png','pdf']:
        fig.savefig(OUT / ('development-tradeoffs.' + ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, dataset in enumerate(data):
        for i, method in enumerate(['hnsw','ivf_flat']):
            ci = intervals[dataset][method]['evidence_recall_minus_exact']
            axes[row,0].errorbar(i, 100*ci['mean'], yerr=[[100*(ci['mean']-ci['low'])],[100*(ci['high']-ci['mean'])]], fmt='o', color=colors[method], capsize=5)
        axes[row,0].set_xticks([0,1], ['hnsw','ivf_flat'])
        axes[row,0].axhline(0, color='gray', linestyle='--')
        axes[row,0].set_ylabel('Evidence recall difference (percentage points)')
        axes[row,0].set_title(dataset + ': paired query bootstrap 95% CI')
        for i, method in enumerate(colors):
            values = [next(s for s in b['summary'] if s['method']==method)['median_ms'] for b in builds[dataset]]
            axes[row,1].scatter([i]*len(values), values, color=colors[method])
        axes[row,1].set_xticks(range(3), list(colors))
        axes[row,1].set_yscale('log')
        axes[row,1].set_ylabel('Median search latency (ms, log scale)')
        axes[row,1].set_title(dataset + ': five independent builds/runs')
    for ext in ['png','pdf']:
        fig.savefig(OUT / ('test-uncertainty.' + ext), dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (OUT / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Analysis is already running')
    data = {name: tuple(json.loads((ROOT/p).read_text()) for p in paths) for name, paths in SOURCES.items()}
    import faiss
    import transformers
    manifest = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'benchmark_sha256': hashlib.sha256((ROOT/'benchmark.py').read_bytes()).hexdigest(),
                'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for paths in SOURCES.values() for p in paths},
                'seeds': SEEDS, 'bootstrap_draws': 5000, 'numpy': np.__version__, 'faiss': faiss.__version__,
                'transformers': transformers.__version__, 'python': sys.version, 'platform': platform.platform()}
    manifest = json.loads(json.dumps(manifest))
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text()) != manifest:
        raise RuntimeError('Analysis inputs changed; archive results/analysis before rerunning')
    atomic_json(OUT/'manifest.json', manifest)
    intervals, audits, builds = {}, {}, {}
    for dataset, (dev, test) in data.items():
        atomic_json(OUT/'status.json', {'stage':'running', 'dataset':dataset, 'task':'bootstrap_and_audit', 'pid':os.getpid()})
        print(f'{dataset}: bootstrap intervals', flush=True)
        intervals[dataset] = uncertainty(test)
        atomic_json(OUT/'confidence-intervals.json', intervals)
        audit_path = OUT/f'{dataset}-truncation.json'
        if not audit_path.exists():
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL, revision=test['metadata']['revision'], device='cpu', local_files_only=True)
            tokenizer, limit = model.tokenizer, model.max_seq_length
            assert model[0].do_lower_case, 'Audit preprocessing must be updated for this model'
            _, docs, qids, queries, _ = load_data(ROOT/'data'/dataset, test['metadata']['split'])
            test_ids = set(test['metadata']['query_ids'])
            print(f'{dataset}: truncation audit at {limit} tokens', flush=True)
            audit = {'documents': truncation(tokenizer, docs, limit),
                     'test_queries': truncation(tokenizer, [QUERY_PREFIX+q for qid,q in zip(qids, queries) if qid in test_ids], limit)}
            atomic_json(audit_path, audit)
            del model
        audits[dataset] = json.loads(audit_path.read_text())
        builds[dataset] = []
        for seed in SEEDS:
            path = OUT/f'{dataset}-seed-{seed}.json'
            if not path.exists():
                atomic_json(OUT/'status.json', {'stage':'running', 'dataset':dataset, 'task':'repeated_build', 'seed':seed, 'pid':os.getpid()})
                print(f'{dataset}: independent build seed {seed}', flush=True)
                atomic_json(path, repeated_build(test, seed))
            builds[dataset].append(json.loads(path.read_text()))
    plots(data, intervals, builds)
    report(data, intervals, audits, builds)
    atomic_json(OUT/'status.json', {'stage':'complete'})
    print('Complete: results/analysis/report.md', flush=True)


def report(data, intervals, audits, builds):
    lines = ['# Retrieval benchmark analysis', '', 'Frozen settings from development selection; original test results are preserved.', '',
             '95% percentile bootstrap intervals use 5,000 query resamples (seed 2026). Differences are paired by query ID. These intervals condition on the original index build and do not measure construction uncertainty. No multiple-comparison correction is applied.', '']
    for dataset, (_, test) in data.items():
        lines += [f'## {dataset}', '', '| Method | ANN recall (95% CI) | Evidence recall (95% CI) | NDCG (95% CI) | Evidence difference vs exact, pp (95% CI) |', '|---|---|---|---|---|']
        for method in ('flat','hnsw','ivf_flat'):
            values = intervals[dataset][method]
            cells = []
            for key in (*METRICS,'evidence_recall_minus_exact'):
                x = values[key]; factor = 100 if key != 'ndcg' else 1
                cells.append(f"{x['mean']*factor:.3f} [{x['low']*factor:.3f}, {x['high']*factor:.3f}]")
            lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Recall values above are percentages; NDCG is on a 0–1 scale.', '',
                  '| Method | Median latency range, ms | p95 latency range, ms | Build time range, s | ANN recall range | Serialized MB |', '|---|---|---|---|---|---|']
        for method in ('flat','hnsw','ivf_flat'):
            rows = [next(s for s in b['summary'] if s['method']==method) for b in builds[dataset]]
            cells = [f"{min(s[k] for s in rows):.4f}–{max(s[k] for s in rows):.4f}" for k in ('median_ms','p95_ms','build_seconds','ann_recall')]
            cells.append(f"{rows[0]['serialized_bytes']/1e6:.2f}")
            lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
        audit = audits[dataset]['documents']
        lines += ['', f"Documents exceeding {audit['max_seq_length']} tokens: {audit['over_limit_count']}/{audit['count']} ({100*audit['over_limit_fraction']:.2f}%). Fraction of token positions beyond the budget: {100*audit['fraction_tokens_beyond_limit']:.2f}%.", '']
    lines += ['## Interpretation limits', '', 'Five independent builds use seeds 101/202/303/404/505 for HNSW and IVF construction. Corpus insertion order is fixed; query order and method measurement order vary by seed. Each method uses the same query order within a run. Timings exclude embeddings and generation. Seed variation and scheduling variation are not separately estimated. Ranges across five runs are descriptive, not confidence intervals.', '',
              'Truncation audit uses the frozen model tokenizer, lowercasing, whitespace stripping, query prefix, and actual maximum sequence length. It does not change the existing embeddings. Serialized bytes are not peak resident memory. No scaling, IVF-PQ, additional datasets, or answer-generation study is included in this analysis.', '']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            atomic_json(OUT/'status.json', {'stage':'failed', 'error':str(exc)})
        raise
