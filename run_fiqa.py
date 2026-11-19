"""Run FiQA development selection and test evaluation with resumable stages."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark import atomic_json

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results' / 'fiqa-run'


def status(stage, **extra):
    atomic_json(RUN / 'status.json', {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid(), **extra})
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {stage}', flush=True)


def select_settings(summary, target=0.95):
    selected = {}
    for method, key in [('hnsw', 'ef_search'), ('ivf_flat', 'nprobe')]:
        eligible = [row for row in summary if row['method'] == method and row['ann_recall'] >= target]
        if not eligible:
            raise RuntimeError(f'{method} did not reach {target} development ANN recall; test evaluation has not started')
        selected[key] = min(eligible, key=lambda row: row['median_ms'])['parameter']
    return selected


def worker():
    revision = (ROOT / 'data/model-revision.txt').read_text().strip()
    paths = [ROOT / 'benchmark.py', Path(__file__), ROOT / 'data/fiqa/corpus.jsonl',
             ROOT / 'data/fiqa/queries.jsonl', ROOT / 'data/fiqa/qrels/dev.tsv', ROOT / 'data/fiqa/qrels/test.tsv']
    fingerprint = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    config = {'revision': revision, 'inputs': fingerprint, 'packages': packages, 'nlist': 128,
              'seed': 42, 'k': 10, 'repeats': 3, 'target_ann_recall': 0.95,
              'ef_search': [16, 32, 64, 128, 256], 'nprobe': [1, 4, 16, 32, 64, 128]}
    manifest = RUN / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError('Code, data, model revision, or packages changed. Archive results/fiqa-run before starting a new experiment.')
    atomic_json(manifest, config)
    (RUN / 'requirements.txt').write_text(packages)
    env = dict(os.environ, HF_HOME=str(Path.home() / '.cache/huggingface'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')

    def benchmark(split, parameters):
        output = RUN / split
        if not (output / 'results.json').exists():
            status(f'{split}_running')
            command = [sys.executable, '-u', str(ROOT / 'benchmark.py'), str(ROOT / 'data/fiqa'),
                       '--revision', revision, '--split', split, '--seed', '42', '--nlist', '128',
                       '--output', str(output), '--cache', str(ROOT / 'data/cache'), *parameters]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        else:
            print(f'Reusing completed {split} stage', flush=True)
        return json.loads((output / 'results.json').read_text())

    dev = benchmark('dev', ['--ef-search', *map(str, config['ef_search']), '--nprobe', *map(str, config['nprobe'])])
    selection = select_settings(dev['summary'])
    atomic_json(RUN / 'selected-settings.json', {'criterion': 'Minimum development median latency at ANN recall@10 >= 0.95', **selection})
    status('settings_selected', **selection)
    test = benchmark('test', ['--ef-search', str(selection['ef_search']), '--nprobe', str(selection['nprobe'])])
    assert not set(dev['metadata']['query_ids']) & set(test['metadata']['query_ids']), 'Development/test query overlap'
    status('complete', selected_settings=selection, test_summary=test['summary'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', action='store_true', help='Detach and append output to results/fiqa-run/run.log')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, 'a') if args.lock_fd is not None else (RUN / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, 'FiQA is already running; see results/fiqa-run/run.log\n')
    if args.background:
        with (RUN / 'run.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--lock-fd', str(lock.fileno())],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        (RUN / 'run.pid').write_text(str(process.pid) + '\n')
        print(f'FiQA launched with PID {process.pid}. Log: {RUN / "run.log"}')
        return
    try:
        status('starting')
        worker()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
