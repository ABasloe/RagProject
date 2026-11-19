# Analysis of the completed SciFact and FiQA runs

`analyze_benchmarks.py` reads the existing development and test results, uses the frozen test configurations, and produces `results/analysis/report.md`, raw JSON measurements, and PNG/PDF plots. It does not retune settings or overwrite the original runs.

Install plotting dependencies with `.venv/bin/python -m pip install -r requirements-analysis.txt` if needed. Launch or resume from this directory:

```bash
mkdir -p results/analysis
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/results/analysis/matplotlib-cache"
export TOKENIZERS_PARALLELISM=false
nohup .venv/bin/python -u analyze_benchmarks.py >> results/analysis/run.log 2>&1 < /dev/null &
echo $! > results/analysis/run.pid
```

Monitor with `tail -f results/analysis/run.log` or `cat results/analysis/status.json`. A file lock prevents concurrent copies. Each dataset audit and each completed seed run is checkpointed atomically. Restarting repeats only incomplete builds/audits; bootstrap calculations and plots are regenerated. The manifest rejects changes to analysis code, original result files, benchmark code, or the recorded library versions. Archive the analysis directory to start a changed protocol. Keep the machine running; background execution does not survive reboot automatically.

The study includes:

- 5,000 query-bootstrap draws and 95% percentile intervals for ANN recall, evidence recall, and NDCG, plus paired differences from exact search. Paired results are aligned by query ID. These intervals describe query sampling conditional on the original index build.
- Five separate builds at seeds 101, 202, 303, 404, 505, with HNSW RNG and IVF training seeds explicitly set. Corpus order and selected settings stay fixed. Query order and method timing order vary per seed, with the same query order for all methods within a run. Three timings per query are retained. Cross-run ranges are descriptive and are not confidence intervals.
- Document and evaluated-test-query token lengths before truncation, using the frozen model tokenizer, input preprocessing, query prefix, and model sequence limit. Token-position excess is not an exact count of omitted natural-language words.
- Development recall/latency and recall/index-size plots, test paired quality intervals, and repeated-run latency plots. PNG and PDF versions are saved.

No new embedding computation is needed. The dataset-content fingerprint must match the original cache for repeated builds. The full corpus is retained for both datasets. Serialized index size is not resident memory; peak-memory measurement, controlled scaling subsets, and downstream answer evaluation remain separate future experiments.

Original FiQA run manifests include installed packages; adding Matplotlib changes that environment fingerprint. Use this analysis runner for this phase rather than relaunching `run_fiqa.py` against its completed output folder.
