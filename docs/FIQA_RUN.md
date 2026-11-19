# FiQA background experiment

From the project directory, launch or resume:

```bash
.venv/bin/python run_fiqa.py --background
```

The launcher detaches from the terminal and redirects input/output, so closing an ordinary SSH session will not send it a terminal hangup. The machine must remain running; a reboot or a server policy that kills logged-out users' jobs can still stop it. Run the same command after an interruption to resume. A file lock prevents two copies of this runner from operating together.

Monitor progress:

```bash
tail -f results/fiqa-run/run.log
cat results/fiqa-run/status.json
```

Ctrl+C exits `tail` without stopping the benchmark. `status.json` reports the stage, last update, and an error if the worker fails. `run.pid` records the launched worker PID; status alone is not proof a process is still alive after a machine crash.

The run uses the model revision saved in `data/model-revision.txt`, the existing model cache at `~/.cache/huggingface`, and the extracted `data/fiqa` dataset. It works offline once these are present. Dependencies must be installed in `.venv`.

Stages:

1. Check the saved experiment manifest against code, dataset hashes, model revision, and installed package versions.
2. Encode FiQA documents and development queries, saving an atomic checkpoint every 512 texts. Completed chunks are reused after an interruption. Corpus chunks are also shared with the test stage.
3. Run development sweeps: HNSW efSearch 16/32/64/128/256; IVF nprobe 1/4/16/32/64/128, with nlist 128. Save partial results after each setting.
4. Freeze the fastest measured setting per approximate method with development ANN recall@10 >= 0.95. Stop with an error if no setting qualifies.
5. Evaluate only the selected settings and exact search on the official test split, then mark the run complete.

All runs use k=10, seed 42, three timing repeats, and single-threaded FAISS. Query encoding and index search are separate. A restart during index construction/search repeats that stage; partial search results are inspection artifacts, not resumable index checkpoints. Embedding time on a restarted run covers only that invocation, not cumulative encoding time.

Outputs are in `results/fiqa-run/`: `dev/summary.csv`, `test/summary.csv`, detailed JSON results, `selected-settings.json`, `manifest.json`, and `requirements.txt`. `status.json` will say `complete` when both stages finish. These are single-build pilot measurements, without confidence intervals.

Keep data/cache and the run folder to resume. A completed stage is reused. If code, data, packages, or the model revision change, archive the run folder before starting a new experiment; the runner refuses to mix incompatible stages. Results and caches are Git-ignored, so copy them to persistent backup storage when finished.
