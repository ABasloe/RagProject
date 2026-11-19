# Search algorithms in retrieval-augmented generation

This project compares exact vector search, HNSW, and IVF-Flat under fixed embedding and dataset conditions. It studies retrieval latency, quality, index construction, corpus scaling, and the effect of retrieved context on scientific claim verification.

The retrieval and scaling experiments on SciFact and FiQA are complete. A frozen, checkpointed SciFact generation evaluation is running; see its [protocol](docs/GENERATION_FINAL_PROTOCOL.md). The [project overview presentation](presentation/RAG_Project_Overview.pdf) is a dated snapshot, not a live status display.

## Repository map

| Path | Purpose |
| --- | --- |
| `benchmark.py` | BEIR loading, embeddings, FAISS retrieval sweeps and metrics |
| `run_fiqa.py` | Resumable FiQA development selection and test run |
| `run_scaling.py` | Controlled nested FiQA corpus-size experiment |
| `analyze_benchmarks.py` | Query bootstrap, repeated builds, truncation audit and plots |
| `generation_pilot.py` / `generation_comparison.py` | Development generation pilots and model comparisons |
| `generation_heldout.py` | Frozen held-out claim-verification experiment |
| `test_*.py` | Unit and small FAISS integration tests |
| `docs/` | Experiment protocols, review notes, project plan and sources |
| `presentation/` | Editable slides, PDF, speaker notes and slide-generation source |
| `data/`, `results/` | Local inputs, caches and outputs; ignored by Git |

## Setup

Use Python 3.11 or 3.12 on Linux; WSL may be useful on Windows. Install CPU PyTorch first to avoid pulling GPU libraries:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest -v
```

For the analysis plots, install `python -m pip install -r requirements-analysis.txt`. Dataset and model downloads need network access and sufficient storage. See the [BEIR dataset list](https://github.com/beir-cellar/beir#datasets) and each dataset's terms.

## Reproduce retrieval experiments

Extract a BEIR dataset under `data/<dataset>/`, with `corpus.jsonl`, `queries.jsonl`, and `qrels/<split>.tsv`. Record an immutable 40-character commit for `BAAI/bge-small-en-v1.5` and substitute it below:

```bash
python benchmark.py data/scifact --revision MODEL_COMMIT --partition dev --seed 42 --nlist 64 --output results/scifact-dev
```

Select search settings using development queries, then run held-out queries with the same seed and fraction. [FiQA instructions](docs/FIQA_RUN.md) cover its official development/test splits. The [scaling protocol](docs/SCALING_RUN.md) and [analysis instructions](ANALYSIS.md) describe the additional experiments. Existing result files are protected against overwriting.

The benchmark uses normalized vectors and k=10 by default. ANN recall measures agreement with exact vector search, while evidence recall and NDCG use dataset judgments. Search timing excludes embedding and generation. Serialized index size is not peak RAM. Inputs longer than the model limit are truncated, and unjudged documents are treated as nonrelevant for reported metrics.

## Generation experiments

[Generation pilot](docs/GENERATION_PILOT.md) explains the task, labels, conditions and metrics. [Development review](docs/GENERATION_FINAL_REVIEW.md) records limitations of the pilot; it is an assistant audit, not human faithfulness scoring. The [frozen held-out protocol](docs/GENERATION_FINAL_PROTOCOL.md) specifies the final evaluation and resume command. Do not change files bound by its manifest while the run is active.

## Data, results and sharing

Git intentionally excludes `data/` and `results/`: model weights, datasets, cached embeddings, raw responses, checkpoints and numerical results will **not** be included in a push. Scripts for analysis and the presentation depend on those local artifacts. Back up and share them separately subject to dataset/model terms and any privacy constraints; a clone of this repository alone cannot reproduce existing reported numbers without those inputs and rerunning the experiments. `docs/generation-final-protocol.json` records hashes for the frozen run, but does not contain the artifacts themselves.

See [project plan](docs/project-plan.md), [resources](docs/resources.md), and [presentation notes](presentation/speaker-notes.md) for context. The slide-deck source requires `python-pptx`, Matplotlib and the local results; the generated PDF and PowerPoint can be viewed without rerunning it.
