# Fresh Unseen 40 Benchmark

This package contains a new frozen Arabic benchmark for the Jordanian Labor Law KG system.

## Composition

- Total: 40 questions
- Retrieve: 32
- Out-of-scope abstain: 8
- Clarify: 0

The 32 retrieve cases use 32 legal articles that were never gold evidence in the previous 120-question regression suite.

- Exact normalized question overlap with the previous 120: 0
- Retrieve gold-article overlap with the previous 120: 0

## Files

- `retrieval_benchmark_fresh_unseen_40_blind.json`
  - Use this for the first run.
  - Contains no gold article numbers or expected answers.

- `retrieval_benchmark_fresh_unseen_40_gold.json`
  - Gold evaluation key.
  - Keep sealed until the first complete run has been saved.

## Freeze rule

Do not modify the model, prompt, routing logic, thresholds, retrieval code, reranker, KG, or Weaviate contents after seeing the first result and still call a rerun “unseen.”

## Install

From Git Bash in the project root:

```bash
cd /c/KGProjects/jordan-legal-kg
mkdir -p data/benchmarks

cp jordan_legal_kg_fresh_unseen_40/data/benchmarks/retrieval_benchmark_fresh_unseen_40_blind.json \
  data/benchmarks/retrieval_benchmark_fresh_unseen_40_blind.json

cp jordan_legal_kg_fresh_unseen_40/data/benchmarks/retrieval_benchmark_fresh_unseen_40_gold.json \
  data/benchmarks/retrieval_benchmark_fresh_unseen_40_gold.json
```

## First blind run

The existing benchmark runner normally needs the gold key to score results. To preserve a truly blind first execution, run the API/batch workflow using the blind file and save all raw outputs first.

Your Step 7 -> Step 8 runner can process all 40 questions:

```bash
python scripts/run_stage7_step8_20.py \
  --questions data/benchmarks/retrieval_benchmark_fresh_unseen_40_blind.json \
  --output-dir data/manual_review/fresh_unseen_40_first_run
```

The script name still contains `20`, but it reads the actual `questions` array and can process 40 cases.

## Retrieval scoring after the blind run

After raw outputs are preserved, use the gold file with the retrieval benchmark runner:

```bash
docker compose exec -T api \
  env PYTHONIOENCODING=utf-8 \
  python -m scripts.run_unseen_30_benchmark \
  --benchmark /app/data/benchmarks/retrieval_benchmark_fresh_unseen_40_gold.json \
  --output /tmp/retrieval_benchmark_fresh_unseen_40_results.json
```

Copy the result:

```bash
docker compose cp \
  api:/tmp/retrieval_benchmark_fresh_unseen_40_results.json \
  ./data/benchmarks/retrieval_benchmark_fresh_unseen_40_results.json
```

## Checksums

Blind SHA-256:

```text
9fe29f91922aaf4574322fee623924d7d8ebe23c99a62c873b7904aa2a379ad6
```

Gold SHA-256:

```text
8f6035dea920b1aef40e4b28cd57635ae64eede1b05fa6a6df0711e6cf01ddc3
```
