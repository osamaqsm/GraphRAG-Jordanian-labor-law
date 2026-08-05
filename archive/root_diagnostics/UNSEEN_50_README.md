# Jordanian Labor Law KG — Frozen Unseen 50 Benchmark

This package evaluates the frozen Stage 7.5-E retrieval system before Stage 8.

## Benchmark composition

- 10 straightforward questions
- 10 paraphrased questions
- 5 questions with spelling/noise variations
- 5 Jordanian colloquial questions
- 5 numerical or duration questions
- 5 multi-article questions
- 5 ambiguous questions that should produce no article retrieval
- 5 out-of-scope questions that should produce no article retrieval

Total: 50 questions.

The benchmark contains 40 article-retrieval cases, 5 clarification cases, and 5 abstention cases.

Frozen benchmark SHA-256:

```text
6fd51eee1102f0498095263e1e1e5f5c8e9c48dadbc6911e1f6ae5e1e76f8745
```

Do not change retrieval code, benchmark questions, expected results, or thresholds until the first complete result has been saved.

## Files

```text
data/benchmarks/retrieval_benchmark_unseen_50.json
scripts/run_unseen_retrieval_benchmark.py
scripts/validate_unseen_benchmark.py
benchmark_manifest.json
```

## Installation

From:

```text
C:\KGProjects\jordan-legal-kg
```

Extract the ZIP into the project root. The files should end up at:

```text
C:\KGProjects\jordan-legal-kg\data\benchmarks\retrieval_benchmark_unseen_50.json
C:\KGProjects\jordan-legal-kg\scripts\run_unseen_retrieval_benchmark.py
C:\KGProjects\jordan-legal-kg\scripts\validate_unseen_benchmark.py
```

## Validate without running retrieval

```powershell
docker compose exec api `
    python -m py_compile `
    /app/scripts/run_unseen_retrieval_benchmark.py `
    /app/scripts/validate_unseen_benchmark.py
```

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.validate_unseen_benchmark
```

Expected SHA-256:

```text
6fd51eee1102f0498095263e1e1e5f5c8e9c48dadbc6911e1f6ae5e1e76f8745
```

## Run the first complete held-out baseline

Run all 50 questions in one execution before making any changes:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_retrieval_benchmark
```

The result is written inside the container to:

```text
/tmp/retrieval_benchmark_unseen_50_results.json
```

Copy it to Windows:

```powershell
docker compose cp `
    api:/tmp/retrieval_benchmark_unseen_50_results.json `
    .\data\benchmarks\retrieval_benchmark_unseen_50_results_baseline.json
```

Open it:

```powershell
code .\data\benchmarks\retrieval_benchmark_unseen_50_results_baseline.json
```

## Metrics

For the 40 retrieval cases:

- Hit@1
- Hit@3
- Mean reciprocal rank
- Required-article recall
- Strict precision and F1
- Exact article-set accuracy
- Concept-hit accuracy

For the 10 safety cases:

- Safe non-answer accuracy
- Clarification accuracy
- Out-of-scope abstention accuracy
- Explicit safety-signal accuracy

The runner also reports performance separately for each question type.

## Interpretation targets

Strong held-out targets are:

```text
Hit@1                         >= 0.90
Hit@3                         >= 0.95
Required-article recall       >= 0.95
Strict precision              >= 0.90
Concept-hit accuracy          >= 0.90
Clarification accuracy        >= 0.90
Out-of-scope abstention       >= 0.90
```

A lower result is useful evidence. Do not modify the benchmark to improve the score; diagnose the retrieval system instead.
