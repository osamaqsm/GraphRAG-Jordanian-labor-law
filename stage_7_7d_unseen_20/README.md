# Stage 7.7-D Fresh Unseen 20 Holdout

This package contains a frozen 20-question Arabic holdout:

- 14 retrieve cases
- 3 clarify cases
- 3 abstain cases
- 15 gold article numbers
- zero gold-article overlap with the earlier 20-, 50-, and 30-question benchmarks

Benchmark SHA-256:

```text
9888e7720c0b0ea07a67fd4df6d5a6d4c6b9532f1a37e0e01fb8fc72a8d232dc
```

## Install

Copy:

```powershell
Copy-Item `
    .\stage_7_7d_unseen_20\data\benchmarks\retrieval_benchmark_unseen_20_stage_7_7d.json `
    .\data\benchmarks\retrieval_benchmark_unseen_20_stage_7_7d.json `
    -Force

Copy-Item `
    .\stage_7_7d_unseen_20\scripts\validate_unseen_20_stage_7_7d.py `
    .\scripts\validate_unseen_20_stage_7_7d.py `
    -Force
```

## Validate before the first run

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.validate_unseen_20_stage_7_7d
```

## Run once and freeze the result

The existing 30-question runner is generic and accepts this 20-question
benchmark through explicit paths:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark `
    --benchmark /app/data/benchmarks/retrieval_benchmark_unseen_20_stage_7_7d.json `
    --output /tmp/retrieval_benchmark_unseen_20_stage_7_7d_results.json
```

Copy the result:

```powershell
docker compose cp `
    api:/tmp/retrieval_benchmark_unseen_20_stage_7_7d_results.json `
    .\data\benchmarks\retrieval_benchmark_unseen_20_stage_7_7d_results.json
```

Do not change prompts, thresholds, retrieval logic, KG data, or reranking
between validation and the first complete run.
