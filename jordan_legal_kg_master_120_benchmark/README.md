# Jordanian Legal KG Master 120 Benchmark

This package merges all four benchmark datasets created so far:

- Original development benchmark: 20
- Unseen benchmark: 50
- Final unseen benchmark: 30
- Stage 7.7-D fresh holdout: 20

Total: **120 unique questions**

Behavior distribution:

- Retrieve: 98
- Clarify: 11
- Abstain: 11

Benchmark SHA-256:

```text
3e87f2828356fbf1f13bedf99012dec1ecde77962e08d93b3af4f80625adfcd9
```

This is a combined **regression benchmark**, not a fresh unseen holdout.

## Install

```powershell
Copy-Item `
    .\jordan_legal_kg_master_120_benchmark\data\benchmarks\retrieval_benchmark_master_all_120.json `
    .\data\benchmarks\retrieval_benchmark_master_all_120.json `
    -Force
```

## Run all 120 questions

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark `
    --benchmark /app/data/benchmarks/retrieval_benchmark_master_all_120.json `
    --output /tmp/retrieval_benchmark_master_all_120_results.json
```

## Copy the result

```powershell
docker compose cp `
    api:/tmp/retrieval_benchmark_master_all_120_results.json `
    .\data\benchmarks\retrieval_benchmark_master_all_120_results.json
```

## Optional: run in chunks

First 30:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark `
    --benchmark /app/data/benchmarks/retrieval_benchmark_master_all_120.json `
    --output /tmp/master_120_part_1.json `
    --start 1 `
    --limit 30
```

Then use starts 31, 61, and 91 for the remaining chunks.
