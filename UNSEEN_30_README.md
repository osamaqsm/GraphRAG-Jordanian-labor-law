# Final unseen 30-question benchmark

This package is a frozen final evaluation set for the Jordanian Labor Law KG retrieval system.
It contains no retrieval-code changes.

## Distribution

- 24 retrieval questions
  - 5 straightforward
  - 5 paraphrased
  - 3 typo/noisy Arabic
  - 3 Jordanian colloquial
  - 3 numerical/deadline questions
  - 5 multi-article questions
- 3 ambiguity cases that should return no articles
- 3 out-of-scope cases that should return no articles

All gold retrieval articles are new: none appeared as required or acceptable gold articles in the earlier 20-question or 50-question benchmark.

## Install

Extract the ZIP from the project root so that the `data` and `scripts` folders merge into the project.

## Validate before the first run

```powershell
docker compose exec api `
    python -m py_compile `
    /app/scripts/run_unseen_30_benchmark.py `
    /app/scripts/validate_unseen_30_benchmark.py
```

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.validate_unseen_30_benchmark
```

## Run exactly once before any further code changes

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark
```

The result is written inside the container to:

```text
/tmp/retrieval_benchmark_unseen_30_final_results.json
```

Copy it to Windows:

```powershell
docker compose cp `
    api:/tmp/retrieval_benchmark_unseen_30_final_results.json `
    .\dataenchmarksetrieval_benchmark_unseen_30_final_results.json
```

## Partial diagnostic runs

These commands should only be used after the full first run has been saved.

```powershell
# First five questions
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark --limit 5

# Five multi-article questions
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark --start 20 --limit 5

# Six safety questions
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_30_benchmark --start 25 --limit 6
```

Do not edit the benchmark after the first run. Preserve its SHA-256 and the raw result file.
