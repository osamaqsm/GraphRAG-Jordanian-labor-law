# Stage 7.6-B — Constrained full-catalog article reranking

## Purpose

Stage 7.6-A solved the safety layer but left provision-level confusion and all unseen multi-article cases unsolved. Stage 7.6-B adds one constrained OpenAI reranking call after deterministic retrieval.

The reranker receives the exact text of all 142 articles and may return only article numbers present in that catalogue. Python validates the output. Any API, schema, or validation failure falls back to the existing deterministic ranking.

## Files

- `app/legal_article_reranker.py` — strict OpenAI JSON-schema article selector
- `app/legal_question_analysis.py` — fixes accompaniment-leave routing
- `app/retrieval_service.py` — integrates full-catalog reranking
- `scripts/test_stage_7_6b.py` — offline routing/schema checks

## Optional environment variables

The defaults work with the current `.env`.

```env
OPENAI_RERANK_ENABLED=true
OPENAI_RERANK_MODEL=
OPENAI_RERANK_REASONING_EFFORT=low
OPENAI_RERANK_ARTICLE_CHAR_LIMIT=2500
```

Leave `OPENAI_RERANK_MODEL` unset to reuse `OPENAI_CHAT_MODEL`.

## Install

Back up the current files:

```powershell
Copy-Item .\app\legal_question_analysis.py .\app\legal_question_analysis_stage_7_6a.py -Force
Copy-Item .\app\retrieval_service.py .\app\retrieval_service_stage_7_6a.py -Force
```

Copy the patch:

```powershell
Copy-Item .\stage_7_6b_patch\app\legal_article_reranker.py .\app\legal_article_reranker.py -Force
Copy-Item .\stage_7_6b_patch\app\legal_question_analysis.py .\app\legal_question_analysis.py -Force
Copy-Item .\stage_7_6b_patch\app\retrieval_service.py .\app\retrieval_service.py -Force
Copy-Item .\stage_7_6b_patch\scripts\test_stage_7_6b.py .\scripts\test_stage_7_6b.py -Force
```

Validate:

```powershell
docker compose exec api python -m py_compile `
    /app/app/legal_article_reranker.py `
    /app/app/legal_question_analysis.py `
    /app/app/retrieval_service.py `
    /app/scripts/test_stage_7_6b.py
```

Run offline checks:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.test_stage_7_6b
```

Restart:

```powershell
docker compose restart api
```

## Targeted smoke tests

```powershell
# U06–U09: neighbouring wage/leave/maternity provisions
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_retrieval_benchmark `
    --start 6 --limit 4

# U36–U40: all multi-article cases
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_retrieval_benchmark `
    --start 36 --limit 5
```

Run the complete benchmark:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_unseen_retrieval_benchmark
```

Copy results:

```powershell
docker compose cp `
    api:/tmp/retrieval_benchmark_unseen_50_results.json `
    .\data\benchmarks\retrieval_benchmark_unseen_50_results_stage_7_6b.json
```

## Important

This benchmark is now a development set because its failures were used to design the patch. After Stage 7.6 is stable, evaluate on a new untouched test set.
