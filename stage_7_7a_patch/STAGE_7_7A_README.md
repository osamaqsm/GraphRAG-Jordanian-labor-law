# Stage 7.7-A — Optional pre-retrieval LLM query planner

This patch adds an optional LLM layer before concept linking and article retrieval.
It does not answer legal questions and does not select article numbers.

## Pipeline

```text
User question
    ↓
Optional LLM query planner
    ├── retrieve: normalize + decompose + generate focused Arabic queries
    ├── clarify: return no articles
    └── abstain: return no articles
    ↓
Existing concept linking + vector/BM25/KG retrieval
    ↓
Existing constrained article reranker
```

## Important behavior

- The planner is **disabled by default** after installation.
- `OPENAI_QUERY_PLANNER_ENABLED=false` performs no planning API call.
- If the planner is disabled, unavailable, times out, or returns invalid JSON,
  the existing deterministic analyzer is used unchanged.
- The planner never returns article numbers.
- Generated atomic queries are independently searched with BM25 against:
  concepts, articles, and paragraphs.
- The original user-question embedding is still retained for vector search.
- The structured issue plan is passed to the existing article reranker as
  additional context.
- Clear deterministic abstention decisions are never weakened by an LLM
  `retrieve` decision.

## Files

```text
app/legal_query_planner.py
app/legal_question_analysis.py
app/retrieval_service.py
scripts/test_stage_7_7a.py
scripts/test_query_planner_live.py
```

## Environment variables

Enable the planner:

```env
OPENAI_QUERY_PLANNER_ENABLED=true
OPENAI_QUERY_PLANNER_MODEL=gpt-5-nano
OPENAI_QUERY_PLANNER_REASONING_EFFORT=low
OPENAI_QUERY_PLANNER_ROUTE_CONFIDENCE=0.80
OPENAI_QUERY_PLANNER_RETRIEVE_OVERRIDE_CONFIDENCE=0.90
```

Disable only the planner:

```env
OPENAI_QUERY_PLANNER_ENABLED=false
```

The article reranker remains independently controlled by:

```env
OPENAI_RERANK_ENABLED=true
```

Possible combinations:

```text
Planner false + reranker true  → current Stage 7.6-C behavior
Planner true  + reranker true  → recommended full pipeline
Planner true  + reranker false → planned hybrid retrieval, deterministic final rank
Planner false + reranker false → fully deterministic retrieval
```

## Installation

Back up the current files:

```powershell
Copy-Item .\app\legal_question_analysis.py .\app\legal_question_analysis_before_7_7a.py -Force
Copy-Item .\app\retrieval_service.py .\app\retrieval_service_before_7_7a.py -Force
```

Copy the patch:

```powershell
Copy-Item .\stage_7_7a_patch\app\legal_query_planner.py .\app\legal_query_planner.py -Force
Copy-Item .\stage_7_7a_patch\app\legal_question_analysis.py .\app\legal_question_analysis.py -Force
Copy-Item .\stage_7_7a_patch\app\retrieval_service.py .\app\retrieval_service.py -Force
Copy-Item .\stage_7_7a_patch\scripts\test_stage_7_7a.py .\scripts\test_stage_7_7a.py -Force
Copy-Item .\stage_7_7a_patch\scripts\test_query_planner_live.py .\scripts\test_query_planner_live.py -Force
```

Validate syntax:

```powershell
docker compose exec api python -m py_compile `
    /app/app/legal_query_planner.py `
    /app/app/legal_question_analysis.py `
    /app/app/retrieval_service.py `
    /app/scripts/test_stage_7_7a.py `
    /app/scripts/test_query_planner_live.py
```

Run offline checks:

```powershell
docker compose exec -T api env PYTHONIOENCODING=utf-8 python -m scripts.test_stage_7_7a
```

After editing `.env`, recreate the API container:

```powershell
docker compose up -d --force-recreate api
```

Verify safe environment variables without printing the API key:

```powershell
docker compose exec api python -c "import os; print('planner=', os.getenv('OPENAI_QUERY_PLANNER_ENABLED')); print('model=', os.getenv('OPENAI_QUERY_PLANNER_MODEL')); print('route_conf=', os.getenv('OPENAI_QUERY_PLANNER_ROUTE_CONFIDENCE')); print('reranker=', os.getenv('OPENAI_RERANK_ENABLED'))"
```

Run the live planner-only smoke test:

```powershell
docker compose exec -T api env PYTHONIOENCODING=utf-8 python -m scripts.test_query_planner_live
```

## Rollback

Disable without changing code:

```env
OPENAI_QUERY_PLANNER_ENABLED=false
```

Then recreate the API container:

```powershell
docker compose up -d --force-recreate api
```
