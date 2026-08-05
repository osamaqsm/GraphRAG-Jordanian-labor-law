# Stage 7.7-B — Prompt-only query planner correction

This patch changes only `LegalQueryPlanner._instructions()`.

Unchanged:
- Pydantic models and strict response schema
- OpenAI model and reasoning effort
- confidence thresholds and merge logic
- deterministic analyzer
- retrieval service
- article reranker

## Expected six-case behavior

- Q1 retrieve, 2 issues, article range 2..2
- Q2 retrieve, 1 issue, article range 1..1
- Q3 clarify
- Q4 clarify
- Q5 abstain
- Q6 abstain

## Files

- `app/legal_query_planner.py`
- `scripts/test_stage_7_7a.py`
