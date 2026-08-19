# Multi-Article Retrieval V2 Patch

This patch upgrades the existing Jordanian Labor Law GraphRAG retrieval path without hard-coding benchmark article numbers.

## What changed

1. **Issue-pair preservation in the planner**
   - Keeps each planner issue label aligned with its retrieval query.

2. **Issue-wise semantic retrieval**
   - The original full-question embedding/retrieval path is preserved.
   - When the planner produces more than one atomic issue, each issue also gets an independent embedding and retrieval path.

3. **Issue-wise ontology/KG retrieval**
   - Each atomic issue performs concept retrieval and KG expansion independently.
   - A combined graph expansion uses a dynamic seed ceiling for genuine multi-issue questions.

4. **Issue-balanced candidate aggregation**
   - Candidate lists are interleaved across issues so one dominant issue cannot consume the entire pool.
   - The default multi-issue reranker pool is 15 candidates (3 per issue for five-issue questions), while simple questions keep the original 12-candidate V1 path.

5. **Explicit evidence coverage checking**
   - For multi-issue questions the reranker returns an issue-by-issue coverage map.
   - If one issue is not covered by the reranker, the strongest independently retrieved candidate for that issue is used as a bounded deterministic repair.
   - A minimum deterministic relevance threshold prevents arbitrary low-relevance candidates from being forced into the answer set.

6. **Single-issue regression protection**
   - One-issue questions keep the original V1 reranker prompt/schema and original one-vector retrieval path.

## Files to replace

- `app/config.py`
- `app/graph_retrieval.py`
- `app/legal_article_reranker.py`
- `app/legal_query_planner.py`
- `app/retrieval_pipeline.py`
- `app/retrieval_service.py`

Optional diagnostic script:

- `scripts/smoke_test_multi_article_v2.py`

## New default settings

No `.env` change is required. Defaults are defined in `app/config.py`:

```env
RETRIEVAL_ISSUE_CANDIDATES_PER_ISSUE=3
RETRIEVAL_ISSUE_GRAPH_SEED_LIMIT=5
RERANKER_MULTI_ISSUE_CANDIDATE_LIMIT=15
```

The existing `RETRIEVAL_ARTICLE_TOP_K=5` remains unchanged.

## Important experiment note

This is a pipeline change. Results produced with V2 are **not directly continuations** of the earlier frozen V1 GPT/Gemini runs. If V2 becomes the final conference-paper pipeline for the model-comparison experiment, all evaluated models must be rerun under the same V2 configuration.


## V2.1 correction
- Atomic-issue embeddings now use concise issue labels, not verbose planner questions.
- Per-issue retrieval uses focused text with generic legal boilerplate removed.
- Coverage repair no longer trusts the first issue candidate; it requires a targeted constrained reranker verification.
- Per-issue candidate pool defaults to 4 and the multi-issue reranker pool to 20.
