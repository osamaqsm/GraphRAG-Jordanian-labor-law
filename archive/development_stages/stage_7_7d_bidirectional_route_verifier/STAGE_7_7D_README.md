# Stage 7.7-D — Bidirectional route verification

This patch addresses the three unstable cases found by the Stage 7.7-C
39-plan stability test:

- employment-related worker invention incorrectly classified as outside scope;
- general medical-fitness authority/publication question incorrectly sent to clarification;
- generic safety-violation question incorrectly sent to retrieval.

## Scope of this patch

Changed:

- `app/legal_query_planner.py`
- `scripts/test_stage_7_7d.py`
- `scripts/test_query_planner_stability.py`

Unchanged:

- deterministic question analysis;
- retrieval service;
- embeddings and Weaviate;
- KG traversal;
- article reranker;
- article-number-free design.

## Routing design

The first planner call remains the main planner.

A second route-verification call is made when:

1. the first plan is `clarify` or `abstain`; or
2. the first plan is `retrieve` with confidence below the configured
   verification boundary.

A high-confidence retrieve still takes the one-call fast path.

## New environment variables

```env
OPENAI_QUERY_PLANNER_VERIFY_LOW_CONFIDENCE_RETRIEVE=true
OPENAI_QUERY_PLANNER_RETRIEVE_VERIFY_BELOW=0.90
```

Existing variables remain in use:

```env
OPENAI_QUERY_PLANNER_VERIFY_NON_ANSWER=true
OPENAI_QUERY_PLANNER_NON_ANSWER_VERIFY_CONFIDENCE=0.80
```

The `NON_ANSWER_VERIFY_CONFIDENCE` value is retained for backward
compatibility and is used as the minimum confidence for the route verifier.

## Expected cost behavior

- Retrieve confidence >= 0.90: one planner call.
- Retrieve confidence < 0.90: two planner calls.
- Clarify or abstain: two planner calls.

## Validation order

1. Install the patch.
2. Run `python -m scripts.test_stage_7_7d`.
3. Recreate the API container.
4. Run the 13-case stability test with three repetitions.
5. Do not run the 30-question regression until the stability test reaches
   39/39.

## Important limitation

This patch corrects route stability only. It does not yet solve article-count
or neighboring-article over-selection. Those should be measured separately
after routing is stable.
