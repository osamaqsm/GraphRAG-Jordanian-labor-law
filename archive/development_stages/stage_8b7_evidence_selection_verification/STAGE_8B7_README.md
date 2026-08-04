# Stage 8-B7 — Generic Evidence Selection and Answer Verification

This patch is based on the sealed unseen findings, but it contains no holdout
question IDs, expected answers, or article-specific fixes.

## Generation boundary

```text
completed retrieval.v1
        ↓
Evidence selection call
  - decompose requested issues
  - choose minimum sufficient retrieved articles
  - prefer specific over general provisions
  - exclude related but unnecessary articles
        ↓
Answer call using selected articles only
  - answer every supported issue
  - preserve actors, conditions, numbers, and exceptions
  - return issue-to-article coverage
        ↓
Deterministic validation
  - selected/excluded articles must account for retrieval
  - every supported issue must be covered
  - citations must be selected and retrieved
  - issue support must match cited articles
  - citation normalization must be idempotent
```

The module still performs zero retrieval, embedding, graph traversal, or
reranking calls.

## New environment variables

```env
OPENAI_ANSWER_EVIDENCE_SELECTION_ENABLED=true
OPENAI_ANSWER_SELECTION_MODEL=gpt-5-nano
OPENAI_ANSWER_SELECTION_REASONING_EFFORT=low
OPENAI_ANSWER_MAX_SELECTED_ARTICLES=4
```

The existing answer-model variables continue to apply.

## Cost and latency

Retrieve routes normally use two model calls: one selection call and one final
answer call. Clarify and abstain routes use zero model calls. Usage in
`generation.v1` is the sum of both calls, and debug output includes a per-stage
breakdown.

## Evaluation rule

Do not rerun the sealed Stage 8-B6 holdout as an unseen test. Use existing
questions only as development/regression cases. After Stage 8-B7 is frozen,
create a different holdout for the next unseen measurement.
