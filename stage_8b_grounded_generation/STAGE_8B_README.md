# Stage 8-B — Grounded Generation From Frozen Retrieval

Stage 8-B adds answer generation without merging it into retrieval.

## Boundary

```text
saved retrieval.v1 JSON
        ↓
GroundedAnswerGenerator
        ↓
generation.v1 JSON
```

The generator:

- does not import Weaviate;
- does not instantiate RetrievalService;
- does not create embeddings;
- does not traverse the KG;
- does not rerank articles;
- accepts only a completed `RetrievalResultV1`;
- rejects citations to articles absent from that retrieval result.

Clarify and abstain decisions return deterministic responses and do not call
the answer model.

## Files

- `app/generation_contract.py`
- `app/grounded_answer_generator.py`
- `app/generation_api.py`
- `scripts/generate_from_retrieval.py`
- `scripts/patch_main_stage_8b.py`
- `scripts/test_stage_8b.py`

## Environment

```env
OPENAI_ANSWER_ENABLED=true
OPENAI_ANSWER_MODEL=gpt-5-nano
OPENAI_ANSWER_REASONING_EFFORT=low
OPENAI_ANSWER_ARTICLE_CHAR_LIMIT=3000
```

## Generation-only CLI

```powershell
python -m scripts.generate_from_retrieval `
    .\data\retrieval_results\stage_8a_retrieval_result.json `
    --output .\data\generation_results\stage_8b_answer.json `
    --debug
```

The command reads the saved retrieval JSON. It does not run retrieval again.

## API

`POST /generate`

```json
{
  "retrieval": {
    "schema_version": "retrieval.v1",
    "...": "complete saved retrieval result"
  },
  "include_debug": false
}
```

## Not included yet

Stage 8-B does not replace or modify the existing `/ask` orchestration.
Connecting `/retrieve` and `/generate` into the final `/ask` path is Stage 8-C.
