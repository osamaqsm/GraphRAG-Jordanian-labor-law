# Final Simple Step 8 — Grounded Generation

This package intentionally removes the Stage 8-B7/B8 evidence-selection and
verification layers.

## Final architecture

```text
completed retrieval.v1
        ↓
one Responses API call
        ↓
generation.v1
```

`clarify` and `abstain` decisions return deterministic responses and make zero
LLM calls.

## Exact model input

The one LLM call receives one JSON user message:

```json
{
  "user_question": "the exact user question",
  "retrieval_result": {"the complete retrieval.v1 output": "..."}
}
```

The system/developer prompt explicitly states that only
`retrieval_result.articles[].text` is legal evidence. The remaining retrieval
fields are metadata.

## OpenAI configuration

Recommended `.env`:

```env
OPENAI_ANSWER_ENABLED=true
OPENAI_ANSWER_MODEL=gpt-5-nano
OPENAI_ANSWER_REASONING_EFFORT=low
OPENAI_ANSWER_VERBOSITY=low
OPENAI_ANSWER_MAX_OUTPUT_TOKENS=1800
OPENAI_ANSWER_TIMEOUT_SECONDS=120
OPENAI_ANSWER_MAX_RETRIES=2
```

The request intentionally does not send `temperature`, `top_p`, `seed`, or
`tools`. It uses strict JSON Schema Structured Outputs, low reasoning effort,
low verbosity, a bounded output, and `store=false`.

## Manual 20-question review

The included runner uses the existing saved real retrieval outputs:

```text
data/generation_real_retrieval_20/retrieval_inputs/
```

It creates:

- a JSON record of all answers;
- a Markdown report with a manual checklist under every answer;
- no pass/fail score;
- no benchmark metric.
