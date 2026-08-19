# Final multi-provider model-comparison patch

This package is based on the exact files supplied on 2026-08-12.

## What changed

1. `app/llm_provider.py`
   - Shared structured-output interface now supports:
     - `openai`
     - `anthropic` (kept for compatibility)
     - `google`
     - `ollama`
   - OpenAI, Anthropic, Gemini, and Ollama all use the same stage-level Pydantic contracts.
   - Exactly one output-budget recovery retry is permitted when a structured response is truncated by the output-token ceiling.
   - Ollama uses native `/api/chat` structured output with JSON Schema, `num_ctx`, and `num_predict`.

2. `app/config.py`
   - Added Google and Ollama settings.
   - Added provider-neutral reranker candidate/context controls.
   - Planner default output budget now matches the frozen experimental value `3000`.

3. `app/retrieval_service.py` + `app/legal_article_reranker.py`
   - The LLM reranker no longer receives the complete 142-article statute.
   - Every model receives the same top-12 hybrid candidate pool.
   - Article text is bounded by one shared total reranker budget (`12000` chars) allocated dynamically across candidates.
   - This is necessary for the exact same semantic stage to run on 8K-context Aya models as well as hosted APIs.

4. `app/grounded_answer_generator.py`
   - `/generate` still consumes the exact `/retrieve` response and never reruns retrieval.
   - The LLM now receives a compact evidence view containing only routing status and the retrieved legal articles needed for answering, rather than graph/support metadata that is irrelevant to answer generation.
   - Strict initialization errors now expose their provider/model cause.

5. `scripts/run_full_pipeline_evaluation.py`
   - Default final mode: **3 independent repetitions**.
   - Produces:
     - `run-1.json`
     - `run-2.json`
     - `run-3.json`
     - `final-summary.json`
   - `final-summary.json` reports mean, sample standard deviation, min, and max for all main retrieval/generation metrics.
   - If any run has an execution failure, the experiment is marked `incomplete` and no partial aggregate mean is reported.
   - Backward compatibility: passing `--output ...` forces a single run, so previous smoke commands remain usable.

6. `requirements.txt`
   - Added `google-genai==2.13.0`.

7. `docker-compose.yml`
   - Added `host.docker.internal` mapping so the API container can call Ollama running on the host.

8. `.env.model-comparison.example`
   - Sanitized final configuration for GPT, Gemini, Qwen3 8B, and Aya 8B.

9. `scripts/smoke_test_pipeline_model.py`
   - Small end-to-end smoke test for the currently active provider/model.

## Benchmark integrity

`jordan_labor_law_final_unseen_40.json` is included unchanged.
SHA-256:

`a356c5a16c20c382aa377294105062152adbf5132dbea82801ebcd9ca2159739`

Because this benchmark has already been inspected during debugging, use it as the **frozen model-comparison benchmark**, not as a never-seen holdout claim.

## Install / replace

Copy these files into the project at the same relative paths:

- `app/config.py`
- `app/llm_provider.py`
- `app/legal_query_planner.py`
- `app/legal_article_reranker.py`
- `app/grounded_answer_generator.py`
- `app/retrieval_service.py`
- `app/generation_api.py`
- `scripts/run_full_pipeline_evaluation.py`
- `scripts/smoke_test_pipeline_model.py`
- `requirements.txt`
- `docker-compose.yml`

Use `.env.model-comparison.example` as the reference for updating the existing `.env`; do not overwrite real keys with `REPLACE_ME`.

Place the benchmark at:

`data/benchmarks/jordan_labor_law_final_unseen_40.json`

## Local Ollama models

Run on the host (not inside the API container):

```bash
ollama pull qwen3:8b
ollama pull aya:8b
```

If the study uses the newer Aya Expanse instead of Aya 23:

```bash
ollama pull aya-expanse:8b
```

Choose exactly one Aya model string before the final freeze and use that exact tag for all three runs.

## Rebuild

```bash
docker compose up -d --build api
```

Check:

```bash
docker compose ps -a
```

## Smoke test for the active model

```bash
docker compose exec -T api sh -c "cd /app && PIPELINE_STRICT_EVALUATION=true PYTHONPATH=/app python scripts/smoke_test_pipeline_model.py"
```

Do this once after switching provider/model and recreating the API container.

## Final three-run command

With the desired provider/model already active in `.env` and the API recreated:

```bash
docker compose exec -T api sh -c "cd /app && PIPELINE_STRICT_EVALUATION=true PYTHONPATH=/app python scripts/run_full_pipeline_evaluation.py --benchmark /app/data/benchmarks/jordan_labor_law_final_unseen_40.json"
```

No `--runs` is needed: the default is `3`.

Outputs are written under:

`/app/data/model_evaluations/<provider>__<model>/`

Example:

```text
data/model_evaluations/openai__gpt-5-nano/
├── run-1.json
├── run-2.json
├── run-3.json
└── final-summary.json
```

## Model switching examples

GPT:

```env
PIPELINE_LLM_PROVIDER=openai
PIPELINE_LLM_MODEL=gpt-5-nano
```

Gemini:

```env
PIPELINE_LLM_PROVIDER=google
PIPELINE_LLM_MODEL=gemini-3.6-flash
```

Qwen:

```env
PIPELINE_LLM_PROVIDER=ollama
PIPELINE_LLM_MODEL=qwen3:8b
```

Aya 23:

```env
PIPELINE_LLM_PROVIDER=ollama
PIPELINE_LLM_MODEL=aya:8b
```

Aya Expanse alternative:

```env
PIPELINE_LLM_PROVIDER=ollama
PIPELINE_LLM_MODEL=aya-expanse:8b
```

After any `.env` model switch, recreate the API so its process receives the new environment:

```bash
docker compose up -d --force-recreate api
```

Do not change prompts, KG, candidate limits, retrieval weights, embedding model, judge model, output budgets, or retry policy after starting the final model runs.
