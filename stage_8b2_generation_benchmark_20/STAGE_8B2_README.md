# Stage 8-B2 — Generation-Only Benchmark (20 Cases)

This benchmark evaluates only:

```text
frozen retrieval.v1 JSON
        ↓
GroundedAnswerGenerator
        ↓
generation.v1 JSON
```

It does not evaluate or execute retrieval.

## Why the inputs are oracle-frozen

Each retrieval input contains the complete correct gold article text. This
isolates the answer generator from retrieval quality. A failure therefore
points to the prompt, answer-generation logic, citation checks, or output
schema—not to vector search, BM25, KG traversal, routing, or reranking.

## Composition

- 20 retrieve cases
- 15 single-article cases
- 5 multi-article cases
- simple direct questions
- numerical questions
- conditional rules
- long statutory lists
- procedures
- multi-provision synthesis

Article 47 is included specifically to detect the overbroad formulation:

```text
لا يجوز حسم أي مبلغ ... إلا لاسترداد السلفة
```

which incorrectly converts one permitted deduction into the only permitted
deduction.

## Automated dimensions

1. Generation status and grounded flag
2. Exact required citations
3. Inline citation coverage
4. Required legal-fact coverage
5. Forbidden or overbroad claims
6. Style:
   - no separate `المصدر:` section
   - no generic warning when evidence is complete
   - empty `key_points` for simple questions
   - answer-length limits
7. Token usage and latency

## Run one case first

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_generation_benchmark_20 `
    --benchmark /app/data/benchmarks/generation_benchmark_20.json `
    --retrieval-dir /app/data/generation_benchmark_20/retrieval_inputs `
    --output /tmp/generation_benchmark_g01_result.json `
    --case G01 `
    --debug
```

## Run all 20

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.run_generation_benchmark_20 `
    --benchmark /app/data/benchmarks/generation_benchmark_20.json `
    --retrieval-dir /app/data/generation_benchmark_20/retrieval_inputs `
    --output /tmp/generation_benchmark_20_results.json `
    --debug
```

By default, failed benchmark cases do not make the command fail; the result JSON is always written. Add `--strict-exit` when using the benchmark in CI and you want exit code `1` if any strict case fails.

Copy the result from `/tmp` because `/app/data` may be read-only.
