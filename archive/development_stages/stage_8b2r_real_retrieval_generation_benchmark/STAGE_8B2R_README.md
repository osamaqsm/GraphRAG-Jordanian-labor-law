# Stage 8-B2R — Generation Benchmark From Actual Retrieval Outputs

This package replaces the earlier oracle-retrieval benchmark.

## Correct boundary

### Phase 1: collect and freeze retrieval

```text
20 benchmark questions
        ↓
the deployed RetrievalOnlyPipeline
        ↓
20 exact retrieval.v1 JSON files
        ↓
SHA-256 frozen snapshot
```

### Phase 2: test generation only

```text
the same frozen retrieval.v1 files
        ↓
GroundedAnswerGenerator
        ↓
generation.v1 results
```

The generation runner performs zero retrieval calls.

## Why metrics are separated

An actual retrieval result may omit a required article, return an extra
article, or choose clarify/abstain. In those cases, incomplete legal coverage
must be attributed to retrieval rather than to the answer prompt.

The report therefore separates:

- retrieval context completeness;
- generation strict accuracy on complete retrieval contexts;
- citation safety against the actual retrieved articles;
- overall usable pipeline rate.

## Included files

- `data/benchmarks/generation_rubric_20_for_real_retrieval.json`
- `scripts/collect_real_retrieval_generation_benchmark_20.py`
- `scripts/validate_real_retrieval_generation_snapshot_20.py`
- `scripts/run_generation_benchmark_real_retrieval_20.py`

No retrieval output is included in this package. The 20 files must be produced
by the user's deployed system so that they are genuinely actual results.
