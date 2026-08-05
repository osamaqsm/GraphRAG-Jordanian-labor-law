# Stage 8-B6 — Fresh Unseen Generation Holdout

This package is a final unseen holdout, not a prompt-development benchmark.

## Composition

- 20 new Arabic questions
- 16 retrieve cases
- 2 clarify cases
- 2 abstain cases
- 18 unique retrieve gold articles
- zero gold-article overlap with the previous 120-question benchmark
- zero exact normalized question overlap with the previous 120 questions

## Required sequence

1. Install and test the generic Stage 8-B5 generator.
2. Validate this holdout against the previous master benchmark.
3. Run the real retrieval pipeline once and freeze the 20 retrieval.v1 files.
4. Validate every frozen retrieval SHA-256.
5. Run generation exactly once with the sealed runner.
6. Preserve both the result JSON and its seal JSON unchanged.

Do not tune the prompt, validator, retrieval, KG, or evaluation rubric after
seeing the first result and then report a rerun as unseen performance.

## Metrics

The sealed result separates:

- retrieval route accuracy;
- retrieval context completeness;
- generation accuracy on complete actual retrieval contexts;
- citation safety;
- legal fact coverage;
- overall pipeline usability.
