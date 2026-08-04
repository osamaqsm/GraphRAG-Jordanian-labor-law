# Model Comparison Protocol

## Experimental setup

The two models are evaluated using the same frozen benchmark containing:

- 40 Arabic questions
- 32 retrieval questions
- 8 out-of-scope questions

## Controlled components

The following components must remain unchanged:

- Benchmark questions
- Gold annotations
- Jordanian Labor Law knowledge graph
- Weaviate contents
- Embedding model
- Retrieval configuration
- Top-K configuration
- Query-planner prompt
- Generation prompt
- Output schemas
- Evaluation scripts
- Docker environment
- Evaluation rubric

## Independent variable

Only the evaluated language model is changed between Model A and Model B.

## Main metrics

1. Hit@1
2. Hit@3
3. Required Article Recall
4. Answer Correctness
5. Faithfulness
6. Citation Correctness

## Result policy

Existing model results must never be overwritten.

Each model receives a separate directory:

- `data/results/model_a`
- `data/results/model_b`

The first complete run for each model is preserved as the primary result.
