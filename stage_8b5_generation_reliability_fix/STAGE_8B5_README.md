# Stage 8-B5 — Generation Reliability Fix

This patch is based on the second real-retrieval 20-case run.

It does not change retrieval or any frozen retrieval JSON.

## Generator changes

- Repairs the safe one-article case where the model returns the correct
  structured article number but omits the visible inline citation.
- Multi-article citation omissions still fail closed.
- Unretrieved citations still fail closed.
- Strengthens actor alignment so a worker duty is not transferred to an
  inspector, employer, union, or ministry.
- Requires the minimum sufficient article set.
- Requires every answer sentence to address the user's actual question.

## Evaluation rubric v1.2

Adds Arabic-equivalent matching for six false negatives while preserving the
real strict failures:

- G10: actor confusion and unnecessary Article 19.
- G16: unnecessary Article 41.
- G18: unnecessary Article 36.
- G19: omitted reasons why settlement failed.
