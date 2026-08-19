# Multi-Article Retrieval V2.2

This patch is applied on top of V2.1 and addresses the four remaining failure modes found by the seven-case multi-article diagnostic.

## Architectural changes

1. **Stricter atomic decomposition**
   - The planner must preserve explicitly enumerated legal elements.
   - Named stages in a procedure should become independent atomic issues when independently verifiable.

2. **One-to-many issue coverage**
   - A single planner issue may now require multiple supporting articles.
   - This removes the previous one-article-per-issue constraint, which was unsuitable for legal frameworks and multi-stage procedures.

3. **Structural statutory-neighbour recovery**
   - For multi-issue questions only, the system considers a small number of nearby article numbers around strong issue candidates.
   - Neighbours are candidates only and are never selected automatically.
   - Candidate neighbours are scored against the atomic issue before entering the reranker catalogue.

4. **Targeted support verification**
   - When an issue is uncovered or the proposed support is not the issue's strongest candidate, a dedicated verifier checks exact article text.
   - The verifier can explicitly return uncovered; it is not forced to choose a related article.
   - A verified issue may require multiple articles.

5. **Broader but bounded candidate budget**
   - Per-issue candidate count defaults to 6.
   - Multi-issue reranker candidate limit defaults to 30.
   - Final answer evidence remains capped at five articles.

## Important methodological note

This is a new pipeline version. If adopted for the paper's official model comparison, all evaluated models must be rerun under this exact V2.2 configuration. Do not combine V1/V2.1 official scores with V2.2 scores.

## Included diagnostic

`scripts/diagnose_multi_article_v2_2.py` evaluates FM40-30 through FM40-36 and prints recall, precision, issue candidates, structural neighbours, reranker output, and issue coverage.
