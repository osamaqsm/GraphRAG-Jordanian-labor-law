# Multi-Article Retrieval V2.3

V2.3 is a corrective patch on top of V2.2.

## Why V2.3 exists
V2.2 improved decomposition and allowed one issue to need multiple provisions, but its post-reranker support verification could corrupt a correct five-article selection. It also allowed per-issue seeds plus structural neighbours to consume the full multi-issue reranker catalogue before strong global GraphRAG candidates were reserved.

## V2.3 changes
1. Reserve up to 8 strong global GraphRAG candidates before statutory-neighbour expansion.
2. Use only the top 3 candidates per issue in the initial balanced seed pass, leaving catalogue capacity for the global retrieval arm.
3. Treat a full-cap, issue-complete reranker selection as final; support repair cannot overwrite it.
4. Run broad targeted recovery only for genuinely uncovered issues.
5. For covered issues with spare final capacity, use a compact challenger catalogue (reranker support + top two issue candidates) and never structural neighbours.
6. Preserve reranker-selected articles first and append only verified missing support while slots remain.
7. For uncovered-issue recovery, use selected/global evidence as additional neighbourhood seeds so statutory adjacency can recover related provisions such as a preceding/following stage in a legal process.

The final evidence limit remains 5 articles.
