# Jordanian Labor Law Ontology-Driven GraphRAG — Final Graph-Only Evidence Architecture

This package is the cleaned final implementation of the Jordanian Labor Law QA pipeline.
It contains **one retrieval architecture only**. The old hybrid/direct Article search,
paragraph retrieval, mixed retrieval, global/V1 retrieval branch, full-law recovery,
and article-neighbour rescue logic are not present.

## Retrieval invariant

Semantic retrieval is used **only to link a question to ontology/KG concepts**.
A legal Article is allowed into the candidate set **only when relation-aware KG
traversal reaches that Article**.

```text
Question
  ↓
Shared evaluated LLM planner / issue decomposition
  ↓
Ontology concept linking
  ├─ Arabic labels + SKOS aliases / BM25
  ├─ fixed OpenAI embeddings
  ├─ planner concept hints
  └─ graph-degree specificity score
  ↓
Thresholded high-confidence semantic graph seeds
  ↓
Relation-aware KG traversal
  ↓
GRAPH-DERIVED ARTICLE CANDIDATES ONLY
  ↓
Graph candidate scoring
  ↓
Shared evaluated LLM evidence reranker
  ├─ minimal supported Article set
  └─ empty selection when graph evidence is insufficient
  ↓
Grounded generator
  ├─ answer + Article citations
  └─ insufficient evidence
```

### Explicitly removed

- direct Article vector/BM25 retrieval
- direct Paragraph retrieval and parent promotion
- broad/mixed document retrieval
- separate global/V1 retrieval path
- `RETRIEVAL_DIRECT_ARTICLE_ENABLED`
- `RETRIEVAL_PARAGRAPH_ENABLED`
- hybrid source weights / agreement bonus
- full-law catalogue recovery
- statutory Article-number neighbour expansion
- deterministic LLM-stage fallbacks

`hasArticle` and `hasParagraph` are structural RDF relations and are intentionally **not**
traversable evidence relations.

## KG coverage

The supplied TTL contains the complete 142 Article nodes. A semantic coverage layer was
added for provisions that previously had no semantic concept-to-Article bridge. The
offline validator now requires **142/142 Articles to be semantically reachable** from
retrieval-eligible concepts through `supportedByArticle`, `regulatedBy`, or reverse
`regulates` evidence bridges.

Article 1, for example, is represented through reusable concepts such as:

- `law_identity_concept`
- `law_commencement_concept`

rather than an Article-1 keyword rule.

## Important security step

Do **not** reuse any API key that was pasted into chat/logs. Rotate exposed keys first.
This ZIP contains no copied secret values. Copy `.env.example` to `.env` and insert the
new keys locally.

## Clean setup

From the project root:

```bash
cp .env.example .env
```

Fill at least:

```env
OPENAI_API_KEY=...
WEAVIATE_API_KEY=...
PIPELINE_LLM_PROVIDER=openai
PIPELINE_LLM_MODEL=gpt-5-nano
```

Start Weaviate:

```bash
docker compose up -d weaviate
```

Build the API image:

```bash
docker compose build api
```

### 1. Offline TTL validation

```bash
docker compose run --rm api python scripts/inspect_ttl.py
```

The report must show:

```text
article_count: 142
semantically_reachable_article_count: 142
unreachable_article_numbers: []
status: valid
```

You can also run:

```bash
docker compose run --rm api python scripts/audit_graph_reachability.py
```

### 2. Recreate schema + ingest

The Weaviate node schema changed (`aliases*` and `retrievalEligible` were added), so a
**clean reset/re-ingestion is required**:

```bash
docker compose run --rm api python scripts/ingest_kg.py --reset
```

### 3. Start API

```bash
docker compose up -d api
```

Check:

```bash
curl http://localhost:8000/health
```

### 4. One retrieval diagnostic

```bash
docker compose exec -T api sh -lc "cd /app && PYTHONPATH=/app python scripts/retrieve.py 'ما اسم قانون العمل الوارد في النص، وبعد كم يوماً من نشره في الجريدة الرسمية يبدأ نفاذه؟' --debug"
```

In debug output, Article evidence must have graph support paths. There is no direct or
paragraph retrieval channel to rescue it.

## Offline invariant tests

```bash
PYTHONPATH=. pytest -q
```

The tests check that:

- all 142 Articles are semantically reachable;
- Article/Paragraph/Definition/Law nodes cannot be ontology-search seeds;
- `hasArticle`, `hasParagraph`, and `rdf:type` are not traversable evidence relations;
- direct Article/Paragraph/mixed retrieval implementations do not exist;
- the reranker contract explicitly allows an empty evidence selection.

## Model comparison

Across compared LLMs, change only:

```env
PIPELINE_LLM_PROVIDER=...
PIPELINE_LLM_MODEL=...
```

Keep the embedding model, graph, prompts, concept-linking weights, traversal parameters,
reranker limits, generator budget, benchmark, and judge fixed.

Example:

```bash
docker compose exec -T api sh -lc "cd /app && PYTHONPATH=/app python scripts/run_full_pipeline_evaluation.py --model-name gpt-5-nano --benchmark /app/data/benchmarks/jordan_labor_law_fresh_final_40_v2_3.json --runs 1 --output-dir /app/data/model_evaluations/graph_only_final/openai__gpt-5-nano"
```

The evaluator has no `--allow-llm-fallback` mode. Provider/infrastructure failures remain
execution failures; model-produced contract violations are scoreable pipeline failures.

## Methodology wording for the paper

> The proposed ontology-driven GraphRAG framework uses lexical and semantic retrieval
> exclusively for ontology concept linking and graph-seed selection. Legal Article
> evidence is introduced only through relation-aware knowledge-graph traversal. A
> constrained LLM reranker selects a minimal supporting subset from graph-derived
> Article candidates, after which the answer generator is restricted to the selected
> statutory evidence.

## Benchmark note

Because the architecture and KG were changed after inspecting earlier benchmark failures,
that already-inspected benchmark should be treated as a **development/frozen comparison
set**, not claimed as a pristine unseen holdout. For a strong final paper result, freeze
this code + TTL first, then create a new untouched final holdout before reporting a
"fresh unseen" result.
