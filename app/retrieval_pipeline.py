from __future__ import annotations

import time
from typing import Any

import weaviate
from openai import OpenAI
from weaviate.classes.init import Auth

from app.config import Settings, get_settings
from app.retrieval_contract import (
    RetrievalDecisionV2,
    RetrievalDiagnosticsV2,
    RetrievalEmbeddingV2,
    RetrievalEvidenceV2,
    RetrievalResultV2,
)
from app.retrieval_service import RetrievalService


def _evidence(hit: Any) -> RetrievalEvidenceV2:
    return RetrievalEvidenceV2(
        uri=str(getattr(hit, "uri", "") or ""),
        local_name=str(getattr(hit, "local_name", "") or ""),
        node_kind=str(getattr(hit, "node_kind", "") or ""),
        labels_ar=[str(v) for v in getattr(hit, "labels_ar", []) or []],
        labels_en=[str(v) for v in getattr(hit, "labels_en", []) or []],
        aliases_ar=[str(v) for v in getattr(hit, "aliases_ar", []) or []],
        aliases_en=[str(v) for v in getattr(hit, "aliases_en", []) or []],
        article_number=getattr(hit, "article_number", None),
        text=str(getattr(hit, "text_preview", "") or ""),
        fused_score=float(getattr(hit, "fused_score", 0.0) or 0.0),
        semantic_score=float(getattr(hit, "semantic_score", 0.0) or 0.0),
        lexical_score=float(getattr(hit, "lexical_score", 0.0) or 0.0),
        planner_hint_score=float(getattr(hit, "planner_hint_score", 0.0) or 0.0),
        specificity_score=float(getattr(hit, "specificity_score", 0.0) or 0.0),
        seed_score=float(getattr(hit, "seed_score", 0.0) or 0.0),
        graph_score=float(getattr(hit, "graph_score", 0.0) or 0.0),
        issue_relevance=float(getattr(hit, "issue_relevance", 0.0) or 0.0),
        final_score=float(getattr(hit, "final_score", 0.0) or 0.0),
        graph_supported=bool(getattr(hit, "graph_supported", False)),
        support_paths=[dict(v) for v in getattr(hit, "support_paths", []) or []],
    )


class RetrievalOnlyPipeline:
    """Planner -> concept linker -> graph traversal -> graph evidence reranker."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        weaviate_client: Any | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_weaviate_client = weaviate_client is None
        self._owns_openai_client = openai_client is None
        self.weaviate_client = weaviate_client or self._connect_weaviate()
        self.openai_client = openai_client or OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )
        self.service = RetrievalService(self.weaviate_client, self.settings)

    def _connect_weaviate(self) -> Any:
        kwargs: dict[str, Any] = {
            "http_host": self.settings.weaviate_http_host,
            "http_port": self.settings.weaviate_http_port,
            "http_secure": False,
            "grpc_host": self.settings.weaviate_grpc_host,
            "grpc_port": self.settings.weaviate_grpc_port,
            "grpc_secure": False,
        }
        if self.settings.weaviate_api_key:
            kwargs["auth_credentials"] = Auth.api_key(self.settings.weaviate_api_key)
        return weaviate.connect_to_custom(**kwargs)

    def close(self) -> None:
        if self._owns_openai_client:
            self.openai_client.close()
        if self._owns_weaviate_client:
            self.weaviate_client.close()

    def __enter__(self) -> "RetrievalOnlyPipeline":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def retrieve(self, question: str, *, include_debug: bool = False) -> RetrievalResultV2:
        question = question.strip()
        if len(question) < 3:
            raise ValueError("Question must contain at least three characters.")

        started = time.perf_counter()
        plan = self.service.query_planner.plan(question)
        embedding_model = self.settings.openai_embedding_model
        embedding_dimensions = 0
        input_tokens = 0
        preview = None

        if plan.decision == "retrieve":
            embedding_inputs = [issue.embedding_text for issue in plan.atomic_issues]
            response = self.openai_client.embeddings.create(
                model=embedding_model,
                input=embedding_inputs,
                encoding_format="float",
            )
            vectors = tuple(list(item.embedding) for item in response.data)
            embedding_dimensions = len(vectors[0]) if vectors else 0
            usage = getattr(response, "usage", None)
            input_tokens = int(
                getattr(usage, "prompt_tokens", 0)
                or getattr(usage, "total_tokens", 0)
                or 0
            )

            preview = self.service.retrieve_plan(
                question=question,
                plan=plan,
                issue_vectors=vectors,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                embedding_input_tokens=input_tokens,
            )

        article_hits = list(getattr(preview, "article_hits", []) or [])
        candidate_hits = list(getattr(preview, "article_candidates", []) or [])
        concept_hits = list(getattr(preview, "concept_hits", []) or [])
        expanded_hits = list(getattr(preview, "expanded_concept_hits", []) or [])
        seed_hits = list(getattr(preview, "selected_seeds", []) or [])

        articles = [_evidence(hit) for hit in article_hits]
        concepts = [_evidence(hit) for hit in concept_hits]
        expanded = [_evidence(hit) for hit in expanded_hits]

        article_numbers = [
            int(article.article_number)
            for article in articles
            if article.article_number is not None
        ]
        candidate_numbers = [
            int(hit.article_number)
            for hit in candidate_hits
            if hit.article_number is not None
        ]

        debug = None
        if include_debug:
            debug = {
                "query_plan": plan.model_dump(mode="json"),
                "planner_provider": self.service.query_planner.provider,
                "planner_model": self.service.query_planner.model,
                "planner_verified": self.service.query_planner.last_verified,
                "issue_wise_retrieval": (
                    list(preview.issue_debug) if preview is not None else []
                ),
                "raw_preview": preview.to_dict() if preview is not None else None,
            }

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return RetrievalResultV2(
            question=question,
            decision=RetrievalDecisionV2(
                behavior=plan.decision,
                reason=plan.decision_reason,
                planner_used=True,
                planner_verified=self.service.query_planner.last_verified,
                planner_confidence=plan.confidence,
            ),
            embedding=RetrievalEmbeddingV2(
                model=embedding_model,
                dimensions=embedding_dimensions,
                input_tokens=input_tokens,
            ),
            articles=articles,
            concepts=concepts,
            expanded_concepts=expanded,
            diagnostics=RetrievalDiagnosticsV2(
                article_numbers=article_numbers,
                graph_candidate_article_numbers=candidate_numbers,
                concept_local_names=[c.local_name for c in concepts if c.local_name],
                selected_seed_local_names=[hit.local_name for hit in seed_hits],
                graph_supported_articles=[
                    int(article.article_number)
                    for article in articles
                    if article.article_number is not None and article.graph_supported
                ],
                concept_count=len(concepts),
                expanded_concept_count=len(expanded),
                graph_candidate_count=len(candidate_hits),
                article_count=len(articles),
            ),
            elapsed_ms=elapsed_ms,
            debug=debug,
        )
