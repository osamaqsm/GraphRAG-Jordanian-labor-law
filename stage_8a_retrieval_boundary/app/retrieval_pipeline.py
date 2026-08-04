from __future__ import annotations

import os
import time
from typing import Any

import weaviate
from openai import OpenAI
from weaviate.classes.init import Auth

from app.config import Settings, get_settings
from app.legal_question_analysis import analyze_legal_question
from app.retrieval_contract import (
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalEvidenceV1,
    RetrievalResultV1,
)
from app.retrieval_service import RetrievalService


def _setting(settings: Settings, name: str, env_name: str, default: Any) -> Any:
    value = getattr(settings, name, None)
    if value is not None and value != "":
        return value
    return os.getenv(env_name, default)


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _support_paths(value: Any) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for item in value or []:
        dumped = _dump(item)
        if isinstance(dumped, dict):
            paths.append(dumped)
    return paths


def _evidence(hit: Any) -> RetrievalEvidenceV1:
    return RetrievalEvidenceV1(
        uri=str(getattr(hit, "uri", "") or ""),
        local_name=str(getattr(hit, "local_name", "") or ""),
        node_kind=str(getattr(hit, "node_kind", "") or ""),
        labels_ar=[str(v) for v in getattr(hit, "labels_ar", []) or []],
        labels_en=[str(v) for v in getattr(hit, "labels_en", []) or []],
        article_number=getattr(hit, "article_number", None),
        text=str(getattr(hit, "text_preview", "") or ""),
        final_score=float(getattr(hit, "final_score", 0.0) or 0.0),
        fused_score=float(getattr(hit, "fused_score", 0.0) or 0.0),
        direct_score=float(getattr(hit, "direct_score", 0.0) or 0.0),
        graph_score=float(getattr(hit, "graph_score", 0.0) or 0.0),
        paragraph_score=float(getattr(hit, "paragraph_score", 0.0) or 0.0),
        graph_supported=bool(getattr(hit, "graph_supported", False)),
        support_paths=_support_paths(getattr(hit, "support_paths", [])),
    )


class RetrievalOnlyPipeline:
    """Run routing and retrieval only. No answer generator is imported here."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        weaviate_client: Any | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_weaviate_client = weaviate_client is None
        self.weaviate_client = weaviate_client or self._connect_weaviate()
        self.openai_client = openai_client or OpenAI(
            api_key=str(_setting(
                self.settings,
                "openai_api_key",
                "OPENAI_API_KEY",
                "",
            )),
            timeout=float(_setting(
                self.settings,
                "openai_timeout_seconds",
                "OPENAI_TIMEOUT_SECONDS",
                120,
            )),
            max_retries=int(_setting(
                self.settings,
                "openai_max_retries",
                "OPENAI_MAX_RETRIES",
                3,
            )),
        )
        self.service = RetrievalService(
            client=self.weaviate_client,
            settings=self.settings,
        )

    def _connect_weaviate(self) -> Any:
        http_host = str(_setting(
            self.settings,
            "weaviate_http_host",
            "WEAVIATE_HTTP_HOST",
            "weaviate",
        ))
        grpc_host = str(_setting(
            self.settings,
            "weaviate_grpc_host",
            "WEAVIATE_GRPC_HOST",
            http_host,
        ))
        http_port = int(_setting(
            self.settings,
            "weaviate_http_port",
            "WEAVIATE_HTTP_PORT",
            8080,
        ))
        grpc_port = int(_setting(
            self.settings,
            "weaviate_grpc_port",
            "WEAVIATE_GRPC_PORT",
            50051,
        ))
        http_secure = _truthy(os.getenv("WEAVIATE_HTTP_SECURE"), default=False)
        grpc_secure = _truthy(os.getenv("WEAVIATE_GRPC_SECURE"), default=False)
        api_key = str(_setting(
            self.settings,
            "weaviate_api_key",
            "WEAVIATE_API_KEY",
            "",
        )).strip()

        kwargs: dict[str, Any] = {
            "http_host": http_host,
            "http_port": http_port,
            "http_secure": http_secure,
            "grpc_host": grpc_host,
            "grpc_port": grpc_port,
            "grpc_secure": grpc_secure,
        }
        if api_key:
            kwargs["auth_credentials"] = Auth.api_key(api_key)
        return weaviate.connect_to_custom(**kwargs)

    def close(self) -> None:
        if self._owns_weaviate_client:
            self.weaviate_client.close()

    def __enter__(self) -> "RetrievalOnlyPipeline":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def retrieve(
        self,
        question: str,
        *,
        include_debug: bool = False,
    ) -> RetrievalResultV1:
        question = question.strip()
        if len(question) < 3:
            raise ValueError("Question must contain at least three characters.")

        started = time.perf_counter()

        # Route once. The same merged analysis is passed into RetrievalService,
        # so Stage 8-A does not duplicate or disagree with the planner call.
        analysis = analyze_legal_question(question)
        query_plan = self.service.query_planner.plan(question)
        analysis = self.service.query_planner.merge_with_analysis(
            analysis=analysis,
            plan=query_plan,
        )

        embedding_model = str(_setting(
            self.settings,
            "openai_embedding_model",
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ))
        vector: list[float] = []
        input_tokens = 0
        preview: Any | None = None

        if analysis.behavior == "retrieve":
            embedding_response = self.openai_client.embeddings.create(
                model=embedding_model,
                input=[question],
                encoding_format="float",
            )
            vector = list(embedding_response.data[0].embedding)
            usage = getattr(embedding_response, "usage", None)
            input_tokens = int(
                getattr(usage, "prompt_tokens", 0)
                or getattr(usage, "total_tokens", 0)
                or 0
            )
            preview = self.service.preview(
                question,
                vector,
                embedding_model=embedding_model,
                embedding_dimensions=len(vector),
                embedding_input_tokens=input_tokens,
                analysis=analysis,
            )

        article_hits = list(getattr(preview, "article_hits", []) or [])
        concept_hits = list(getattr(preview, "concept_hits", []) or [])
        expanded_hits = list(
            getattr(preview, "expanded_concept_hits", []) or []
        )

        articles = [_evidence(hit) for hit in article_hits]
        concepts = [_evidence(hit) for hit in concept_hits]
        expanded_concepts = [_evidence(hit) for hit in expanded_hits]

        article_numbers = [
            article.article_number
            for article in articles
            if article.article_number is not None
        ]
        graph_supported_articles = [
            article.article_number
            for article in articles
            if article.article_number is not None and article.graph_supported
        ]

        debug: dict[str, Any] | None = None
        if include_debug:
            debug = {
                "analysis": {
                    "normalized_question": analysis.normalized_question,
                    "issue_ids": list(analysis.issue_ids),
                    "planner_queries": list(analysis.planner_queries),
                    "planner_issue_labels": list(analysis.planner_issue_labels),
                    "max_final_articles": analysis.max_final_articles,
                },
                "query_plan": _dump(query_plan) if query_plan is not None else None,
                "raw_preview": _dump(preview) if preview is not None else None,
            }

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))

        return RetrievalResultV1(
            question=question,
            decision=RetrievalDecisionV1(
                behavior=analysis.behavior,
                reason=analysis.behavior_reason,
                clarification_question_ar=analysis.clarification_question,
                planner_used=analysis.planner_used,
                planner_confidence=analysis.planner_confidence,
            ),
            embedding=RetrievalEmbeddingV1(
                model=embedding_model,
                dimensions=len(vector),
                input_tokens=input_tokens,
            ),
            articles=articles,
            concepts=concepts,
            expanded_concepts=expanded_concepts,
            diagnostics=RetrievalDiagnosticsV1(
                article_numbers=article_numbers,
                concept_local_names=[
                    concept.local_name
                    for concept in concepts
                    if concept.local_name
                ],
                graph_supported_articles=graph_supported_articles,
                concept_count=len(concepts),
                expanded_concept_count=len(expanded_concepts),
                article_count=len(articles),
            ),
            elapsed_ms=elapsed_ms,
            debug=debug,
        )
