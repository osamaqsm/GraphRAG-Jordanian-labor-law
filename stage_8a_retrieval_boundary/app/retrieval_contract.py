from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RETRIEVAL_CONTRACT_VERSION = "retrieval.v1"


class FrozenContractModel(BaseModel):
    """Versioned JSON model passed from retrieval to later pipeline stages."""

    model_config = ConfigDict(extra="forbid")


class RetrievalRequestV1(FrozenContractModel):
    question: str = Field(min_length=3)
    include_debug: bool = False


class RetrievalDecisionV1(FrozenContractModel):
    behavior: Literal["retrieve", "clarify", "abstain"]
    reason: str = ""
    clarification_question_ar: str = ""
    planner_used: bool = False
    planner_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalEmbeddingV1(FrozenContractModel):
    model: str
    dimensions: int = Field(ge=0)
    input_tokens: int = Field(ge=0)


class RetrievalEvidenceV1(FrozenContractModel):
    uri: str
    local_name: str
    node_kind: str
    labels_ar: list[str]
    labels_en: list[str]
    article_number: int | None = None
    text: str = ""
    final_score: float = 0.0
    fused_score: float = 0.0
    direct_score: float = 0.0
    graph_score: float = 0.0
    paragraph_score: float = 0.0
    graph_supported: bool = False
    support_paths: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDiagnosticsV1(FrozenContractModel):
    article_numbers: list[int] = Field(default_factory=list)
    concept_local_names: list[str] = Field(default_factory=list)
    graph_supported_articles: list[int] = Field(default_factory=list)
    concept_count: int = Field(default=0, ge=0)
    expanded_concept_count: int = Field(default=0, ge=0)
    article_count: int = Field(default=0, ge=0)


class RetrievalResultV1(FrozenContractModel):
    """Stable retrieval-only boundary. It deliberately has no answer field."""

    schema_version: Literal["retrieval.v1"] = RETRIEVAL_CONTRACT_VERSION
    status: Literal["success"] = "success"
    question: str
    decision: RetrievalDecisionV1
    embedding: RetrievalEmbeddingV1
    articles: list[RetrievalEvidenceV1] = Field(default_factory=list)
    concepts: list[RetrievalEvidenceV1] = Field(default_factory=list)
    expanded_concepts: list[RetrievalEvidenceV1] = Field(default_factory=list)
    diagnostics: RetrievalDiagnosticsV1
    elapsed_ms: int = Field(ge=0)
    debug: dict[str, Any] | None = None
