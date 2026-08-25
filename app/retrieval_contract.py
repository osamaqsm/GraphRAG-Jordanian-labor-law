from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RETRIEVAL_CONTRACT_VERSION = "retrieval.v2"


class FrozenContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalRequestV2(FrozenContractModel):
    question: str = Field(min_length=3)
    include_debug: bool = False


class RetrievalDecisionV2(FrozenContractModel):
    behavior: Literal["retrieve", "abstain"]
    reason: str = ""
    planner_used: bool = True
    planner_verified: bool = False
    planner_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalEmbeddingV2(FrozenContractModel):
    model: str
    dimensions: int = Field(ge=0)
    input_tokens: int = Field(ge=0)


class RetrievalEvidenceV2(FrozenContractModel):
    uri: str
    local_name: str
    node_kind: str
    labels_ar: list[str]
    labels_en: list[str]
    aliases_ar: list[str] = Field(default_factory=list)
    aliases_en: list[str] = Field(default_factory=list)
    article_number: int | None = None
    text: str = ""

    fused_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    planner_hint_score: float = 0.0
    specificity_score: float = 0.0
    seed_score: float = 0.0
    graph_score: float = 0.0
    issue_relevance: float = 0.0
    final_score: float = 0.0
    graph_supported: bool = False
    support_paths: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalDiagnosticsV2(FrozenContractModel):
    article_numbers: list[int] = Field(default_factory=list)
    graph_candidate_article_numbers: list[int] = Field(default_factory=list)
    concept_local_names: list[str] = Field(default_factory=list)
    selected_seed_local_names: list[str] = Field(default_factory=list)
    graph_supported_articles: list[int] = Field(default_factory=list)
    concept_count: int = Field(default=0, ge=0)
    expanded_concept_count: int = Field(default=0, ge=0)
    graph_candidate_count: int = Field(default=0, ge=0)
    article_count: int = Field(default=0, ge=0)


class RetrievalResultV2(FrozenContractModel):
    """Stable graph-only retrieval boundary consumed by generation."""

    schema_version: Literal["retrieval.v2"] = RETRIEVAL_CONTRACT_VERSION
    status: Literal["success"] = "success"
    question: str
    decision: RetrievalDecisionV2
    embedding: RetrievalEmbeddingV2
    articles: list[RetrievalEvidenceV2] = Field(default_factory=list)
    concepts: list[RetrievalEvidenceV2] = Field(default_factory=list)
    expanded_concepts: list[RetrievalEvidenceV2] = Field(default_factory=list)
    diagnostics: RetrievalDiagnosticsV2
    elapsed_ms: int = Field(ge=0)
    debug: dict[str, Any] | None = None
