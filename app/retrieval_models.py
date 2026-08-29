from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalHit:
    """One semantic concept or graph-derived Article candidate."""

    uri: str
    local_name: str
    node_kind: str
    labels_ar: list[str] = field(default_factory=list)
    labels_en: list[str] = field(default_factory=list)
    aliases_ar: list[str] = field(default_factory=list)
    aliases_en: list[str] = field(default_factory=list)
    article_number: int | None = None
    text_preview: str = ""

    # Concept-linking evidence.
    vector_rank: int | None = None
    vector_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    hint_rank: int | None = None
    hint_score: float = 0.0
    fused_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    planner_hint_score: float = 0.0
    specificity_score: float = 0.0
    seed_score: float = 0.0

    # Graph-derived article evidence.
    graph_score: float = 0.0
    issue_relevance: float = 0.0
    final_score: float = 0.0
    graph_supported: bool = False
    support_paths: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        numeric_fields = (
            "vector_distance",
            "bm25_score",
            "hint_score",
            "fused_score",
            "semantic_score",
            "lexical_score",
            "planner_hint_score",
            "specificity_score",
            "seed_score",
            "graph_score",
            "issue_relevance",
            "final_score",
        )
        for field_name in numeric_fields:
            value = result[field_name]
            if value is not None:
                result[field_name] = round(float(value), 8)
        for path in result["support_paths"]:
            if "score" in path:
                path["score"] = round(float(path["score"]), 8)
        return result


@dataclass(frozen=True, slots=True)
class RetrievalPreview:
    question: str
    embedding_model: str
    embedding_dimensions: int
    embedding_input_tokens: int
    concept_hits: list[RetrievalHit]
    expanded_concept_hits: list[RetrievalHit]
    article_candidates: list[RetrievalHit]
    article_hits: list[RetrievalHit]
    selected_seeds: list[RetrievalHit] = field(default_factory=list)
    issue_debug: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_input_tokens": self.embedding_input_tokens,
            "concept_hits": [item.to_dict() for item in self.concept_hits],
            "expanded_concept_hits": [item.to_dict() for item in self.expanded_concept_hits],
            "article_candidates": [item.to_dict() for item in self.article_candidates],
            "article_hits": [item.to_dict() for item in self.article_hits],
            "selected_seeds": [item.to_dict() for item in self.selected_seeds],
            "issue_debug": self.issue_debug,
        }
