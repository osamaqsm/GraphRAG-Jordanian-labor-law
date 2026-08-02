from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalHit:
    """
    One ontology concept, paragraph or article returned
    by the retrieval pipeline.
    """

    uri: str
    local_name: str
    node_kind: str

    labels_ar: list[str] = field(
        default_factory=list
    )

    labels_en: list[str] = field(
        default_factory=list
    )

    article_number: int | None = None
    text_preview: str = ""

    # Direct vector search information
    vector_rank: int | None = None
    vector_distance: float | None = None

    # Direct BM25 information
    bm25_rank: int | None = None
    bm25_score: float | None = None

    # Direct RRF score
    fused_score: float = 0.0

    # New graph-aware ranking components
    direct_score: float = 0.0
    graph_score: float = 0.0
    paragraph_score: float = 0.0
    final_score: float = 0.0

    graph_supported: bool = False

    support_paths: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        numeric_fields = (
            "vector_distance",
            "bm25_score",
            "fused_score",
            "direct_score",
            "graph_score",
            "paragraph_score",
            "final_score",
        )

        for field_name in numeric_fields:
            value = result[field_name]

            if value is not None:
                result[field_name] = round(
                    float(value),
                    8,
                )

        for path in result["support_paths"]:
            if "score" in path:
                path["score"] = round(
                    float(path["score"]),
                    8,
                )

        return result


@dataclass(frozen=True, slots=True)
class RetrievalPreview:
    """
    Full graph-aware retrieval output.
    """

    question: str

    embedding_model: str
    embedding_dimensions: int
    embedding_input_tokens: int

    # Concepts returned directly by filtered retrieval
    concept_hits: list[RetrievalHit]

    # Concepts discovered by traversing the KG
    expanded_concept_hits: list[RetrievalHit]

    # Final graph-first legal article ranking
    article_hits: list[RetrievalHit]

    # Diagnostic broad retrieval results
    raw_mixed_hits: list[RetrievalHit]

    def to_dict(self) -> dict[str, Any]:
        graph_paths: list[dict[str, Any]] = []

        for article in self.article_hits:
            graph_paths.extend(
                article.support_paths
            )

        return {
            "question": self.question,
            "embedding": {
                "model": self.embedding_model,
                "dimensions": (
                    self.embedding_dimensions
                ),
                "input_tokens": (
                    self.embedding_input_tokens
                ),
            },
            "concept_hits": [
                hit.to_dict()
                for hit in self.concept_hits
            ],
            "expanded_concept_hits": [
                hit.to_dict()
                for hit in self.expanded_concept_hits
            ],
            "article_hits": [
                hit.to_dict()
                for hit in self.article_hits
            ],
            "graph_paths": graph_paths,
            "raw_mixed_hits": [
                hit.to_dict()
                for hit in self.raw_mixed_hits
            ],
        }