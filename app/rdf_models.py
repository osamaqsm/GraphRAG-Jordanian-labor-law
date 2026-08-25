from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RdfNodeRecord:
    uri: str
    local_name: str
    node_kind: str
    rdf_types: tuple[str, ...] = field(default_factory=tuple)

    labels_ar: tuple[str, ...] = field(default_factory=tuple)
    labels_en: tuple[str, ...] = field(default_factory=tuple)
    labels_other: tuple[str, ...] = field(default_factory=tuple)

    aliases_ar: tuple[str, ...] = field(default_factory=tuple)
    aliases_en: tuple[str, ...] = field(default_factory=tuple)
    aliases_other: tuple[str, ...] = field(default_factory=tuple)

    comments_ar: tuple[str, ...] = field(default_factory=tuple)
    comments_en: tuple[str, ...] = field(default_factory=tuple)
    comments_other: tuple[str, ...] = field(default_factory=tuple)

    article_number: int | None = None
    retrieval_eligible: bool = False
    searchable_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RdfEdgeRecord:
    source_uri: str
    predicate_uri: str
    predicate_local_name: str
    object_kind: Literal["uri", "literal"]
    target_uri: str | None = None
    literal_value: str | None = None
    literal_language: str | None = None
    literal_datatype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KgInspectionReport:
    ttl_path: str
    file_size_bytes: int
    triple_count: int
    node_count: int
    edge_count: int
    article_count: int
    paragraph_count: int
    definition_count: int
    ontology_class_count: int
    object_property_count: int
    datatype_property_count: int
    retrieval_eligible_concept_count: int
    semantically_reachable_article_count: int
    article_numbers: tuple[int, ...]
    missing_article_numbers: tuple[int, ...]
    duplicate_article_numbers: tuple[int, ...]
    unreachable_article_numbers: tuple[int, ...]
    articles_without_arabic_text: tuple[str, ...]
    articles_without_labels: tuple[str, ...]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
