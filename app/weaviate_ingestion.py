from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Sequence, TypeVar

from weaviate.util import generate_uuid5

from app.config import Settings
from app.openai_service import OpenAIService
from app.rdf_models import (
    RdfEdgeRecord,
    RdfNodeRecord,
)


T = TypeVar("T")

ProgressCallback = Callable[
    [str],
    None,
]


@dataclass(frozen=True, slots=True)
class KgIngestionSummary:
    """
    Summary returned after KG ingestion.
    """

    requested_nodes: int
    inserted_nodes: int

    requested_edges: int
    inserted_edges: int

    embedding_input_tokens: int
    embedding_dimensions: int

    node_failures: int
    edge_failures: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunked(
    values: Sequence[T],
    size: int,
) -> Iterator[Sequence[T]]:
    """
    Yield fixed-size slices from a sequence.
    """

    if size < 1:
        raise ValueError(
            "Chunk size must be at least one."
        )

    for start in range(
        0,
        len(values),
        size,
    ):
        yield values[
            start:start + size
        ]


def node_properties(
    record: RdfNodeRecord,
) -> dict[str, Any]:
    """
    Convert an RDF node record to Weaviate properties.

    Empty optional properties are omitted rather than
    inserted as null values.
    """

    properties: dict[str, Any] = {
        "uri": record.uri,
        "localName": record.local_name,
        "nodeKind": record.node_kind,
        "searchableText": record.searchable_text,
        "retrievalEligible": record.retrieval_eligible,
    }

    optional_arrays = {
        "rdfTypes": record.rdf_types,
        "labelsAr": record.labels_ar,
        "labelsEn": record.labels_en,
        "labelsOther": record.labels_other,
        "aliasesAr": record.aliases_ar,
        "aliasesEn": record.aliases_en,
        "aliasesOther": record.aliases_other,
        "commentsAr": record.comments_ar,
        "commentsEn": record.comments_en,
        "commentsOther": record.comments_other,
    }

    for property_name, values in optional_arrays.items():
        if values:
            properties[property_name] = list(values)

    if record.article_number is not None:
        properties["articleNumber"] = (
            record.article_number
        )

    return properties


def edge_properties(
    record: RdfEdgeRecord,
) -> dict[str, Any]:
    """
    Convert one RDF triple to Weaviate properties.
    """

    properties: dict[str, Any] = {
        "sourceUri": record.source_uri,
        "predicateUri": record.predicate_uri,
        "predicateLocalName": (
            record.predicate_local_name
        ),
        "objectKind": record.object_kind,
    }

    if record.target_uri is not None:
        properties["targetUri"] = (
            record.target_uri
        )

    if record.literal_value is not None:
        properties["literalValue"] = (
            record.literal_value
        )

    if record.literal_language is not None:
        properties["literalLanguage"] = (
            record.literal_language
        )

    if record.literal_datatype is not None:
        properties["literalDatatype"] = (
            record.literal_datatype
        )

    return properties


def node_uuid(
    record: RdfNodeRecord,
):
    """
    Generate a stable UUID from the RDF URI.
    """

    return generate_uuid5(
        {
            "resource_type": "rdf_node",
            "uri": record.uri,
        }
    )


def edge_uuid(
    record: RdfEdgeRecord,
):
    """
    Generate a stable UUID from all parts of the RDF triple.
    """

    return generate_uuid5(
        {
            "resource_type": "rdf_edge",
            "source_uri": record.source_uri,
            "predicate_uri": record.predicate_uri,
            "object_kind": record.object_kind,
            "target_uri": record.target_uri,
            "literal_value": record.literal_value,
            "literal_language": (
                record.literal_language
            ),
            "literal_datatype": (
                record.literal_datatype
            ),
        }
    )


def collection_count(
    client: Any,
    collection_name: str,
) -> int:
    """
    Return the total object count in a collection.
    """

    collection = client.collections.use(
        collection_name
    )

    result = collection.aggregate.over_all(
        total_count=True
    )

    return int(
        result.total_count or 0
    )


def ingest_nodes(
    client: Any,
    settings: Settings,
    service: OpenAIService,
    records: list[RdfNodeRecord],
    progress: ProgressCallback | None = None,
) -> tuple[int, int, int]:
    """
    Embed and insert all RDF nodes.

    Returns:
        embedding token count
        embedding dimensions
        failed object count
    """

    collection = client.collections.use(
        settings.weaviate_node_collection
    )

    total_tokens = 0
    embedding_dimensions = 0
    processed = 0

    with collection.batch.fixed_size(
        batch_size=(
            settings.weaviate_node_batch_size
        ),
        concurrent_requests=2,
    ) as batch:
        for record_batch in chunked(
            records,
            settings.embedding_batch_size,
        ):
            embedding_result = service.embed_texts(
                [
                    record.searchable_text
                    for record in record_batch
                ]
            )

            total_tokens += (
                embedding_result.input_tokens
            )

            embedding_dimensions = (
                embedding_result.dimensions
            )

            for record, vector in zip(
                record_batch,
                embedding_result.vectors,
                strict=True,
            ):
                batch.add_object(
                    properties=node_properties(
                        record
                    ),
                    uuid=node_uuid(record),
                    vector=vector,
                )

            processed += len(record_batch)

            if progress is not None:
                progress(
                    "Embedded and queued "
                    f"{processed}/{len(records)} nodes."
                )

            if batch.number_errors > 10:
                raise RuntimeError(
                    "Node ingestion stopped because more "
                    "than 10 batch errors occurred."
                )

    failures = list(
        collection.batch.failed_objects
    )

    if failures:
        first_failure = failures[0]

        raise RuntimeError(
            f"{len(failures)} node objects failed. "
            f"First failure: {first_failure}"
        )

    return (
        total_tokens,
        embedding_dimensions,
        len(failures),
    )


def ingest_edges(
    client: Any,
    settings: Settings,
    records: list[RdfEdgeRecord],
    progress: ProgressCallback | None = None,
) -> int:
    """
    Insert all RDF triples without vectors.
    """

    collection = client.collections.use(
        settings.weaviate_edge_collection
    )

    processed = 0

    with collection.batch.fixed_size(
        batch_size=(
            settings.weaviate_edge_batch_size
        ),
        concurrent_requests=2,
    ) as batch:
        for record in records:
            batch.add_object(
                properties=edge_properties(
                    record
                ),
                uuid=edge_uuid(record),
            )

            processed += 1

            if (
                progress is not None
                and processed % 250 == 0
            ):
                progress(
                    "Queued "
                    f"{processed}/{len(records)} edges."
                )

            if batch.number_errors > 10:
                raise RuntimeError(
                    "Edge ingestion stopped because more "
                    "than 10 batch errors occurred."
                )

    failures = list(
        collection.batch.failed_objects
    )

    if failures:
        first_failure = failures[0]

        raise RuntimeError(
            f"{len(failures)} edge objects failed. "
            f"First failure: {first_failure}"
        )

    if progress is not None:
        progress(
            f"Queued {processed}/{len(records)} edges."
        )

    return len(failures)


def ingest_kg_records(
    client: Any,
    settings: Settings,
    service: OpenAIService,
    node_records: list[RdfNodeRecord],
    edge_records: list[RdfEdgeRecord],
    progress: ProgressCallback | None = None,
) -> KgIngestionSummary:
    """
    Ingest all KG nodes and edges and verify final counts.
    """

    (
        embedding_tokens,
        embedding_dimensions,
        node_failures,
    ) = ingest_nodes(
        client=client,
        settings=settings,
        service=service,
        records=node_records,
        progress=progress,
    )

    edge_failures = ingest_edges(
        client=client,
        settings=settings,
        records=edge_records,
        progress=progress,
    )

    inserted_nodes = collection_count(
        client,
        settings.weaviate_node_collection,
    )

    inserted_edges = collection_count(
        client,
        settings.weaviate_edge_collection,
    )

    if inserted_nodes != len(node_records):
        raise RuntimeError(
            "Node-count validation failed. "
            f"Expected {len(node_records)}, "
            f"found {inserted_nodes}."
        )

    if inserted_edges != len(edge_records):
        raise RuntimeError(
            "Edge-count validation failed. "
            f"Expected {len(edge_records)}, "
            f"found {inserted_edges}."
        )

    return KgIngestionSummary(
        requested_nodes=len(node_records),
        inserted_nodes=inserted_nodes,
        requested_edges=len(edge_records),
        inserted_edges=inserted_edges,
        embedding_input_tokens=(
            embedding_tokens
        ),
        embedding_dimensions=(
            embedding_dimensions
        ),
        node_failures=node_failures,
        edge_failures=edge_failures,
    )