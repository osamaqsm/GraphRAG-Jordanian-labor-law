from __future__ import annotations

from typing import Any

from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)

from app.config import Settings


def _exact_text_property(
    name: str,
    description: str,
) -> Property:
    """
    Create a text property intended for exact filtering.

    FIELD tokenization stores the complete string as one field.
    This is suitable for URIs, identifiers and controlled values.
    """

    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    )


def _searchable_text_property(
    name: str,
    description: str,
) -> Property:
    """
    Create a text property suitable for BM25 keyword search.
    """

    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    )


def _searchable_text_array_property(
    name: str,
    description: str,
) -> Property:
    """
    Create a searchable array of text values.
    """

    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT_ARRAY,
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    )


def _exact_text_array_property(
    name: str,
    description: str,
) -> Property:
    """
    Create a text-array property intended for exact filters.
    """

    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT_ARRAY,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    )


def create_node_collection(
    client: Any,
    settings: Settings,
) -> None:
    """
    Create the collection that stores RDF URI resources.

    Examples:
        Articles
        Paragraphs
        Definitions
        Rights
        Obligations
        Violations
        Conditions
        Consequences
        Actors
        OWL classes and properties

    Vectors will be generated later using:
        text-embedding-3-small

    Therefore, Weaviate is configured to accept
    self-provided vectors.
    """

    client.collections.create(
        name=settings.weaviate_node_collection,
        description=(
            "URI resources extracted from the Jordanian "
            "Labour Law ontology and knowledge graph."
        ),
        vector_config=(
    Configure.Vectors.self_provided(
        vector_index_config=(
            Configure.VectorIndex.hnsw(
                distance_metric=(
                    VectorDistances.COSINE
                )
            )
        )
    )
),
        properties=[
            _exact_text_property(
                name="uri",
                description=(
                    "Complete globally unique RDF URI."
                ),
            ),
            _exact_text_property(
                name="localName",
                description=(
                    "Final fragment or path component "
                    "of the RDF URI."
                ),
            ),
            _exact_text_property(
                name="nodeKind",
                description=(
                    "High-level resource type such as "
                    "Article, Paragraph or Definition."
                ),
            ),
            _exact_text_array_property(
                name="rdfTypes",
                description=(
                    "Complete rdf:type URIs assigned "
                    "to the resource."
                ),
            ),
            _searchable_text_array_property(
                name="labelsAr",
                description="Arabic rdfs:label values.",
            ),
            _searchable_text_array_property(
                name="labelsEn",
                description="English rdfs:label values.",
            ),
            _searchable_text_array_property(
                name="labelsOther",
                description=(
                    "Labels without Arabic or English "
                    "language tags."
                ),
            ),
            _searchable_text_array_property(
                name="commentsAr",
                description=(
                    "Arabic rdfs:comment values, including "
                    "complete legal article content."
                ),
            ),
            _searchable_text_array_property(
                name="commentsEn",
                description="English rdfs:comment values.",
            ),
            _searchable_text_array_property(
                name="commentsOther",
                description=(
                    "Comments without Arabic or English "
                    "language tags."
                ),
            ),
            Property(
                name="articleNumber",
                description=(
                    "Numeric article number when the "
                    "resource is a legal article."
                ),
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filters=True,
            ),
            _searchable_text_property(
                name="searchableText",
                description=(
                    "Combined labels, types and legal "
                    "text used for BM25 and embedding."
                ),
            ),
        ],
    )


def create_edge_collection(
    client: Any,
    settings: Settings,
) -> None:
    """
    Create the collection that preserves every RDF triple.

    URI triple:
        sourceUri -> predicateUri -> targetUri

    Literal triple:
        sourceUri -> predicateUri -> literalValue

    We do not need semantic vectors for edge objects,
    but self-provided vector configuration allows objects
    to exist without an internal vectorizer module.
    """

    client.collections.create(
        name=settings.weaviate_edge_collection,
        description=(
            "RDF triples extracted from the Jordanian "
            "Labour Law ontology and knowledge graph."
        ),
        vector_config=(
    Configure.Vectors.self_provided(
        vector_index_config=(
            Configure.VectorIndex.none()
        )
    )
),
        properties=[
            _exact_text_property(
                name="sourceUri",
                description=(
                    "URI of the RDF triple subject."
                ),
            ),
            _exact_text_property(
                name="predicateUri",
                description=(
                    "Complete URI of the RDF predicate."
                ),
            ),
            _exact_text_property(
                name="predicateLocalName",
                description=(
                    "Local name of the RDF predicate."
                ),
            ),
            _exact_text_property(
                name="objectKind",
                description=(
                    "Whether the RDF object is a URI "
                    "or a literal."
                ),
            ),
            _exact_text_property(
                name="targetUri",
                description=(
                    "URI of the RDF object when "
                    "objectKind is uri."
                ),
            ),
            _searchable_text_property(
                name="literalValue",
                description=(
                    "Literal RDF value when objectKind "
                    "is literal."
                ),
            ),
            _exact_text_property(
                name="literalLanguage",
                description=(
                    "Language tag of a literal, such "
                    "as ar or en."
                ),
            ),
            _exact_text_property(
                name="literalDatatype",
                description=(
                    "Datatype URI assigned to a literal."
                ),
            ),
        ],
    )


def ensure_collections(
    client: Any,
    settings: Settings,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Create both collections.

    When reset is False:
        Existing collections are retained.

    When reset is True:
        Existing node and edge collections are deleted
        and recreated.

    Warning:
        reset=True deletes all objects currently stored
        inside these two collections.
    """

    collection_names = (
        settings.weaviate_node_collection,
        settings.weaviate_edge_collection,
    )

    deleted: list[str] = []
    created: list[str] = []
    retained: list[str] = []

    if reset:
        for collection_name in collection_names:
            if client.collections.exists(
                collection_name
            ):
                client.collections.delete(
                    collection_name
                )
                deleted.append(collection_name)

    if client.collections.exists(
        settings.weaviate_node_collection
    ):
        retained.append(
            settings.weaviate_node_collection
        )
    else:
        create_node_collection(
            client=client,
            settings=settings,
        )
        created.append(
            settings.weaviate_node_collection
        )

    if client.collections.exists(
        settings.weaviate_edge_collection
    ):
        retained.append(
            settings.weaviate_edge_collection
        )
    else:
        create_edge_collection(
            client=client,
            settings=settings,
        )
        created.append(
            settings.weaviate_edge_collection
        )

    return {
        "deleted": deleted,
        "created": created,
        "retained": retained,
    }


def inspect_collection(
    client: Any,
    collection_name: str,
) -> dict[str, Any]:
    """
    Return a simple readable summary of one collection.
    """

    collection = client.collections.use(
        collection_name
    )

    configuration = collection.config.get()

    return {
        "name": collection_name,
        "description": configuration.description,
        "properties": [
            {
                "name": prop.name,
                "data_type": str(
                    prop.data_type
                ),
                "tokenization": (
                    str(prop.tokenization)
                    if prop.tokenization
                    else None
                ),
                "index_filterable": (
                    prop.index_filterable
                ),
                "index_searchable": (
                    prop.index_searchable
                ),
            }
            for prop in configuration.properties
        ],
    }