from __future__ import annotations

from typing import Any

from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances

from app.config import Settings


def _exact_text_property(name: str, description: str) -> Property:
    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    )


def _searchable_text_property(name: str, description: str) -> Property:
    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT,
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    )


def _searchable_text_array_property(name: str, description: str) -> Property:
    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT_ARRAY,
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    )


def _exact_text_array_property(name: str, description: str) -> Property:
    return Property(
        name=name,
        description=description,
        data_type=DataType.TEXT_ARRAY,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    )


def create_node_collection(client: Any, settings: Settings) -> None:
    client.collections.create(
        name=settings.weaviate_node_collection,
        description=(
            "RDF URI resources for the final Jordanian Labor Law GraphRAG. "
            "Only retrievalEligible=true semantic concept individuals may be "
            "used by vector/BM25 concept linking."
        ),
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            )
        ),
        properties=[
            _exact_text_property("uri", "Complete RDF URI."),
            _exact_text_property("localName", "RDF local name."),
            _exact_text_property("nodeKind", "High-level RDF/legal node kind."),
            _exact_text_array_property("rdfTypes", "Complete rdf:type URIs."),
            _searchable_text_array_property("labelsAr", "Arabic rdfs:label values."),
            _searchable_text_array_property("labelsEn", "English rdfs:label values."),
            _searchable_text_array_property("labelsOther", "Other rdfs:label values."),
            _searchable_text_array_property("aliasesAr", "Arabic skos:altLabel values."),
            _searchable_text_array_property("aliasesEn", "English skos:altLabel values."),
            _searchable_text_array_property("aliasesOther", "Other skos:altLabel values."),
            _searchable_text_array_property("commentsAr", "Arabic rdfs:comment values."),
            _searchable_text_array_property("commentsEn", "English rdfs:comment values."),
            _searchable_text_array_property("commentsOther", "Other rdfs:comment values."),
            Property(
                name="articleNumber",
                description="Numeric legal article number when nodeKind=Article.",
                data_type=DataType.INT,
                index_filterable=True,
                index_range_filters=True,
            ),
            Property(
                name="retrievalEligible",
                description=(
                    "True only for semantic concept individuals eligible for "
                    "ontology concept linking. Article/Paragraph/Law/schema nodes are false."
                ),
                data_type=DataType.BOOL,
                index_filterable=True,
            ),
            _searchable_text_property(
                "searchableText",
                "Combined semantic labels, aliases, types and descriptive text.",
            ),
        ],
    )


def create_edge_collection(client: Any, settings: Settings) -> None:
    client.collections.create(
        name=settings.weaviate_edge_collection,
        description="All RDF triples for relation-aware graph traversal.",
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.none()
        ),
        properties=[
            _exact_text_property("sourceUri", "RDF subject URI."),
            _exact_text_property("predicateUri", "Complete RDF predicate URI."),
            _exact_text_property("predicateLocalName", "RDF predicate local name."),
            _exact_text_property("objectKind", "uri or literal."),
            _exact_text_property("targetUri", "RDF object URI for URI edges."),
            _searchable_text_property("literalValue", "Literal RDF object value."),
            _exact_text_property("literalLanguage", "Literal language tag."),
            _exact_text_property("literalDatatype", "Literal datatype URI."),
        ],
    )


def ensure_collections(client: Any, settings: Settings, reset: bool = False) -> dict[str, Any]:
    names = (settings.weaviate_node_collection, settings.weaviate_edge_collection)
    deleted: list[str] = []
    created: list[str] = []
    retained: list[str] = []

    if reset:
        for name in names:
            if client.collections.exists(name):
                client.collections.delete(name)
                deleted.append(name)

    if client.collections.exists(settings.weaviate_node_collection):
        retained.append(settings.weaviate_node_collection)
    else:
        create_node_collection(client, settings)
        created.append(settings.weaviate_node_collection)

    if client.collections.exists(settings.weaviate_edge_collection):
        retained.append(settings.weaviate_edge_collection)
    else:
        create_edge_collection(client, settings)
        created.append(settings.weaviate_edge_collection)

    return {"deleted": deleted, "created": created, "retained": retained}


def inspect_collection(client: Any, collection_name: str) -> dict[str, Any]:
    configuration = client.collections.use(collection_name).config.get()
    return {
        "name": collection_name,
        "description": configuration.description,
        "properties": [
            {
                "name": prop.name,
                "data_type": str(prop.data_type),
                "tokenization": str(prop.tokenization) if prop.tokenization else None,
                "index_filterable": prop.index_filterable,
                "index_searchable": prop.index_searchable,
            }
            for prop in configuration.properties
        ],
    }
