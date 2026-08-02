from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from app.rdf_models import (
    KgInspectionReport,
    RdfEdgeRecord,
    RdfNodeRecord,
)


# Base namespace used by your ontology and KG.
LAW = Namespace(
    "http://example.org/jordan-labor-law#"
)


# Article URIs follow this structure:
# http://example.org/jordan-labor-law#article_46
ARTICLE_URI_PATTERN = re.compile(
    r"(?:^|[#/])article_(\d+)$"
)


def local_name(value: URIRef | str) -> str:
    """
    Return the fragment or final path component of a URI.

    Examples:
        http://example.org/test#Article
            -> Article

        http://example.org/test/article_46
            -> article_46
    """

    text = str(value)

    if "#" in text:
        return text.rsplit("#", 1)[1]

    return text.rstrip("/").rsplit("/", 1)[-1]


def load_graph(
    ttl_path: str | Path,
) -> Graph:
    """
    Load a local Turtle file into an RDFLib Graph.

    The function fails clearly when:
        - The file does not exist.
        - The path is not a file.
        - The extension is incorrect.
        - The graph is empty.
        - Turtle syntax is invalid.
    """

    path = Path(ttl_path)

    if not path.exists():
        raise FileNotFoundError(
            f"TTL file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"TTL path is not a file: {path}"
        )

    if path.suffix.lower() not in {
        ".ttl",
        ".turtle",
    }:
        raise ValueError(
            "Expected a Turtle file ending with "
            f".ttl or .turtle: {path}"
        )

    graph = Graph()

    graph.parse(
        source=path,
        format="turtle",
    )

    if len(graph) == 0:
        raise ValueError(
            f"The parsed RDF graph is empty: {path}"
        )

    return graph


def _literal_groups(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """
    Group literal values by language.

    Returns:
        Arabic values
        English values
        Other or language-less values
    """

    arabic: set[str] = set()
    english: set[str] = set()
    other: set[str] = set()

    for value in graph.objects(
        subject,
        predicate,
    ):
        if not isinstance(value, Literal):
            continue

        text = str(value).strip()

        if not text:
            continue

        language = (
            value.language or ""
        ).lower()

        if (
            language == "ar"
            or language.startswith("ar-")
        ):
            arabic.add(text)

        elif (
            language == "en"
            or language.startswith("en-")
        ):
            english.add(text)

        else:
            other.add(text)

    return (
        tuple(sorted(arabic)),
        tuple(sorted(english)),
        tuple(sorted(other)),
    )


def _article_number(
    uri: URIRef,
) -> int | None:
    """
    Extract an article number from an article URI.
    """

    match = ARTICLE_URI_PATTERN.search(
        str(uri)
    )

    if match is None:
        return None

    return int(match.group(1))


def _node_kind(
    graph: Graph,
    uri: URIRef,
) -> str:
    """
    Assign one high-level kind to a URI resource.

    Specific legal document types take priority over
    generic OWL types.
    """

    rdf_types = set(
        graph.objects(
            uri,
            RDF.type,
        )
    )

    if LAW.Article in rdf_types:
        return "Article"

    if LAW.Paragraph in rdf_types:
        return "Paragraph"

    if LAW.Definition in rdf_types:
        return "Definition"

    if LAW.Law in rdf_types:
        return "Law"

    if OWL.Class in rdf_types:
        return "OntologyClass"

    if OWL.ObjectProperty in rdf_types:
        return "ObjectProperty"

    if OWL.DatatypeProperty in rdf_types:
        return "DatatypeProperty"

    if OWL.AnnotationProperty in rdf_types:
        return "AnnotationProperty"

    if OWL.Ontology in rdf_types:
        return "Ontology"

    # For legal individuals, retain one direct ontology type.
    domain_types = sorted(
        local_name(item)
        for item in rdf_types
        if (
            isinstance(item, URIRef)
            and str(item).startswith(str(LAW))
        )
    )

    if domain_types:
        return domain_types[0]

    return "Resource"


def _build_searchable_text(
    uri: URIRef,
    node_kind: str,
    rdf_types: Iterable[str],
    labels_ar: Iterable[str],
    labels_en: Iterable[str],
    labels_other: Iterable[str],
    comments_ar: Iterable[str],
    comments_en: Iterable[str],
    comments_other: Iterable[str],
) -> str:
    """
    Build the text that will later be embedded using
    text-embedding-3-small.

    The original full article text remains stored separately.
    This field only combines useful searchable information.
    """

    pieces: list[str] = [
        f"URI: {uri}",
        f"Local name: {local_name(uri)}",
        f"Node kind: {node_kind}",
    ]

    type_names = [
        local_name(value)
        for value in rdf_types
    ]

    if type_names:
        pieces.append(
            "RDF types: "
            + ", ".join(sorted(type_names))
        )

    grouped_values = (
        ("Arabic labels", labels_ar),
        ("English labels", labels_en),
        ("Other labels", labels_other),
        ("Arabic text", comments_ar),
        ("English text", comments_en),
        ("Other text", comments_other),
    )

    for title, values in grouped_values:
        values_list = list(values)

        if values_list:
            pieces.append(
                f"{title}: "
                + "\n".join(values_list)
            )

    return "\n".join(pieces)


def extract_node_records(
    graph: Graph,
) -> list[RdfNodeRecord]:
    """
    Convert URI subjects and URI objects into node records.

    Predicates are also included when they are declared as
    ontology resources in the TTL.
    """

    resources: set[URIRef] = set()

    for subject, _, obj in graph:
        if isinstance(subject, URIRef):
            resources.add(subject)

        if isinstance(obj, URIRef):
            resources.add(obj)

    records: list[RdfNodeRecord] = []

    for uri in sorted(
        resources,
        key=str,
    ):
        rdf_types = tuple(
            sorted(
                str(value)
                for value in graph.objects(
                    uri,
                    RDF.type,
                )
                if isinstance(value, URIRef)
            )
        )

        (
            labels_ar,
            labels_en,
            labels_other,
        ) = _literal_groups(
            graph,
            uri,
            RDFS.label,
        )

        (
            comments_ar,
            comments_en,
            comments_other,
        ) = _literal_groups(
            graph,
            uri,
            RDFS.comment,
        )

        node_kind = _node_kind(
            graph,
            uri,
        )

        article_number = None

        if node_kind == "Article":
            article_number = _article_number(
                uri
            )

        searchable_text = _build_searchable_text(
            uri=uri,
            node_kind=node_kind,
            rdf_types=rdf_types,
            labels_ar=labels_ar,
            labels_en=labels_en,
            labels_other=labels_other,
            comments_ar=comments_ar,
            comments_en=comments_en,
            comments_other=comments_other,
        )

        records.append(
            RdfNodeRecord(
                uri=str(uri),
                local_name=local_name(uri),
                node_kind=node_kind,
                rdf_types=rdf_types,
                labels_ar=labels_ar,
                labels_en=labels_en,
                labels_other=labels_other,
                comments_ar=comments_ar,
                comments_en=comments_en,
                comments_other=comments_other,
                article_number=article_number,
                searchable_text=searchable_text,
            )
        )

    return records


def extract_edge_records(
    graph: Graph,
) -> list[RdfEdgeRecord]:
    """
    Convert every RDF triple into one edge record.

    Current KG objects are either:
        - URI resources
        - Literal values

    The current KG does not contain blank nodes. If one is
    found in a future version, this function fails rather
    than silently dropping information.
    """

    records: list[RdfEdgeRecord] = []

    sorted_triples = sorted(
        graph,
        key=lambda triple: tuple(
            str(item)
            for item in triple
        ),
    )

    for subject, predicate, obj in sorted_triples:
        if not isinstance(subject, URIRef):
            raise ValueError(
                "Blank-node subjects are not supported "
                "by this first ingestion design: "
                f"{subject}"
            )

        if isinstance(obj, URIRef):
            records.append(
                RdfEdgeRecord(
                    source_uri=str(subject),
                    predicate_uri=str(predicate),
                    predicate_local_name=(
                        local_name(predicate)
                    ),
                    object_kind="uri",
                    target_uri=str(obj),
                )
            )

            continue

        if isinstance(obj, Literal):
            records.append(
                RdfEdgeRecord(
                    source_uri=str(subject),
                    predicate_uri=str(predicate),
                    predicate_local_name=(
                        local_name(predicate)
                    ),
                    object_kind="literal",
                    literal_value=str(obj),
                    literal_language=obj.language,
                    literal_datatype=(
                        str(obj.datatype)
                        if obj.datatype
                        else None
                    ),
                )
            )

            continue

        raise ValueError(
            "Unsupported RDF object type: "
            f"{type(obj).__name__}: {obj}"
        )

    return records


def build_inspection_report(
    graph: Graph,
    ttl_path: str | Path,
    node_records: list[RdfNodeRecord],
    edge_records: list[RdfEdgeRecord],
) -> KgInspectionReport:
    """
    Validate the graph before Weaviate ingestion.
    """

    articles = [
        record
        for record in node_records
        if record.node_kind == "Article"
    ]

    article_number_to_uris: dict[
        int,
        list[str],
    ] = defaultdict(list)

    for article in articles:
        if article.article_number is not None:
            article_number_to_uris[
                article.article_number
            ].append(article.uri)

    article_numbers = tuple(
        sorted(article_number_to_uris)
    )

    expected_numbers = set(
        range(1, 143)
    )
    actual_numbers = set(
        article_numbers
    )

    missing_article_numbers = tuple(
        sorted(
            expected_numbers
            - actual_numbers
        )
    )

    duplicate_article_numbers = tuple(
        sorted(
            number
            for number, uris
            in article_number_to_uris.items()
            if len(uris) > 1
        )
    )

    articles_without_arabic_text = tuple(
        sorted(
            article.uri
            for article in articles
            if not article.comments_ar
        )
    )

    articles_without_labels = tuple(
        sorted(
            article.uri
            for article in articles
            if not (
                article.labels_ar
                or article.labels_en
                or article.labels_other
            )
        )
    )

    node_kind_counts = Counter(
        record.node_kind
        for record in node_records
    )

    is_valid = all(
        (
            len(graph) > 0,
            len(edge_records) == len(graph),
            len(articles) == 142,
            not missing_article_numbers,
            not duplicate_article_numbers,
            not articles_without_arabic_text,
            not articles_without_labels,
        )
    )

    path = Path(ttl_path)

    return KgInspectionReport(
        ttl_path=str(path),
        file_size_bytes=path.stat().st_size,
        triple_count=len(graph),
        node_count=len(node_records),
        edge_count=len(edge_records),
        article_count=len(articles),
        paragraph_count=node_kind_counts[
            "Paragraph"
        ],
        definition_count=node_kind_counts[
            "Definition"
        ],
        ontology_class_count=node_kind_counts[
            "OntologyClass"
        ],
        object_property_count=node_kind_counts[
            "ObjectProperty"
        ],
        datatype_property_count=node_kind_counts[
            "DatatypeProperty"
        ],
        article_numbers=article_numbers,
        missing_article_numbers=(
            missing_article_numbers
        ),
        duplicate_article_numbers=(
            duplicate_article_numbers
        ),
        articles_without_arabic_text=(
            articles_without_arabic_text
        ),
        articles_without_labels=(
            articles_without_labels
        ),
        is_valid=is_valid,
    )


def load_and_inspect(
    ttl_path: str | Path,
) -> tuple[
    Graph,
    list[RdfNodeRecord],
    list[RdfEdgeRecord],
    KgInspectionReport,
]:
    """
    Load the TTL and return all extracted structures.
    """

    graph = load_graph(ttl_path)

    nodes = extract_node_records(
        graph
    )

    edges = extract_edge_records(
        graph
    )

    report = build_inspection_report(
        graph=graph,
        ttl_path=ttl_path,
        node_records=nodes,
        edge_records=edges,
    )

    return (
        graph,
        nodes,
        edges,
        report,
    )