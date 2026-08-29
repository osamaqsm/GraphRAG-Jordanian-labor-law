from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from app.rdf_models import KgInspectionReport, RdfEdgeRecord, RdfNodeRecord


LAW = Namespace("http://example.org/jordan-labor-law#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
ARTICLE_URI_PATTERN = re.compile(r"(?:^|[#/])article_(\d+)$")

# These bridge relations are the only direct concept->Article evidence routes.
SEMANTIC_ARTICLE_BRIDGES = {
    "supportedByArticle",
    "regulatedBy",
    "regulates",  # inverse direction Article -> concept
}


def local_name(value: URIRef | str) -> str:
    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def load_graph(ttl_path: str | Path) -> Graph:
    path = Path(ttl_path)
    if not path.exists():
        raise FileNotFoundError(f"TTL file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"TTL path is not a file: {path}")
    if path.suffix.lower() not in {".ttl", ".turtle"}:
        raise ValueError(f"Expected a Turtle file: {path}")

    graph = Graph()
    graph.parse(source=path, format="turtle")
    if len(graph) == 0:
        raise ValueError(f"The parsed RDF graph is empty: {path}")
    return graph


def _literal_groups(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    arabic: set[str] = set()
    english: set[str] = set()
    other: set[str] = set()
    for value in graph.objects(subject, predicate):
        if not isinstance(value, Literal):
            continue
        text = str(value).strip()
        if not text:
            continue
        language = (value.language or "").lower()
        if language == "ar" or language.startswith("ar-"):
            arabic.add(text)
        elif language == "en" or language.startswith("en-"):
            english.add(text)
        else:
            other.add(text)
    return tuple(sorted(arabic)), tuple(sorted(english)), tuple(sorted(other))


def _article_number(uri: URIRef | str) -> int | None:
    match = ARTICLE_URI_PATTERN.search(str(uri))
    return int(match.group(1)) if match else None


def _node_kind(graph: Graph, uri: URIRef) -> str:
    rdf_types = set(graph.objects(uri, RDF.type))

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

    domain_types = sorted(
        local_name(item)
        for item in rdf_types
        if isinstance(item, URIRef) and str(item).startswith(str(LAW))
    )
    return domain_types[0] if domain_types else "Resource"


def _is_subclass_of(graph: Graph, child: URIRef, ancestor: URIRef) -> bool:
    if child == ancestor:
        return True
    frontier = [child]
    visited: set[URIRef] = set()
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        for parent in graph.objects(current, RDFS.subClassOf):
            if not isinstance(parent, URIRef):
                continue
            if parent == ancestor:
                return True
            frontier.append(parent)
    return False


def _retrieval_eligible(graph: Graph, uri: URIRef, node_kind: str) -> bool:
    """Positive eligibility: only semantic concept individuals are searchable."""
    if node_kind in {
        "Article",
        "Paragraph",
        "Definition",
        "Law",
        "Ontology",
        "OntologyClass",
        "ObjectProperty",
        "DatatypeProperty",
        "AnnotationProperty",
        "Resource",
    }:
        return False

    ancestors = (LAW.LegalConcept, LAW.LaborDomainConcept, LAW.StatutoryConcept)
    for rdf_type in graph.objects(uri, RDF.type):
        if not isinstance(rdf_type, URIRef):
            continue
        if any(_is_subclass_of(graph, rdf_type, ancestor) for ancestor in ancestors):
            return True
    return False


def _build_searchable_text(
    *,
    uri: URIRef,
    node_kind: str,
    rdf_types: Iterable[str],
    labels_ar: Iterable[str],
    labels_en: Iterable[str],
    labels_other: Iterable[str],
    aliases_ar: Iterable[str],
    aliases_en: Iterable[str],
    aliases_other: Iterable[str],
    comments_ar: Iterable[str],
    comments_en: Iterable[str],
    comments_other: Iterable[str],
) -> str:
    pieces: list[str] = [
        f"Local name: {local_name(uri)}",
        f"Node kind: {node_kind}",
    ]

    type_names = [local_name(value) for value in rdf_types]
    if type_names:
        pieces.append("RDF types: " + ", ".join(sorted(type_names)))

    grouped_values = (
        ("Arabic labels", labels_ar),
        ("Arabic aliases", aliases_ar),
        ("English labels", labels_en),
        ("English aliases", aliases_en),
        ("Other labels", labels_other),
        ("Other aliases", aliases_other),
        ("Arabic text", comments_ar),
        ("English text", comments_en),
        ("Other text", comments_other),
    )
    for title, values in grouped_values:
        values_list = list(values)
        if values_list:
            pieces.append(f"{title}: " + "\n".join(values_list))

    return "\n".join(pieces)


def extract_node_records(graph: Graph) -> list[RdfNodeRecord]:
    resources: set[URIRef] = set()
    for subject, _, obj in graph:
        if isinstance(subject, URIRef):
            resources.add(subject)
        if isinstance(obj, URIRef):
            resources.add(obj)

    records: list[RdfNodeRecord] = []
    for uri in sorted(resources, key=str):
        rdf_types = tuple(
            sorted(
                str(value)
                for value in graph.objects(uri, RDF.type)
                if isinstance(value, URIRef)
            )
        )
        labels_ar, labels_en, labels_other = _literal_groups(graph, uri, RDFS.label)
        aliases_ar, aliases_en, aliases_other = _literal_groups(graph, uri, SKOS.altLabel)
        comments_ar, comments_en, comments_other = _literal_groups(graph, uri, RDFS.comment)

        node_kind = _node_kind(graph, uri)
        article_number = _article_number(uri) if node_kind == "Article" else None
        retrieval_eligible = _retrieval_eligible(graph, uri, node_kind)
        searchable_text = _build_searchable_text(
            uri=uri,
            node_kind=node_kind,
            rdf_types=rdf_types,
            labels_ar=labels_ar,
            labels_en=labels_en,
            labels_other=labels_other,
            aliases_ar=aliases_ar,
            aliases_en=aliases_en,
            aliases_other=aliases_other,
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
                aliases_ar=aliases_ar,
                aliases_en=aliases_en,
                aliases_other=aliases_other,
                comments_ar=comments_ar,
                comments_en=comments_en,
                comments_other=comments_other,
                article_number=article_number,
                retrieval_eligible=retrieval_eligible,
                searchable_text=searchable_text,
            )
        )
    return records


def extract_edge_records(graph: Graph) -> list[RdfEdgeRecord]:
    records: list[RdfEdgeRecord] = []
    for subject, predicate, obj in sorted(
        graph,
        key=lambda triple: tuple(str(item) for item in triple),
    ):
        if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
            raise ValueError("Blank-node RDF subjects/predicates are not supported.")

        if isinstance(obj, URIRef):
            records.append(
                RdfEdgeRecord(
                    source_uri=str(subject),
                    predicate_uri=str(predicate),
                    predicate_local_name=local_name(predicate),
                    object_kind="uri",
                    target_uri=str(obj),
                )
            )
        elif isinstance(obj, Literal):
            records.append(
                RdfEdgeRecord(
                    source_uri=str(subject),
                    predicate_uri=str(predicate),
                    predicate_local_name=local_name(predicate),
                    object_kind="literal",
                    literal_value=str(obj),
                    literal_language=obj.language,
                    literal_datatype=str(obj.datatype) if obj.datatype else None,
                )
            )
        else:
            raise ValueError(f"Unsupported RDF object type: {type(obj).__name__}")
    return records


def _semantic_article_reachability(graph: Graph, eligible_uris: set[str]) -> set[int]:
    reachable: set[int] = set()
    for subject, predicate, obj in graph:
        relation = local_name(predicate)
        if relation not in SEMANTIC_ARTICLE_BRIDGES:
            continue

        subject_uri = str(subject) if isinstance(subject, URIRef) else ""
        object_uri = str(obj) if isinstance(obj, URIRef) else ""

        if relation in {"supportedByArticle", "regulatedBy"}:
            if subject_uri not in eligible_uris:
                continue
            number = _article_number(object_uri)
        else:  # regulates: Article -> semantic concept
            if object_uri not in eligible_uris:
                continue
            number = _article_number(subject_uri)

        if number is not None:
            reachable.add(number)
    return reachable


def build_inspection_report(
    *,
    graph: Graph,
    ttl_path: str | Path,
    node_records: list[RdfNodeRecord],
    edge_records: list[RdfEdgeRecord],
) -> KgInspectionReport:
    articles = [record for record in node_records if record.node_kind == "Article"]
    article_number_to_uris: dict[int, list[str]] = defaultdict(list)
    for article in articles:
        if article.article_number is not None:
            article_number_to_uris[article.article_number].append(article.uri)

    article_numbers = tuple(sorted(article_number_to_uris))
    expected_numbers = set(range(1, 143))
    actual_numbers = set(article_numbers)
    missing_article_numbers = tuple(sorted(expected_numbers - actual_numbers))
    duplicate_article_numbers = tuple(
        sorted(number for number, uris in article_number_to_uris.items() if len(uris) > 1)
    )
    articles_without_arabic_text = tuple(
        sorted(article.uri for article in articles if not article.comments_ar)
    )
    articles_without_labels = tuple(
        sorted(
            article.uri
            for article in articles
            if not (article.labels_ar or article.labels_en or article.labels_other)
        )
    )

    node_kind_counts = Counter(record.node_kind for record in node_records)
    eligible_uris = {record.uri for record in node_records if record.retrieval_eligible}
    reachable = _semantic_article_reachability(graph, eligible_uris)
    unreachable_article_numbers = tuple(sorted(expected_numbers - reachable))

    is_valid = all(
        (
            len(graph) > 0,
            len(edge_records) == len(graph),
            len(articles) == 142,
            not missing_article_numbers,
            not duplicate_article_numbers,
            not articles_without_arabic_text,
            not articles_without_labels,
            not unreachable_article_numbers,
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
        paragraph_count=node_kind_counts["Paragraph"],
        definition_count=node_kind_counts["Definition"],
        ontology_class_count=node_kind_counts["OntologyClass"],
        object_property_count=node_kind_counts["ObjectProperty"],
        datatype_property_count=node_kind_counts["DatatypeProperty"],
        retrieval_eligible_concept_count=len(eligible_uris),
        semantically_reachable_article_count=len(reachable),
        article_numbers=article_numbers,
        missing_article_numbers=missing_article_numbers,
        duplicate_article_numbers=duplicate_article_numbers,
        unreachable_article_numbers=unreachable_article_numbers,
        articles_without_arabic_text=articles_without_arabic_text,
        articles_without_labels=articles_without_labels,
        is_valid=is_valid,
    )


def load_and_inspect(
    ttl_path: str | Path,
) -> tuple[Graph, list[RdfNodeRecord], list[RdfEdgeRecord], KgInspectionReport]:
    graph = load_graph(ttl_path)
    nodes = extract_node_records(graph)
    edges = extract_edge_records(graph)
    report = build_inspection_report(
        graph=graph,
        ttl_path=ttl_path,
        node_records=nodes,
        edge_records=edges,
    )
    return graph, nodes, edges, report
