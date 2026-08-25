from pathlib import Path

from app.rdf_loader import load_and_inspect


TTL = Path(__file__).resolve().parents[1] / "data" / "jordan_labor_law_full_knowledge_graph.ttl"


def test_all_142_articles_are_semantically_reachable():
    _, nodes, _, report = load_and_inspect(TTL)
    assert report.is_valid
    assert report.article_count == 142
    assert report.semantically_reachable_article_count == 142
    assert report.unreachable_article_numbers == ()
    assert sum(1 for node in nodes if node.retrieval_eligible) >= 100


def test_document_nodes_are_not_retrieval_eligible():
    _, nodes, _, _ = load_and_inspect(TTL)
    forbidden = {"Article", "Paragraph", "Definition", "Law"}
    assert not [
        node.uri
        for node in nodes
        if node.retrieval_eligible and node.node_kind in forbidden
    ]


def test_article_1_has_real_semantic_concepts():
    _, nodes, edges, _ = load_and_inspect(TTL)
    eligible = {node.uri for node in nodes if node.retrieval_eligible}
    article_1 = "http://example.org/jordan-labor-law#article_1"
    bridge_sources = {
        edge.source_uri
        for edge in edges
        if edge.target_uri == article_1
        and edge.predicate_local_name in {"supportedByArticle", "regulatedBy"}
    }
    assert bridge_sources & eligible
    assert any(uri.endswith("#law_commencement_concept") for uri in bridge_sources)
