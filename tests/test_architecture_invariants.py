from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_structural_catalogue_edges_are_not_traversable():
    graph_source = (ROOT / "app" / "graph_retrieval.py").read_text(encoding="utf-8")
    relation_block = graph_source.split("EVIDENCE_RELATION_WEIGHTS", 1)[1].split(
        "EDGE_RETURN_PROPERTIES", 1
    )[0]
    assert '"hasArticle"' not in relation_block
    assert '"hasParagraph"' not in relation_block
    assert '"type"' not in relation_block


def test_no_direct_or_paragraph_retrieval_implementation_remains():
    service = (ROOT / "app" / "retrieval_service.py").read_text(encoding="utf-8")
    config = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    forbidden = [
        "search_articles_direct",
        "search_paragraphs",
        "search_mixed",
        "retrieval_direct_article_enabled",
        "retrieval_paragraph_enabled",
        "raw_mixed_hits",
    ]
    for value in forbidden:
        assert value not in service
        assert value not in config


def test_reranker_explicitly_allows_empty_graph_evidence_selection():
    source = (ROOT / "app" / "legal_article_reranker.py").read_text(encoding="utf-8")
    assert "An empty selection is a valid outcome" in source
    assert "selected_article_numbers=[]" in source
    assert "Do not force" in source
