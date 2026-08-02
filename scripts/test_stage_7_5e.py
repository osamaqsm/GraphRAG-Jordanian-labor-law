from __future__ import annotations

import os
from pathlib import Path

from rdflib import Graph, Namespace, RDFS

from app.legal_question_analysis import analyze_legal_question
from app.retrieval_models import RetrievalHit
from app.retrieval_service import RetrievalService


KG = Namespace("http://example.org/jordan-labor-law#")

CASES = (
    (
        "Q01",
        "صاحب العمل لم يدفع راتبي منذ عشرة أيام، ماذا أستطيع أن أفعل؟",
        ((46, 0.90), (50, 0.82), (54, 0.60)),
        [46, 54],
    ),
    (
        "Q08",
        "كلفني صاحب العمل بعمل مختلف اختلافا واضحا عن العمل المتفق عليه، هل يلزمني تنفيذه وهل أستطيع ترك العمل؟",
        ((17, 0.90), (18, 0.82), (29, 0.60)),
        [17, 29],
    ),
    (
        "Q09",
        "هل يلتزم العامل بالمحافظة على أسرار صاحب العمل، وما نتيجة إفشائها؟",
        ((19, 0.90), (30, 0.82), (28, 0.60)),
        [19, 28],
    ),
)


def _ttl_path() -> Path:
    return Path(
        os.getenv(
            "KG_TTL_PATH",
            "/app/data/jordan_labor_law_full_knowledge_graph.ttl",
        )
    )


def _article_hit(
    graph: Graph,
    article_number: int,
    final_score: float,
) -> RetrievalHit:
    uri = KG[f"article_{article_number}"]
    text = next(
        (str(value) for value in graph.objects(uri, RDFS.comment)),
        "",
    )

    hit = RetrievalHit(
        uri=str(uri),
        local_name=f"article_{article_number}",
        node_kind="Article",
        labels_ar=[f"المادة {article_number}"],
        labels_en=[],
        article_number=article_number,
        text_preview=text,
    )
    hit.final_score = final_score
    return hit


def main() -> int:
    graph = Graph()
    graph.parse(_ttl_path(), format="turtle")

    service = RetrievalService.__new__(RetrievalService)

    for question_id, question, candidates, expected in CASES:
        analysis = analyze_legal_question(question)
        ranked = [
            _article_hit(graph, article_number, score)
            for article_number, score in candidates
        ]
        selected = service._select_complementary_articles(
            ranked=ranked,
            analysis=analysis,
            final_limit=analysis.max_final_articles,
        )
        actual = [hit.article_number for hit in selected]

        print(
            f"{question_id} | expected={expected} | actual={actual}"
        )

        if actual != expected:
            return 1

    print("Stage 7.5-E complementary-selection checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())