from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.config import get_settings
from app.openai_service import OpenAIService
from app.retrieval_service import RetrievalService
from app.weaviate_db import (
    connect_to_weaviate_with_retry,
)
from app.weaviate_ingestion import (
    collection_count,
)


DEFAULT_QUESTION = (
    "صاحب العمل لم يدفع راتبي منذ عشرة أيام، "
    "ماذا أستطيع أن أفعل؟"
)


def parse_arguments() -> argparse.Namespace:
    """
    Read an optional legal question from the command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test graph-aware retrieval against the "
            "Jordanian Labour Law KG."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Arabic or English legal question.",
    )

    return parser.parse_args()


def main() -> int:
    """
    Embed the question and run graph-aware retrieval.
    """

    settings = get_settings()
    arguments = parse_arguments()

    question = arguments.question.strip()

    if not question:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": "Question cannot be empty.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        return 2

    client: Any | None = None
    openai_service: OpenAIService | None = None

    try:
        client = connect_to_weaviate_with_retry(
            settings
        )

        node_count = collection_count(
            client,
            settings.weaviate_node_collection,
        )

        edge_count = collection_count(
            client,
            settings.weaviate_edge_collection,
        )

        if node_count == 0:
            raise RuntimeError(
                "The node collection is empty. "
                "Run scripts.ingest_kg first."
            )

        openai_service = OpenAIService(
            settings
        )

        embedding = openai_service.embed_texts(
            [question]
        )

        query_vector = embedding.vectors[0]

        retrieval_service = RetrievalService(
            client=client,
            settings=settings,
        )

        preview = retrieval_service.preview(
            question=question,
            query_vector=query_vector,
            embedding_model=(
                settings.openai_embedding_model
            ),
            embedding_dimensions=(
                embedding.dimensions
            ),
            embedding_input_tokens=(
                embedding.input_tokens
            ),
        )

        output = {
            "status": "success",
            "collection_counts": {
                "nodes": node_count,
                "edges": edge_count,
            },
            "diagnostics": {
                "direct_concept_count": len(
                    preview.concept_hits
                ),
                "expanded_concept_count": len(
                    preview.expanded_concept_hits
                ),
                "top_article_numbers": [
                    hit.article_number
                    for hit in preview.article_hits
                ],
                "graph_supported_articles": [
                    hit.article_number
                    for hit in preview.article_hits
                    if hit.graph_supported
                ],
            },
            "retrieval": preview.to_dict(),
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        return 1

    finally:
        if openai_service is not None:
            openai_service.close()

        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())