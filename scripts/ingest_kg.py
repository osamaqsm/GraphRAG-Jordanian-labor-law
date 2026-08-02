from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.config import get_settings
from app.openai_service import OpenAIService
from app.rdf_loader import load_and_inspect
from app.weaviate_db import (
    connect_to_weaviate_with_retry,
)
from app.weaviate_ingestion import (
    collection_count,
    ingest_kg_records,
)
from app.weaviate_schema import (
    ensure_collections,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed and ingest the Jordan Labour Law KG "
            "into Weaviate."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate the two project "
            "collections before ingestion."
        ),
    )

    return parser.parse_args()


def progress(message: str) -> None:
    """
    Print progress to stderr so final JSON stays clean.
    """

    print(
        message,
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    settings = get_settings()
    arguments = parse_arguments()

    client: Any | None = None
    openai_service: OpenAIService | None = None

    try:
        progress("Reading and validating the TTL file.")

        (
            _,
            nodes,
            edges,
            report,
        ) = load_and_inspect(
            settings.kg_ttl_path
        )

        if not report.is_valid:
            raise RuntimeError(
                "The KG failed offline validation. "
                "Ingestion was cancelled."
            )

        progress(
            f"TTL is valid: {len(nodes)} nodes and "
            f"{len(edges)} edges."
        )

        client = connect_to_weaviate_with_retry(
            settings
        )

        openai_service = OpenAIService(
            settings
        )

        progress(
            "Testing the OpenAI embedding connection."
        )

        probe = openai_service.embed_texts(
            ["اختبار اتصال لنظام قانون العمل الأردني"]
        )

        progress(
            "OpenAI connection succeeded. "
            f"Embedding dimension: {probe.dimensions}."
        )

        changes = ensure_collections(
            client=client,
            settings=settings,
            reset=arguments.reset,
        )

        node_count_before = collection_count(
            client,
            settings.weaviate_node_collection,
        )

        edge_count_before = collection_count(
            client,
            settings.weaviate_edge_collection,
        )

        if (
            not arguments.reset
            and (
                node_count_before > 0
                or edge_count_before > 0
            )
        ):
            raise RuntimeError(
                "The collections are not empty. "
                "Run this command with --reset to "
                "perform a clean deterministic import."
            )

        progress(
            "Starting node embedding and ingestion."
        )

        summary = ingest_kg_records(
            client=client,
            settings=settings,
            service=openai_service,
            node_records=nodes,
            edge_records=edges,
            progress=progress,
        )

        output = {
            "status": "success",
            "schema_changes": changes,
            "ttl_validation": {
                "triples": report.triple_count,
                "nodes": report.node_count,
                "articles": report.article_count,
                "paragraphs": (
                    report.paragraph_count
                ),
                "definitions": (
                    report.definition_count
                ),
            },
            "openai": {
                "embedding_model": (
                    settings.openai_embedding_model
                ),
                "probe_dimensions": (
                    probe.dimensions
                ),
            },
            "ingestion": summary.to_dict(),
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