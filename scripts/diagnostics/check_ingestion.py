from __future__ import annotations

import json
import sys
from typing import Any

from app.config import get_settings
from app.rdf_loader import load_and_inspect
from app.weaviate_db import connect_to_weaviate_with_retry
from app.weaviate_ingestion import collection_count


def main() -> int:
    settings = get_settings()
    client: Any | None = None
    try:
        _, _, _, report = load_and_inspect(settings.kg_ttl_path)
        if not report.is_valid:
            raise RuntimeError(
                "Configured TTL is not valid for graph-only retrieval; "
                f"unreachable_articles={list(report.unreachable_article_numbers)}"
            )

        client = connect_to_weaviate_with_retry(settings)
        actual_nodes = collection_count(client, settings.weaviate_node_collection)
        actual_edges = collection_count(client, settings.weaviate_edge_collection)

        output = {
            "status": "success" if (
                actual_nodes == report.node_count and actual_edges == report.edge_count
            ) else "mismatch",
            "expected": {
                "nodes": report.node_count,
                "edges": report.edge_count,
                "reachable_articles": report.semantically_reachable_article_count,
            },
            "actual": {
                "nodes": actual_nodes,
                "edges": actual_edges,
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if output["status"] == "success" else 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
