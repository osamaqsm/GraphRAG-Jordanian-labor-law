from __future__ import annotations

import json
import sys
from typing import Any

from weaviate.classes.query import Filter

from app.config import get_settings
from app.weaviate_db import (
    connect_to_weaviate_with_retry,
)
from app.weaviate_ingestion import (
    collection_count,
)


ARTICLE_46_URI = (
    "http://example.org/"
    "jordan-labor-law#article_46"
)


def vector_dimensions(
    vector: Any,
) -> Any:
    """
    Handle either an unnamed vector list or a named-vector
    dictionary returned by Weaviate.
    """

    if isinstance(vector, dict):
        return {
            name: len(values)
            for name, values in vector.items()
        }

    if isinstance(vector, list):
        return len(vector)

    return None


def main() -> int:
    settings = get_settings()

    client: Any | None = None

    try:
        client = connect_to_weaviate_with_retry(
            settings
        )

        nodes = client.collections.use(
            settings.weaviate_node_collection
        )

        edges = client.collections.use(
            settings.weaviate_edge_collection
        )

        article_response = (
            nodes.query.fetch_objects(
                filters=(
                    Filter
                    .by_property("articleNumber")
                    .equal(46)
                ),
                limit=1,
                include_vector=True,
            )
        )

        if not article_response.objects:
            raise RuntimeError(
                "Article 46 was not found."
            )

        article_object = (
            article_response.objects[0]
        )

        article_properties = (
            article_object.properties
        )

        comments = article_properties.get(
            "commentsAr",
            [],
        )

        article_text = (
            comments[0]
            if comments
            else ""
        )

        edge_response = (
            edges.query.fetch_objects(
                filters=(
                    Filter
                    .by_property("sourceUri")
                    .equal(ARTICLE_46_URI)
                ),
                limit=20,
            )
        )

        output = {
            "status": "success",
            "counts": {
                "nodes": collection_count(
                    client,
                    settings.weaviate_node_collection,
                ),
                "edges": collection_count(
                    client,
                    settings.weaviate_edge_collection,
                ),
            },
            "article_46": {
                "uuid": str(
                    article_object.uuid
                ),
                "uri": article_properties.get(
                    "uri"
                ),
                "labels_ar": (
                    article_properties.get(
                        "labelsAr",
                        [],
                    )
                ),
                "article_number": (
                    article_properties.get(
                        "articleNumber"
                    )
                ),
                "text_length": len(
                    article_text
                ),
                "text_preview": (
                    article_text[:350]
                ),
                "vector_dimensions": (
                    vector_dimensions(
                        article_object.vector
                    )
                ),
            },
            "article_46_outgoing_edges": [
                {
                    "predicate": (
                        item.properties.get(
                            "predicateLocalName"
                        )
                    ),
                    "object_kind": (
                        item.properties.get(
                            "objectKind"
                        )
                    ),
                    "target_uri": (
                        item.properties.get(
                            "targetUri"
                        )
                    ),
                    "literal_language": (
                        item.properties.get(
                            "literalLanguage"
                        )
                    ),
                }
                for item in edge_response.objects
            ],
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

        expected_nodes = 671
        expected_edges = 3694

        if (
            output["counts"]["nodes"]
            != expected_nodes
        ):
            return 2

        if (
            output["counts"]["edges"]
            != expected_edges
        ):
            return 2

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
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())