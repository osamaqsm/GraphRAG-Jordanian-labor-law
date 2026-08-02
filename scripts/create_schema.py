from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.config import get_settings
from app.weaviate_db import (
    connect_to_weaviate_with_retry,
)
from app.weaviate_schema import (
    ensure_collections,
    inspect_collection,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create the Weaviate collections used by "
            "the Jordanian Labour Law KG."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate the node and edge "
            "collections. Existing objects in these "
            "collections will be lost."
        ),
    )

    return parser.parse_args()


def main() -> int:
    settings = get_settings()
    arguments = parse_arguments()

    client: Any | None = None

    try:
        client = connect_to_weaviate_with_retry(
            settings
        )

        changes = ensure_collections(
            client=client,
            settings=settings,
            reset=arguments.reset,
        )

        node_schema = inspect_collection(
            client=client,
            collection_name=(
                settings.weaviate_node_collection
            ),
        )

        edge_schema = inspect_collection(
            client=client,
            collection_name=(
                settings.weaviate_edge_collection
            ),
        )

        output = {
            "status": "success",
            "changes": changes,
            "collections": {
                "node": node_schema,
                "edge": edge_schema,
            },
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
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())