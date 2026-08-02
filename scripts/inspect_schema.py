from __future__ import annotations

import json
import sys
from typing import Any

from app.config import get_settings
from app.weaviate_db import (
    connect_to_weaviate_with_retry,
)
from app.weaviate_schema import (
    inspect_collection,
)


def main() -> int:
    settings = get_settings()

    client: Any | None = None

    try:
        client = connect_to_weaviate_with_retry(
            settings
        )

        collection_names = [
            settings.weaviate_node_collection,
            settings.weaviate_edge_collection,
        ]

        missing = [
            name
            for name in collection_names
            if not client.collections.exists(name)
        ]

        if missing:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "message": (
                            "One or more collections "
                            "do not exist."
                        ),
                        "missing": missing,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            return 2

        output = {
            "status": "success",
            "collections": [
                inspect_collection(
                    client=client,
                    collection_name=name,
                )
                for name in collection_names
            ],
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