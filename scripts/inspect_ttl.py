from __future__ import annotations

import json
import sys

from app.config import get_settings
from app.rdf_loader import load_and_inspect


def main() -> int:
    """
    Parse the configured KG and print a validation report.

    Exit codes:
        0 = valid KG
        1 = parsing/runtime error
        2 = KG parsed but failed validation
    """

    settings = get_settings()

    try:
        (
            _,
            nodes,
            edges,
            report,
        ) = load_and_inspect(
            settings.kg_ttl_path
        )

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

    report_data = report.to_dict()

    # Avoid printing a list containing all 142 numbers.
    article_numbers = report_data.pop(
        "article_numbers"
    )

    report_data["article_number_count"] = len(
        article_numbers
    )

    report_data["article_number_range"] = (
        [
            article_numbers[0],
            article_numbers[-1],
        ]
        if article_numbers
        else []
    )

    # Show Article 46 because it is a useful test case
    # for wage-delay questions.
    article_46 = next(
        (
            node.to_dict()
            for node in nodes
            if node.uri.endswith(
                "#article_46"
            )
        ),
        None,
    )

    semantic_edges = [
        edge.to_dict()
        for edge in edges
        if edge.predicate_local_name
        in {
            "hasCondition",
            "violatesRight",
            "breachesObligation",
            "resultsIn",
            "supportedByArticle",
        }
    ][:5]

    output = {
        "status": (
            "valid"
            if report.is_valid
            else "invalid"
        ),
        "report": report_data,
        "article_46_sample": article_46,
        "semantic_edge_samples": (
            semantic_edges
        ),
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if report.is_valid:
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())