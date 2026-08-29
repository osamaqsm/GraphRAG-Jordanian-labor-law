from __future__ import annotations

import json
import sys

from app.config import get_settings
from app.rdf_loader import load_and_inspect


def main() -> int:
    settings = get_settings()
    try:
        _, nodes, edges, report = load_and_inspect(settings.kg_ttl_path)
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1

    report_data = report.to_dict()
    article_numbers = report_data.pop("article_numbers")
    report_data["article_number_count"] = len(article_numbers)
    report_data["article_number_range"] = (
        [article_numbers[0], article_numbers[-1]] if article_numbers else []
    )

    eligible_samples = [
        node.to_dict() for node in nodes if node.retrieval_eligible
    ][:8]
    bridge_samples = [
        edge.to_dict()
        for edge in edges
        if edge.predicate_local_name in {"supportedByArticle", "regulatedBy", "regulates"}
    ][:8]

    output = {
        "status": "valid" if report.is_valid else "invalid",
        "report": report_data,
        "retrieval_eligible_concept_samples": eligible_samples,
        "semantic_article_bridge_samples": bridge_samples,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    sys.exit(main())
