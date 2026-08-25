from __future__ import annotations

import json

from app.config import get_settings
from app.rdf_loader import load_and_inspect


def main() -> int:
    settings = get_settings()
    _, nodes, _, report = load_and_inspect(settings.kg_ttl_path)
    eligible = [node for node in nodes if node.retrieval_eligible]
    print(json.dumps({
        "status": "pass" if not report.unreachable_article_numbers else "fail",
        "retrieval_eligible_concepts": len(eligible),
        "reachable_articles": report.semantically_reachable_article_count,
        "unreachable_articles": list(report.unreachable_article_numbers),
        "article_count": report.article_count,
    }, ensure_ascii=False, indent=2))
    return 0 if not report.unreachable_article_numbers else 2


if __name__ == "__main__":
    raise SystemExit(main())
