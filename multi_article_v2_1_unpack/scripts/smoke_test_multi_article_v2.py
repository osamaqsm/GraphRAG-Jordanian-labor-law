from __future__ import annotations

import argparse
import json

from app.retrieval_pipeline import RetrievalOnlyPipeline


DEFAULT_QUESTION = (
    "قارن بين إجازة تربية الأطفال، وإجازة مرافقة الزوج، وإجازة الأمومة، "
    "وساعة الرضاعة، والتزام توفير مكان لرعاية أطفال العاملات."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test V2 issue-wise multi-article retrieval."
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Arabic legal question to retrieve.",
    )
    args = parser.parse_args()

    with RetrievalOnlyPipeline() as pipeline:
        result = pipeline.retrieve(
            args.question,
            include_debug=True,
        )

    payload = result.model_dump(mode="json")
    debug = payload.get("debug") or {}
    issue_debug = debug.get("issue_wise_retrieval") or {}

    print("QUESTION:")
    print(args.question)
    print("\nRETRIEVED ARTICLES:")
    print(payload["diagnostics"]["article_numbers"])
    print("\nISSUE-WISE DEBUG:")
    print(
        json.dumps(
            issue_debug,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
