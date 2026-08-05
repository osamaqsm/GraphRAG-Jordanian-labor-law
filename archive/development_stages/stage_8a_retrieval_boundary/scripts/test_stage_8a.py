from __future__ import annotations

import inspect
import json

from app.retrieval_contract import (
    RETRIEVAL_CONTRACT_VERSION,
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalResultV1,
)
from app.retrieval_service import RetrievalService


def main() -> None:
    signature = inspect.signature(RetrievalService.preview)
    assert "analysis" in signature.parameters
    assert signature.parameters["analysis"].default is None

    result = RetrievalResultV1(
        question="اختبار الاسترجاع فقط",
        decision=RetrievalDecisionV1(
            behavior="retrieve",
            reason="test",
        ),
        embedding=RetrievalEmbeddingV1(
            model="text-embedding-3-small",
            dimensions=1536,
            input_tokens=5,
        ),
        diagnostics=RetrievalDiagnosticsV1(),
        elapsed_ms=1,
    )
    payload = result.model_dump(mode="json")
    assert payload["schema_version"] == RETRIEVAL_CONTRACT_VERSION
    assert "answer" not in payload
    assert "generation" not in payload
    roundtrip = RetrievalResultV1.model_validate_json(
        json.dumps(payload, ensure_ascii=False)
    )
    assert roundtrip.question == result.question

    print("Stage 8-A offline checks passed.")
    print("Retrieval contract: retrieval.v1")
    print("RetrievalService accepts one precomputed analysis")
    print("No answer or generation field exists in the contract")


if __name__ == "__main__":
    main()
