from __future__ import annotations

import json
from types import SimpleNamespace

from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import (
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalEvidenceV1,
    RetrievalResultV1,
)


class FakeUsage:
    input_tokens = 10
    output_tokens = 5
    total_tokens = 15


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(
            payload,
            ensure_ascii=False,
        )
        self.usage = FakeUsage()


class FakeResponses:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def create(self, **kwargs):
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.responses = FakeResponses(payload)


def retrieval(numbers: list[int]) -> RetrievalResultV1:
    return RetrievalResultV1(
        question="سؤال تجريبي",
        decision=RetrievalDecisionV1(
            behavior="retrieve",
            reason="test",
        ),
        embedding=RetrievalEmbeddingV1(
            model="test",
            dimensions=0,
            input_tokens=0,
        ),
        articles=[
            RetrievalEvidenceV1(
                uri=f"http://example.org/article_{number}",
                local_name=f"article_{number}",
                node_kind="Article",
                labels_ar=[f"المادة {number}"],
                labels_en=[],
                article_number=number,
                text=f"نص المادة {number}",
            )
            for number in numbers
        ],
        diagnostics=RetrievalDiagnosticsV1(
            article_numbers=numbers,
            article_count=len(numbers),
        ),
        elapsed_ms=0,
    )


def main() -> None:
    prompt = " ".join(
        GroundedAnswerGenerator._instructions().split()
    )

    for required in [
        "Never transfer a duty, right, restriction, or procedure",
        "Use the minimum sufficient set of retrieved articles",
        "map every sentence to a requested sub-question",
        "The answer must contain at least one visible inline citation",
    ]:
        assert required in prompt, required

    # Exactly one safe structured citation with no inline citation is repaired.
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=FakeClient(
            {
                "answer_ar": "الإجابة القانونية الصحيحة.",
                "key_points": [],
                "cited_article_numbers": [61],
                "limitations": [],
            }
        ),
    )
    result = generator.generate(
        retrieval([61]),
        include_debug=True,
    )
    assert result.status == "generated"
    assert result.answer_ar.endswith("[المادة 61]")
    assert result.cited_article_numbers == [61]
    assert result.debug["citation_repair_applied"] is True

    # Multiple structured citations with no inline placement remain rejected.
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=FakeClient(
            {
                "answer_ar": "حكمان قانونيان مختلفان.",
                "key_points": [],
                "cited_article_numbers": [39, 40],
                "limitations": [],
            }
        ),
    )
    result = generator.generate(
        retrieval([39, 40]),
        include_debug=True,
    )
    assert result.status == "insufficient_evidence"
    assert result.debug["citation_repair_applied"] is False

    # Unretrieved citations remain rejected.
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=FakeClient(
            {
                "answer_ar": "حكم غير مسموح [المادة 999].",
                "key_points": [],
                "cited_article_numbers": [999],
                "limitations": [],
            }
        ),
    )
    result = generator.generate(
        retrieval([61]),
        include_debug=True,
    )
    assert result.status == "insufficient_evidence"

    print("Stage 8-B5 checks passed.")
    print("Single-article missing inline citations are repaired")
    print("Multi-article missing citation placement still fails closed")
    print("Unretrieved citations remain rejected")
    print("Prompt enforces actor alignment and minimal evidence")


if __name__ == "__main__":
    main()
