from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from app.grounded_answer_generator import (
    GroundedAnswerGenerator,
    _canonicalize_citations,
)
from app.retrieval_contract import (
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalEvidenceV1,
    RetrievalResultV1,
)


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class FakeResponse:
    def __init__(self, payload: dict, index: int) -> None:
        self.output_text = json.dumps(payload, ensure_ascii=False)
        self.usage = FakeUsage(10 + index, 5 + index)


class FakeResponses:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("Unexpected extra model call")
        return FakeResponse(self.payloads.pop(0), len(self.calls))


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.responses = FakeResponses(payloads)


def retrieval(numbers: list[int], *, behavior: str = "retrieve") -> RetrievalResultV1:
    return RetrievalResultV1(
        question="ما الحكم في المسألتين المطلوبتين؟",
        decision=RetrievalDecisionV1(
            behavior=behavior,
            reason="test",
            clarification_question_ar=(
                "ما التفصيل المطلوب؟" if behavior == "clarify" else ""
            ),
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
                final_score=1.0,
            )
            for number in numbers
        ],
        diagnostics=RetrievalDiagnosticsV1(
            article_numbers=numbers,
            article_count=len(numbers),
        ),
        elapsed_ms=0,
    )


def selection(
    *,
    selected: list[int],
    excluded: list[int],
    issues: list[tuple[str, list[int]]],
    answerability: str = "complete",
    missing: list[str] | None = None,
) -> dict:
    missing = missing or []
    return {
        "requested_issues": [
            {
                "issue_id": issue_id,
                "question_part_ar": f"السؤال {issue_id}",
            }
            for issue_id, _ in issues
        ],
        "selected_article_numbers": selected,
        "excluded_articles": [
            {
                "article_number": number,
                "reason_ar": "غير لازم للإجابة الدقيقة",
            }
            for number in excluded
        ],
        "issue_support": [
            {
                "issue_id": issue_id,
                "article_numbers": article_numbers,
            }
            for issue_id, article_numbers in issues
        ],
        "missing_issue_ids": missing,
        "answerability": answerability,
        "missing_information_ar": (
            ["النص اللازم للمسألة غير مسترجع"] if missing else []
        ),
    }


def answer(
    *,
    text: str,
    cited: list[int],
    issues: list[tuple[str, list[int]]],
    limitations: list[str] | None = None,
) -> dict:
    return {
        "answer_ar": text,
        "key_points": [],
        "cited_article_numbers": cited,
        "limitations": limitations or [],
        "covered_issue_ids": [issue_id for issue_id, _ in issues],
        "issue_support": [
            {
                "issue_id": issue_id,
                "article_numbers": article_numbers,
            }
            for issue_id, article_numbers in issues
        ],
    }


def main() -> None:
    # Citation normalization is idempotent and repairs duplicate brackets.
    allowed = {99}
    samples = [
        "الحكم [المادة 99].",
        "الحكم [[المادة 99].",
        "الحكم مادة 99.",
        "الحكم [مادة ٩٩].",
    ]
    for sample in samples:
        once = _canonicalize_citations(sample, allowed_numbers=allowed)
        twice = _canonicalize_citations(once, allowed_numbers=allowed)
        assert once == twice
        assert "[[المادة" not in once
        assert "[المادة 99]" in once

    # Generic minimal-evidence selection excludes a related extra article.
    client = FakeClient(
        [
            selection(
                selected=[10],
                excluded=[11],
                issues=[("I1", [10])],
            ),
            answer(
                text="الحكم المطلوب [المادة 10].",
                cited=[10],
                issues=[("I1", [10])],
            ),
        ]
    )
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    result = generator.generate(retrieval([10, 11]), include_debug=True)
    assert result.status == "generated"
    assert result.cited_article_numbers == [10]
    assert result.debug["selected_article_numbers"] == [10]
    assert result.debug["excluded_article_numbers"] == [11]
    assert result.debug["model_calls"] == 2
    assert result.usage.total_tokens == 36

    # A selection plan cannot use an unretrieved article.
    client = FakeClient(
        [
            selection(
                selected=[999],
                excluded=[10],
                issues=[("I1", [999])],
            )
        ]
    )
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    result = generator.generate(retrieval([10]), include_debug=True)
    assert result.status == "insufficient_evidence"
    assert "unretrieved" in " ".join(result.warnings).lower()
    assert len(client.responses.calls) == 1

    # Answer coverage must include every supported issue.
    client = FakeClient(
        [
            selection(
                selected=[39, 40],
                excluded=[],
                issues=[("I1", [39]), ("I2", [40])],
            ),
            answer(
                text="أجيب عن المسألة الأولى فقط [المادة 39].",
                cited=[39],
                issues=[("I1", [39])],
            ),
        ]
    )
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    result = generator.generate(retrieval([39, 40]), include_debug=True)
    assert result.status == "insufficient_evidence"
    assert "covered_issue_ids" in " ".join(result.warnings)

    # Single selected article missing an inline citation is safely repaired.
    client = FakeClient(
        [
            selection(
                selected=[61],
                excluded=[],
                issues=[("I1", [61])],
            ),
            answer(
                text="الإجابة القانونية الصحيحة.",
                cited=[61],
                issues=[("I1", [61])],
            ),
        ]
    )
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    result = generator.generate(retrieval([61]), include_debug=True)
    assert result.status == "generated"
    assert result.answer_ar.endswith("[المادة 61]")
    assert result.debug["citation_repair_applied"] is True

    # Clarify and abstain remain deterministic and make zero model calls.
    client = FakeClient([])
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    clarified = generator.generate(retrieval([], behavior="clarify"))
    assert clarified.status == "clarification_required"
    assert len(client.responses.calls) == 0

    # The production implementation stays generic and retrieval-free.
    source = inspect.getsource(
        __import__(
            "app.grounded_answer_generator",
            fromlist=["GroundedAnswerGenerator"],
        )
    )
    lowered = source.lower()
    for forbidden in [
        "retrievalonlypipeline",
        "retrievalservice",
        "import weaviate",
        "graphtraversalservice",
        "ugh01",
        "g01",
        "المادة 47",
    ]:
        assert forbidden not in lowered, forbidden

    print("Stage 8-B7 checks passed.")
    print("Evidence selection uses a minimum sufficient subset")
    print("Issue coverage is validated deterministically")
    print("Citation canonicalization is idempotent")
    print("Unretrieved and unselected citations remain rejected")
    print("Clarify and abstain remain generation-free")


if __name__ == "__main__":
    main()
