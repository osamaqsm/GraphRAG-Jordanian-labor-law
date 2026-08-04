from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from app.generation_contract import GroundedAnswerResultV1
from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import (
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalEvidenceV1,
    RetrievalResultV1,
)


class FakeUsage:
    input_tokens = 100
    output_tokens = 40
    total_tokens = 140


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(payload, ensure_ascii=False)
        self.usage = FakeUsage()


class FakeResponses:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.responses = FakeResponses(payload)


def retrieval(
    behavior: str,
    *,
    article: bool = False,
) -> RetrievalResultV1:
    articles = []
    if article:
        articles = [
            RetrievalEvidenceV1(
                uri="http://example.org/jordan-labor-law#article_47",
                local_name="article_47",
                node_kind="Article",
                labels_ar=["المادة 47"],
                labels_en=[],
                article_number=47,
                text=(
                    "لا يجوز حسم أكثر من عشرة في المائة من أجر العامل "
                    "استيفاء لما يكون قد أقرضه صاحب العمل."
                ),
            )
        ]

    return RetrievalResultV1(
        question="كم يجوز حسمه من الأجر لاسترداد سلفة؟",
        decision=RetrievalDecisionV1(
            behavior=behavior,
            reason="سبب تجريبي",
            clarification_question_ar=(
                "ما نوع القرار المقصود؟"
                if behavior == "clarify"
                else ""
            ),
        ),
        embedding=RetrievalEmbeddingV1(
            model="text-embedding-3-small",
            dimensions=1536 if article else 0,
            input_tokens=10 if article else 0,
        ),
        articles=articles,
        diagnostics=RetrievalDiagnosticsV1(
            article_numbers=[47] if article else [],
            article_count=1 if article else 0,
        ),
        elapsed_ms=10,
    )


def main() -> None:
    source = inspect.getsource(
        __import__(
            "app.grounded_answer_generator",
            fromlist=["GroundedAnswerGenerator"],
        )
    ).lower()

    forbidden_dependencies = [
        "import weaviate",
        "retrievalservice",
        "retrievalonlypipeline",
        "graphtraversalservice",
    ]
    for dependency in forbidden_dependencies:
        assert dependency not in source, dependency

    valid_payload = {
        "answer_ar": (
            "يجوز الحسم ضمن الحد الوارد في النص المسترجع "
            "[المادة 47]"
        ),
        "key_points": ["الحسم مقيد بالحد القانوني [المادة 47]"],
        "cited_article_numbers": [47],
        "limitations": [],
    }
    valid_client = FakeClient(valid_payload)
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=valid_client,
    )

    clarify_result = generator.generate(retrieval("clarify"))
    assert clarify_result.status == "clarification_required"
    assert valid_client.responses.calls == 0

    abstain_result = generator.generate(retrieval("abstain"))
    assert abstain_result.status == "out_of_scope"
    assert valid_client.responses.calls == 0

    empty_result = generator.generate(retrieval("retrieve"))
    assert empty_result.status == "insufficient_evidence"
    assert valid_client.responses.calls == 0

    generated = generator.generate(retrieval("retrieve", article=True))
    assert generated.status == "generated"
    assert generated.grounded is True
    assert generated.cited_article_numbers == [47]
    assert generated.citations[0].article_number == 47
    assert valid_client.responses.calls == 1

    hallucinated_client = FakeClient(
        {
            "answer_ar": "إجابة غير مسموحة [المادة 999]",
            "key_points": [],
            "cited_article_numbers": [999],
            "limitations": [],
        }
    )
    hallucinated_generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=hallucinated_client,
    )
    rejected = hallucinated_generator.generate(
        retrieval("retrieve", article=True)
    )
    assert rejected.status == "insufficient_evidence"
    assert rejected.grounded is False
    assert rejected.citations == []

    dumped = GroundedAnswerResultV1.model_validate(
        generated.model_dump(mode="json")
    )
    assert dumped.schema_version == "generation.v1"

    cli_source = Path("scripts/generate_from_retrieval.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "retrievalonlypipeline" not in cli_source
    assert "retrievalservice" not in cli_source
    assert "weaviate" not in cli_source

    print("Stage 8-B offline checks passed.")
    print("Generation contract: generation.v1")
    print("Clarify/abstain do not call the model")
    print("Generator has no retrieval or Weaviate dependency")
    print("Unretrieved article citations are rejected")


if __name__ == "__main__":
    main()
