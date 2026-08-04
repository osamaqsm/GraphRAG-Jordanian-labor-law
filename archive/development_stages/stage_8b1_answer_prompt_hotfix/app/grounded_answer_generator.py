from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.generation_contract import (
    AnswerCitationV1,
    GenerationUsageV1,
    GroundedAnswerResultV1,
)
from app.retrieval_contract import RetrievalEvidenceV1, RetrievalResultV1


INLINE_CITATION_RE = re.compile(r"\[\s*المادة\s+(\d+)\s*\]")


def _setting(settings: Settings, name: str, env_name: str, default: Any) -> Any:
    value = getattr(settings, name, None)
    if value is not None and value != "":
        return value
    return os.getenv(env_name, default)


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _unique_numbers(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = int(value)
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_ar: str = Field(min_length=1)
    key_points: list[str]
    cited_article_numbers: list[int]
    limitations: list[str]


class GroundedAnswerGenerator:
    """
    Generate an Arabic answer from RetrievalResultV1 only.

    This module deliberately has no Weaviate, graph traversal, embedding,
    retrieval service, or reranking dependency.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAI | Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.enabled = _truthy(
            _setting(
                self.settings,
                "openai_answer_enabled",
                "OPENAI_ANSWER_ENABLED",
                "true",
            ),
            default=True,
        )
        self.model = str(
            _setting(
                self.settings,
                "openai_answer_model",
                "OPENAI_ANSWER_MODEL",
                _setting(
                    self.settings,
                    "openai_chat_model",
                    "OPENAI_CHAT_MODEL",
                    "gpt-5-nano",
                ),
            )
        )
        self.reasoning_effort = str(
            _setting(
                self.settings,
                "openai_answer_reasoning_effort",
                "OPENAI_ANSWER_REASONING_EFFORT",
                _setting(
                    self.settings,
                    "openai_reasoning_effort",
                    "OPENAI_REASONING_EFFORT",
                    "low",
                ),
            )
        )
        self.max_article_chars = max(
            500,
            int(
                _setting(
                    self.settings,
                    "openai_answer_article_char_limit",
                    "OPENAI_ANSWER_ARTICLE_CHAR_LIMIT",
                    3000,
                )
            ),
        )

        self.client = client
        if self.client is None and self.enabled:
            self.client = OpenAI(
                api_key=str(
                    _setting(
                        self.settings,
                        "openai_api_key",
                        "OPENAI_API_KEY",
                        "",
                    )
                ),
                timeout=float(
                    _setting(
                        self.settings,
                        "openai_timeout_seconds",
                        "OPENAI_TIMEOUT_SECONDS",
                        120,
                    )
                ),
                max_retries=int(
                    _setting(
                        self.settings,
                        "openai_max_retries",
                        "OPENAI_MAX_RETRIES",
                        3,
                    )
                ),
            )

    @staticmethod
    def _instructions() -> str:
        return r"""
You are a grounded Arabic legal-information answer generator.

You receive:
1. one user question; and
2. a closed list of retrieved provisions from the Jordanian Labor Law.

Your task is to answer the user's exact question clearly and naturally from
those provisions only.

HARD GROUNDING RULES

- Use only the supplied retrieved provisions.
- Do not use memory, general legal knowledge, web knowledge, or unstated facts.
- Do not invent an article, right, deadline, amount, procedure, authority,
  exception, remedy, or factual circumstance.
- Do not claim that a provision says something absent from its supplied text.
- If the retrieved text is materially insufficient, say exactly what is
  missing instead of completing the answer from outside knowledge.
- Distinguish the general legal rule from applying it to facts not supplied by
  the user.
- Never cite an article merely because it is related.

ARABIC ANSWER STYLE

- Answer in clear, natural Modern Standard Arabic.
- Start immediately with the legal rule or direct answer. Do not add an
  introduction.
- Prefer the terminology used in the supplied provision.
- For a simple single-rule question, write one concise paragraph of one to
  three sentences.
- State numbers, percentages, periods, and conditions explicitly.
- Avoid awkward wording, repetition, unnecessary hedging, and vague phrases
  such as "قد توجد حالات أخرى" unless those other cases are necessary to
  answer the question.
- Do not reproduce the entire article when a concise explanation is enough.
- Do not add a separate source line, bibliography, heading named "المصدر",
  or repeat the article citation after already citing the supporting sentence.
- Do not append a generic legal disclaimer or a statement that the answer is
  limited to the supplied text. Add a limitation only when a real missing fact
  or missing provision materially prevents a complete answer.

CITATION RULES

- Every legal claim must have an inline citation using exactly:
  [المادة N]
- N must be one of the supplied article numbers.
- Never cite an article that was not supplied.
- cited_article_numbers must contain the unique cited numbers in first-use
  order.
- Do not include uncited legal claims.
- Place the citation immediately after the sentence or clause it supports.

OUTPUT RULES

- answer_ar:
  Give the final user-facing answer only. It must be concise, direct, and
  self-contained. Do not include "المصدر:" or a references section.

- key_points:
  For a simple question answered by one rule, return an empty list.
  Use key_points only when the question contains multiple distinct issues or
  the answer genuinely benefits from two or more separate points.
  Every non-empty point must include an inline citation.

- cited_article_numbers:
  Include only cited supplied articles, in first-use order across answer_ar
  and then key_points.

- limitations:
  Return an empty list when the supplied provisions answer the question.
  Include only a specific material limitation that changes or prevents the
  answer. Never include generic boilerplate such as:
  "المعلومات مقتصرة على النص المرفق"
  or
  "لا يجوز توسيع الاستنتاج خارج النص".

QUALITY EXAMPLE

For a question asking for a statutory percentage, prefer a direct form such
as:

"يجوز لصاحب العمل استرداد السلفة من أجر العامل، بشرط ألا يزيد كل قسط يُحسم
لاستردادها على النسبة المحددة في النص [المادة N]."

Do not add a second paragraph saying "المصدر: المادة N".
""".strip()

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_ar": {"type": "string"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cited_article_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "answer_ar",
                "key_points",
                "cited_article_numbers",
                "limitations",
            ],
        }

    def _non_retrieval_result(
        self,
        retrieval: RetrievalResultV1,
        *,
        status: str,
        answer_ar: str,
        started: float,
        warning: str | None = None,
        include_debug: bool = False,
    ) -> GroundedAnswerResultV1:
        warnings = [warning] if warning else []
        return GroundedAnswerResultV1(
            status=status,
            question=retrieval.question,
            answer_ar=answer_ar,
            grounded=False,
            warnings=warnings,
            elapsed_ms=max(
                0,
                round((time.perf_counter() - started) * 1000),
            ),
            debug=(
                {
                    "decision": retrieval.decision.model_dump(mode="json"),
                    "model_called": False,
                }
                if include_debug
                else None
            ),
        )

    def _article_payload(
        self,
        articles: list[RetrievalEvidenceV1],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for article in articles:
            if article.article_number is None:
                continue
            payload.append(
                {
                    "article_number": article.article_number,
                    "labels_ar": article.labels_ar,
                    "text": article.text[: self.max_article_chars],
                    "graph_supported": article.graph_supported,
                    "support_paths": article.support_paths[:5],
                }
            )
        return payload

    @staticmethod
    def _citation(
        article: RetrievalEvidenceV1,
    ) -> AnswerCitationV1:
        label = (
            article.labels_ar[0]
            if article.labels_ar
            else f"المادة {article.article_number}"
        )
        excerpt = " ".join(article.text.split())[:500]
        return AnswerCitationV1(
            article_number=int(article.article_number),
            label_ar=label,
            uri=article.uri,
            excerpt=excerpt,
        )

    def generate(
        self,
        retrieval: RetrievalResultV1 | dict[str, Any],
        *,
        include_debug: bool = False,
    ) -> GroundedAnswerResultV1:
        started = time.perf_counter()
        if not isinstance(retrieval, RetrievalResultV1):
            retrieval = RetrievalResultV1.model_validate(retrieval)

        behavior = retrieval.decision.behavior

        if behavior == "clarify":
            question = (
                retrieval.decision.clarification_question_ar.strip()
                or retrieval.decision.reason.strip()
                or "يرجى توضيح الوقائع اللازمة قبل تحديد النص القانوني."
            )
            return self._non_retrieval_result(
                retrieval,
                status="clarification_required",
                answer_ar=question,
                started=started,
                include_debug=include_debug,
            )

        if behavior == "abstain":
            reason = retrieval.decision.reason.strip()
            answer = (
                reason
                or "السؤال خارج نطاق قانون العمل الأردني المتاح في هذا النظام."
            )
            return self._non_retrieval_result(
                retrieval,
                status="out_of_scope",
                answer_ar=answer,
                started=started,
                include_debug=include_debug,
            )

        article_map = {
            int(article.article_number): article
            for article in retrieval.articles
            if article.article_number is not None
        }
        allowed_numbers = set(article_map)

        if not allowed_numbers:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar=(
                    "لم تُسترجع مادة قانونية يمكن الاعتماد عليها لإنشاء "
                    "إجابة موثقة."
                ),
                started=started,
                warning="Retrieval returned no citable legal article.",
                include_debug=include_debug,
            )

        if not self.enabled or self.client is None:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="توليد الإجابة معطل حالياً.",
                started=started,
                warning="OPENAI_ANSWER_ENABLED is false or no client is available.",
                include_debug=include_debug,
            )

        payload = {
            "user_question": retrieval.question,
            "retrieval_schema_version": retrieval.schema_version,
            "retrieved_articles": self._article_payload(retrieval.articles),
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=self._instructions(),
            input=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grounded_jordan_labor_answer",
                    "schema": self._response_schema(),
                    "strict": True,
                }
            },
            store=False,
        )

        output_text = str(getattr(response, "output_text", "") or "")
        if not output_text:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر إنشاء إجابة موثقة من المواد المسترجعة.",
                started=started,
                warning="OpenAI returned no answer output.",
                include_debug=include_debug,
            )

        draft = AnswerDraft.model_validate(json.loads(output_text))
        cited_numbers = _unique_numbers(draft.cited_article_numbers)
        answer_inline_numbers = [
            int(value)
            for value in INLINE_CITATION_RE.findall(draft.answer_ar)
        ]
        key_point_inline_numbers: list[int] = []
        uncited_key_points: list[int] = []
        for index, key_point in enumerate(draft.key_points, start=1):
            point_numbers = [
                int(value)
                for value in INLINE_CITATION_RE.findall(key_point)
            ]
            if key_point.strip() and not point_numbers:
                uncited_key_points.append(index)
            key_point_inline_numbers.extend(point_numbers)

        inline_numbers = _unique_numbers(
            answer_inline_numbers + key_point_inline_numbers
        )

        invalid_numbers = sorted(
            (set(cited_numbers) | set(inline_numbers)) - allowed_numbers
        )
        citation_mismatch = cited_numbers != inline_numbers

        if (
            invalid_numbers
            or citation_mismatch
            or not cited_numbers
            or uncited_key_points
        ):
            reasons: list[str] = []
            if invalid_numbers:
                reasons.append(
                    "Model cited unretrieved articles: "
                    + ", ".join(str(value) for value in invalid_numbers)
                )
            if citation_mismatch:
                reasons.append(
                    "Inline citations do not match cited_article_numbers."
                )
            if not cited_numbers:
                reasons.append("Generated answer contains no valid citation.")
            if uncited_key_points:
                reasons.append(
                    "Uncited key_points indexes: "
                    + ", ".join(str(value) for value in uncited_key_points)
                )

            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar=(
                    "تعذر اعتماد الإجابة لأن توثيقها لم يطابق المواد "
                    "المسترجعة."
                ),
                started=started,
                warning=" ".join(reasons),
                include_debug=include_debug,
            )

        citations = [
            self._citation(article_map[number])
            for number in cited_numbers
        ]

        usage_obj = getattr(response, "usage", None)
        input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage_obj, "total_tokens", 0)
            or input_tokens + output_tokens
        )

        debug: dict[str, Any] | None = None
        if include_debug:
            debug = {
                "model_called": True,
                "allowed_article_numbers": sorted(allowed_numbers),
                "inline_article_numbers": inline_numbers,
                "draft": draft.model_dump(mode="json"),
            }

        return GroundedAnswerResultV1(
            status="generated",
            question=retrieval.question,
            answer_ar=draft.answer_ar.strip(),
            key_points=[
                value.strip()
                for value in draft.key_points
                if value.strip()
            ],
            citations=citations,
            cited_article_numbers=cited_numbers,
            grounded=True,
            warnings=[
                value.strip()
                for value in draft.limitations
                if value.strip()
            ],
            model=self.model,
            usage=GenerationUsageV1(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            elapsed_ms=max(
                0,
                round((time.perf_counter() - started) * 1000),
            ),
            debug=debug,
        )
