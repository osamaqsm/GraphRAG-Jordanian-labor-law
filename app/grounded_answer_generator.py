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


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
INLINE_CITATION_RE = re.compile(
    r"\[\s*المادة\s+([0-9٠-٩]+)\s*\]"
)
ARTICLE_MENTION_RE = re.compile(
    r"(?<![\w\u0600-\u06ff])"
    r"\[*\s*(?:المادة|مادة)\s+([0-9٠-٩]+)\s*\]*"
)


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


def _to_int(value: str) -> int:
    return int(value.translate(ARABIC_DIGITS))


def _canonicalize_citations(text: str, allowed_numbers: set[int]) -> str:
    """Normalize article citation variants without changing their meaning."""

    def replacement(match: re.Match[str]) -> str:
        number = _to_int(match.group(1))
        if number not in allowed_numbers:
            return match.group(0)
        return f"[المادة {number}]"

    return ARTICLE_MENTION_RE.sub(replacement, text)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_ar: str = Field(min_length=1)
    cited_article_numbers: list[int]
    limitations: list[str]


class GroundedAnswerGenerator:
    """
    One-call generation boundary.

    Input: one completed retrieval.v1 object.
    Output: one generation.v1 object.

    This module has no retrieval, vector database, graph traversal, embedding,
    reranking, evidence-selection, or answer-verification dependency.
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
                "gpt-5-nano",
            )
        )
        self.reasoning_effort = str(
            _setting(
                self.settings,
                "openai_answer_reasoning_effort",
                "OPENAI_ANSWER_REASONING_EFFORT",
                "low",
            )
        )
        self.verbosity = str(
            _setting(
                self.settings,
                "openai_answer_verbosity",
                "OPENAI_ANSWER_VERBOSITY",
                "low",
            )
        )
        self.max_output_tokens = max(
            500,
            int(
                _setting(
                    self.settings,
                    "openai_answer_max_output_tokens",
                    "OPENAI_ANSWER_MAX_OUTPUT_TOKENS",
                    1800,
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
                        "openai_answer_timeout_seconds",
                        "OPENAI_ANSWER_TIMEOUT_SECONDS",
                        _setting(
                            self.settings,
                            "openai_timeout_seconds",
                            "OPENAI_TIMEOUT_SECONDS",
                            120,
                        ),
                    )
                ),
                max_retries=int(
                    _setting(
                        self.settings,
                        "openai_answer_max_retries",
                        "OPENAI_ANSWER_MAX_RETRIES",
                        _setting(
                            self.settings,
                            "openai_max_retries",
                            "OPENAI_MAX_RETRIES",
                            2,
                        ),
                    )
                ),
            )

    @staticmethod
    def _instructions() -> str:
        return r"""
You generate a concise Arabic legal-information answer from one completed
retrieval result.

INPUT

The user message is one JSON object with exactly these top-level fields:
- user_question: the exact question asked by the user.
- retrieval_result: the complete output from the previous retrieval step,
  following retrieval.v1.

Use retrieval_result.articles[].text as the only legal evidence. Other fields
in retrieval_result are metadata and must not be treated as legal text.

RULES

1. Answer the exact user_question directly in clear Modern Standard Arabic.
2. Use only the supplied article texts. Do not use memory, outside legal
   knowledge, web knowledge, or unstated facts.
3. Use only articles necessary for the question. Ignore retrieved articles that
   are related but do not answer a requested issue.
4. Preserve the correct legal actor, every material condition and exception,
   and every number, percentage, duration, deadline, and amount.
5. Answer every distinct part of a multi-part question.
6. When the question asks for a numerical result and supplies a quantity,
   perform the simple calculation explicitly from the statutory number.
7. If the retrieved articles do not contain enough evidence, answer only the
   supported part and describe the exact missing information in limitations.
8. Do not reproduce entire articles, add unrelated rules, add a source heading,
   or add a generic legal disclaimer.
9. Keep a simple answer to one short paragraph. Use short separate paragraphs
   only when the question contains multiple distinct parts.

CITATIONS

- Put an inline citation immediately after every legal sentence using exactly
  [المادة N].
- N must be an article_number present in retrieval_result.articles.
- Never cite an article that was not supplied.
- cited_article_numbers must list the unique cited articles in first-use order.

OUTPUT

- answer_ar: only the final user-facing answer.
- cited_article_numbers: unique cited article numbers in first-use order.
- limitations: an empty list when the retrieved evidence is sufficient;
  otherwise only specific missing evidence. No generic disclaimer.

Before returning, check that the answer is direct, all requested parts are
covered, legal actors and conditions are correct, numbers are exact, and every
legal sentence has a valid inline citation.
""".strip()

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer_ar": {"type": "string"},
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
                "cited_article_numbers",
                "limitations",
            ],
        }

    @staticmethod
    def _citation(article: RetrievalEvidenceV1) -> AnswerCitationV1:
        label = (
            article.labels_ar[0]
            if article.labels_ar
            else f"المادة {article.article_number}"
        )
        return AnswerCitationV1(
            article_number=int(article.article_number),
            label_ar=label,
            uri=article.uri,
            excerpt=" ".join(article.text.split())[:500],
        )

    def _non_model_result(
        self,
        retrieval: RetrievalResultV1,
        *,
        status: str,
        answer_ar: str,
        started: float,
        warning: str | None = None,
        include_debug: bool = False,
        model_called: bool = False,
        debug_details: dict[str, Any] | None = None,
    ) -> GroundedAnswerResultV1:
        debug: dict[str, Any] | None = None
        if include_debug:
            debug = {
                "model_called": model_called,
                "decision": retrieval.decision.model_dump(mode="json"),
            }
            if debug_details:
                debug.update(debug_details)

        return GroundedAnswerResultV1(
            status=status,
            question=retrieval.question,
            answer_ar=answer_ar,
            grounded=False,
            warnings=[warning] if warning else [],
            elapsed_ms=max(0, round((time.perf_counter() - started) * 1000)),
            debug=debug,
        )

    def _request_payload(self, retrieval: RetrievalResultV1) -> dict[str, Any]:
        # Deliberately include both the exact user question and the complete
        # previous-step output. This is the only model input for Step 8.
        return {
            "user_question": retrieval.question,
            "retrieval_result": retrieval.model_dump(mode="json"),
        }

    def _request_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": self._instructions(),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    ],
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "verbosity": self.verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "grounded_jordan_labor_answer",
                    "schema": self._response_schema(),
                    "strict": True,
                },
            },
            "store": False,
        }

    def generate(
        self,
        retrieval: RetrievalResultV1 | dict[str, Any],
        *,
        include_debug: bool = False,
    ) -> GroundedAnswerResultV1:
        started = time.perf_counter()
        if not isinstance(retrieval, RetrievalResultV1):
            retrieval = RetrievalResultV1.model_validate(retrieval)

        if retrieval.decision.behavior == "clarify":
            answer = (
                retrieval.decision.clarification_question_ar.strip()
                or retrieval.decision.reason.strip()
                or "يرجى توضيح الوقائع اللازمة للإجابة."
            )
            return self._non_model_result(
                retrieval,
                status="clarification_required",
                answer_ar=answer,
                started=started,
                include_debug=include_debug,
            )

        if retrieval.decision.behavior == "abstain":
            answer = (
                retrieval.decision.reason.strip()
                or "السؤال خارج نطاق قانون العمل الأردني المتاح في النظام."
            )
            return self._non_model_result(
                retrieval,
                status="out_of_scope",
                answer_ar=answer,
                started=started,
                include_debug=include_debug,
            )

        article_map = {
            int(article.article_number): article
            for article in retrieval.articles
            if article.article_number is not None and article.text.strip()
        }
        allowed_numbers = set(article_map)

        if not allowed_numbers:
            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="لم تُسترجع مادة قانونية يمكن الاعتماد عليها للإجابة.",
                started=started,
                warning="Retrieval returned no citable article text.",
                include_debug=include_debug,
            )

        if not self.enabled or self.client is None:
            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="توليد الإجابة معطل حالياً.",
                started=started,
                warning="OPENAI_ANSWER_ENABLED is false or no client is available.",
                include_debug=include_debug,
            )

        payload = self._request_payload(retrieval)
        request_kwargs = self._request_kwargs(payload)
        response = self.client.responses.create(**request_kwargs)

        output_text = str(getattr(response, "output_text", "") or "").strip()
        if not output_text:
            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر إنشاء إجابة من المواد المسترجعة.",
                started=started,
                warning="OpenAI returned no output text.",
                include_debug=include_debug,
                model_called=True,
            )

        draft = AnswerDraft.model_validate(json.loads(output_text))
        answer = _canonicalize_citations(draft.answer_ar, allowed_numbers)
        structured_numbers = _unique_numbers(draft.cited_article_numbers)
        inline_numbers = _unique_numbers(
            [_to_int(value) for value in INLINE_CITATION_RE.findall(answer)]
        )
        mentioned_numbers = _unique_numbers(
            [_to_int(value) for value in ARTICLE_MENTION_RE.findall(answer)]
        )

        citation_repair_applied = False
        if (
            not inline_numbers
            and len(structured_numbers) == 1
            and structured_numbers[0] in allowed_numbers
        ):
            answer = answer.rstrip() + f" [المادة {structured_numbers[0]}]"
            inline_numbers = structured_numbers.copy()
            mentioned_numbers = _unique_numbers(
                mentioned_numbers + structured_numbers
            )
            citation_repair_applied = True

        if not structured_numbers and inline_numbers:
            structured_numbers = inline_numbers.copy()
        if set(structured_numbers) == set(inline_numbers):
            structured_numbers = inline_numbers.copy()

        invalid_numbers = sorted(
            (set(structured_numbers) | set(inline_numbers) | set(mentioned_numbers))
            - allowed_numbers
        )
        mismatch = set(structured_numbers) != set(inline_numbers)

        if invalid_numbers or mismatch or not structured_numbers:
            reasons: list[str] = []
            if invalid_numbers:
                reasons.append(
                    "Answer referenced unretrieved articles: "
                    + ", ".join(str(value) for value in invalid_numbers)
                )
            if mismatch:
                reasons.append(
                    "Inline citations do not match cited_article_numbers."
                )
            if not structured_numbers:
                reasons.append("Answer contains no valid article citation.")

            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر اعتماد الإجابة لأن توثيقها غير صالح.",
                started=started,
                warning=" ".join(reasons),
                include_debug=include_debug,
                model_called=True,
                debug_details={
                    "allowed_article_numbers": sorted(allowed_numbers),
                    "rejected_draft": draft.model_dump(mode="json"),
                    "normalized_answer_ar": answer,
                },
            )

        citations = [
            self._citation(article_map[number])
            for number in structured_numbers
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
                "model_calls": 1,
                "input_included_exact_user_question": (
                    payload["user_question"] == retrieval.question
                ),
                "input_included_complete_retrieval_result": (
                    payload["retrieval_result"]
                    == retrieval.model_dump(mode="json")
                ),
                "model_call_config": {
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "verbosity": self.verbosity,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature_sent": False,
                    "top_p_sent": False,
                    "store": False,
                    "structured_output_strict": True,
                },
                "allowed_article_numbers": sorted(allowed_numbers),
                "cited_article_numbers": structured_numbers,
                "citation_repair_applied": citation_repair_applied,
                "draft": draft.model_dump(mode="json"),
            }

        return GroundedAnswerResultV1(
            status="generated",
            question=retrieval.question,
            answer_ar=answer.strip(),
            key_points=[],
            citations=citations,
            cited_article_numbers=structured_numbers,
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
            elapsed_ms=max(0, round((time.perf_counter() - started) * 1000)),
            debug=debug,
        )
