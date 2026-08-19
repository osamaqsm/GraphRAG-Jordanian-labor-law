from __future__ import annotations

import os
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.llm_provider import build_pipeline_llm
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

BRACKETED_CITATION_RE = re.compile(
    r"\[+\s*(?:المادة|مادة)\s+([0-9٠-٩]+)\s*\]+"
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
    """Normalize valid article citation variants without changing meaning."""

    def replacement(match: re.Match[str]) -> str:
        number = _to_int(match.group(1))
        if number not in allowed_numbers:
            return match.group(0)
        return f"[المادة {number}]"

    return ARTICLE_MENTION_RE.sub(replacement, text)


def _normalize_bracketed_citations(text: str) -> str:
    """Normalize malformed bracketed citations such as [[المادة 5]."""

    def replacement(match: re.Match[str]) -> str:
        return f"[المادة {_to_int(match.group(1))}]"

    normalized = BRACKETED_CITATION_RE.sub(replacement, text)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\s+([،؛,.!?])", r"\1", normalized)
    return normalized.strip()


def _citation_state(
    answer: str,
    structured_numbers: list[int],
    allowed_numbers: set[int],
) -> dict[str, Any]:
    """Return a general citation-validation state for one generated draft."""

    normalized_answer = _normalize_bracketed_citations(answer)
    normalized_answer = _canonicalize_citations(
        normalized_answer,
        allowed_numbers,
    )

    inline_numbers = _unique_numbers(
        [_to_int(value) for value in INLINE_CITATION_RE.findall(normalized_answer)]
    )
    mentioned_numbers = _unique_numbers(
        [_to_int(value) for value in ARTICLE_MENTION_RE.findall(normalized_answer)]
    )
    structured_numbers = _unique_numbers(structured_numbers)

    invalid_inline = sorted(set(inline_numbers) - allowed_numbers)
    invalid_mentions = sorted(set(mentioned_numbers) - allowed_numbers)
    invalid_structured = sorted(set(structured_numbers) - allowed_numbers)

    valid_inline = [
        number for number in inline_numbers if number in allowed_numbers
    ]
    valid_structured = [
        number for number in structured_numbers if number in allowed_numbers
    ]

    # Inline citations are the source of truth when present. Otherwise the
    # structured list can be repaired into inline citations.
    final_numbers = valid_inline or valid_structured

    needs_retry = bool(
        invalid_inline
        or invalid_mentions
        or invalid_structured
        or not final_numbers
    )

    return {
        "answer": normalized_answer,
        "inline_numbers": inline_numbers,
        "mentioned_numbers": mentioned_numbers,
        "structured_numbers": structured_numbers,
        "valid_inline": valid_inline,
        "valid_structured": valid_structured,
        "final_numbers": final_numbers,
        "invalid_inline": invalid_inline,
        "invalid_mentions": invalid_mentions,
        "invalid_structured": invalid_structured,
        "needs_retry": needs_retry,
    }


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
        client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        strict_value = os.getenv(
            "PIPELINE_STRICT_EVALUATION",
            "false",
        ).strip().lower()
        self.strict_evaluation = strict_value not in {
            "0", "false", "no", "off", ""
        }

        self.enabled = _truthy(
            _setting(
                self.settings,
                "answer_enabled",
                "PIPELINE_ANSWER_ENABLED",
                _setting(
                    self.settings,
                    "openai_answer_enabled",
                    "OPENAI_ANSWER_ENABLED",
                    "true",
                ),
            ),
            default=True,
        )

        self.provider = str(
            getattr(
                self.settings,
                "pipeline_llm_provider",
                "openai",
            )
        ).strip().lower()

        self.model = str(
            getattr(
                self.settings,
                "pipeline_llm_model",
                "gpt-5-nano",
            )
        ).strip()

        self.max_output_tokens = max(
            500,
            int(
                getattr(
                    self.settings,
                    "generator_max_output_tokens",
                    3000,
                )
            ),
        )

        self.llm = client
        self.initialization_error: str | None = None

        if self.llm is None and self.enabled:
            try:
                self.llm = build_pipeline_llm(self.settings)
            except Exception as exc:
                self.llm = None
                self.initialization_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _instructions() -> str:
        return r"""
You generate a concise Arabic legal-information answer from one completed
retrieval result.

INPUT

The user message is one JSON object with exactly these top-level fields:
- user_question: the exact question asked by the user.
- retrieval_evidence: a compact provider-neutral evidence view derived from the
  exact retrieval.v1 object produced by the previous step.

Use retrieval_evidence.articles[].text as the only legal evidence. The decision
field is routing metadata and must not be treated as legal text.

RULES

1. Answer the exact user_question directly in clear Modern Standard Arabic.
2. Use only retrieval_result.articles[].text. Do not use memory, outside legal
   knowledge, web knowledge, metadata, graph paths, scores, or unstated facts.
3. First split the question internally into its independently requested parts.
   For each part, identify the article text that directly answers it.
4. Cover every requested part, but do not summarize every retrieved article.
   Retrieved articles are candidates, not a checklist that must all be used.
5. Prefer the smallest sufficient evidence set. Ignore any article that supplies
   only background, a neighbouring rule, or an unrequested consequence.
6. When the first-ranked articles directly answer the question, prioritize them,
   but always follow the statutory text rather than ranking alone.
7. Preserve the correct legal actor, every material condition and exception,
   and every number, percentage, duration, deadline, amount, procedural step,
   authority, and penalty expressly requested.
8. For multi-part questions, organize the answer by the question's parts. Do not
   replace a requested part with a broadly related rule.
9. When a supplied article contains both requested and unrequested material,
   include only the requested material.
10. When the question asks for a numerical result and supplies a quantity,
    perform the simple calculation explicitly from the statutory number.
11. If the retrieved articles do not contain enough evidence for a requested
    part, answer the supported parts and state the precise missing part in
    limitations. Never fill the gap from memory.
12. Do not reproduce entire articles, add unrelated rules, add a source heading,
    or add a generic legal disclaimer.
13. Keep a simple answer to one short paragraph. Use short separate paragraphs
    or compact numbered parts only when the question contains multiple distinct
    parts.
14. If the retrieval result indicates that the question is out of scope, write
    the entire user-facing answer in Arabic only, without English words.

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

Before returning, perform this final check:
- every requested part is explicitly answered or named in limitations;
- every included legal statement directly serves the question;
- no retrieved article was used merely because it was available;
- legal actors, conditions, exceptions, procedures, and numbers are exact;
- every legal sentence has a valid inline citation immediately after it.
""".strip()

    @staticmethod
    def _retry_instructions(allowed_numbers: list[int]) -> str:
        allowed = ", ".join(str(number) for number in allowed_numbers)
        return (
            GroundedAnswerGenerator._instructions()
            + "\n\nRETRY CORRECTION\n"
            + "The previous draft failed citation validation. Rewrite the answer "
              "from scratch. Use only these article numbers in citations: "
            + allowed
            + ". Every legal sentence must end with a valid citation in exactly "
              "this form: [المادة N]. Do not mention or cite any other article "
              "number. Ensure cited_article_numbers exactly matches the unique "
              "inline citations in first-use order."
        )

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
        # /generate still consumes the exact retrieval.v1 object and never
        # reruns retrieval.  The LLM itself receives only the fields needed for
        # grounded answering so hosted and 8K-context local models see the same
        # compact semantic input rather than provider-irrelevant graph metadata.
        evidence_articles: list[dict[str, Any]] = []
        for article in retrieval.articles:
            if article.article_number is None or not article.text.strip():
                continue
            evidence_articles.append(
                {
                    "article_number": int(article.article_number),
                    "labels_ar": list(article.labels_ar),
                    "text": article.text,
                }
            )

        return {
            "user_question": retrieval.question,
            "retrieval_evidence": {
                "decision": {
                    "behavior": retrieval.decision.behavior,
                    "reason": retrieval.decision.reason,
                },
                "articles": evidence_articles,
            },
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

        if retrieval.decision.behavior == "abstain":
            # User-facing out-of-scope behavior is deterministic and
            # provider-independent. Keep the planner's reason only as
            # diagnostic metadata so every evaluated model receives the same
            # final OOS response policy.
            answer = (
                "السؤال خارج نطاق قانون العمل الأردني "
                "الممثل في قاعدة المعرفة المتاحة في النظام."
            )
            return self._non_model_result(
                retrieval,
                status="out_of_scope",
                answer_ar=answer,
                started=started,
                include_debug=include_debug,
                debug_details={
                    "routing_reason": retrieval.decision.reason.strip(),
                },
            )

        article_map = {
            int(article.article_number): article
            for article in retrieval.articles
            if article.article_number is not None
            and article.text.strip()
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

        if not self.enabled or self.llm is None:
            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: answer generator is unavailable. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={self.initialization_error}"
                )
            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="توليد الإجابة معطل حالياً.",
                started=started,
                warning=(
                    "PIPELINE_ANSWER_ENABLED is false or no pipeline "
                    "LLM provider is available."
                ),
                include_debug=include_debug,
            )

        payload = self._request_payload(retrieval)

        try:
            result = self.llm.generate_structured(
                instructions=self._instructions(),
                payload=payload,
                response_model=AnswerDraft,
                schema_name="grounded_jordan_labor_answer",
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: answer generation failed. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={type(exc).__name__}: {exc}"
                ) from exc
            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر إنشاء إجابة من المواد المسترجعة.",
                started=started,
                warning=(
                    "Pipeline answer generation failed. "
                    f"Provider={self.provider} Model={self.model} Error={exc}"
                ),
                include_debug=include_debug,
                model_called=True,
            )

        draft = AnswerDraft.model_validate(result.data)
        initial_draft = draft

        state = _citation_state(
            draft.answer_ar,
            draft.cited_article_numbers,
            allowed_numbers,
        )

        retry_applied = False
        retry_result = None
        retry_draft = None

        if state["needs_retry"]:
            retry_applied = True

            try:
                retry_result = self.llm.generate_structured(
                    instructions=self._retry_instructions(
                        sorted(allowed_numbers)
                    ),
                    payload=payload,
                    response_model=AnswerDraft,
                    schema_name="grounded_jordan_labor_answer_retry",
                    max_output_tokens=self.max_output_tokens,
                )

                retry_draft = AnswerDraft.model_validate(
                    retry_result.data
                )

                state = _citation_state(
                    retry_draft.answer_ar,
                    retry_draft.cited_article_numbers,
                    allowed_numbers,
                )
                draft = retry_draft

            except Exception as exc:
                if self.strict_evaluation:
                    raise RuntimeError(
                        "Strict evaluation aborted: citation-repair retry failed. "
                        f"Provider={self.provider} Model={self.model} "
                        f"Error={type(exc).__name__}: {exc}"
                    ) from exc
                retry_result = None
                retry_draft = None

        answer = state["answer"]
        structured_numbers = list(state["final_numbers"])

        citation_repair_applied = (
            answer != draft.answer_ar
            or set(structured_numbers)
            != set(
                _unique_numbers(
                    draft.cited_article_numbers
                )
            )
        )

        if not state["valid_inline"] and structured_numbers:
            suffix = "".join(
                f"[المادة {number}]"
                for number in structured_numbers
            )
            answer = answer.rstrip() + " " + suffix
            citation_repair_applied = True

        if state["needs_retry"] or not structured_numbers:
            reasons: list[str] = []

            invalid_numbers = sorted(
                set(state["invalid_inline"])
                | set(state["invalid_mentions"])
                | set(state["invalid_structured"])
            )

            if invalid_numbers:
                reasons.append(
                    "Answer referenced unretrieved articles: "
                    + ", ".join(
                        str(value)
                        for value in invalid_numbers
                    )
                )

            if not structured_numbers:
                reasons.append(
                    "Answer contains no valid article citation."
                )

            if retry_applied:
                reasons.append(
                    "Citation repair retry failed."
                )

            return self._non_model_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر اعتماد الإجابة لأن توثيقها غير صالح.",
                started=started,
                warning=" ".join(reasons),
                include_debug=include_debug,
                model_called=True,
                debug_details={
                    "provider": self.provider,
                    "model": self.model,
                    "allowed_article_numbers": sorted(
                        allowed_numbers
                    ),
                    "initial_draft": (
                        initial_draft.model_dump(
                            mode="json"
                        )
                    ),
                    "retry_draft": (
                        retry_draft.model_dump(
                            mode="json"
                        )
                        if retry_draft is not None
                        else None
                    ),
                    "final_citation_state": state,
                },
            )

        citations = [
            self._citation(article_map[number])
            for number in structured_numbers
        ]

        input_tokens = result.usage.input_tokens
        output_tokens = result.usage.output_tokens

        if retry_result is not None:
            input_tokens += retry_result.usage.input_tokens
            output_tokens += retry_result.usage.output_tokens

        total_tokens = input_tokens + output_tokens

        debug: dict[str, Any] | None = None

        if include_debug:
            debug = {
                "model_called": True,
                "model_calls": 2 if retry_applied else 1,
                "provider": self.provider,
                "model": self.model,
                "citation_retry_applied": retry_applied,
                "input_included_exact_user_question": (
                    payload["user_question"] == retrieval.question
                ),
                "input_used_compact_retrieval_evidence": True,
                "model_input_article_numbers": [
                    item["article_number"]
                    for item in payload["retrieval_evidence"]["articles"]
                ],
                "model_call_config": {
                    "provider": self.provider,
                    "model": self.model,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature_sent": False,
                    "top_p_sent": False,
                    "structured_output": True,
                },
                "allowed_article_numbers": sorted(
                    allowed_numbers
                ),
                "cited_article_numbers": structured_numbers,
                "citation_repair_applied": citation_repair_applied,
                "citation_retry_applied": retry_applied,
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
            elapsed_ms=max(
                0,
                round(
                    (time.perf_counter() - started) * 1000
                ),
            ),
            debug=debug,
        )