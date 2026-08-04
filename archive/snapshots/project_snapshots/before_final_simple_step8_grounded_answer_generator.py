from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Literal

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
BRACKETED_ARTICLE_RE = re.compile(
    r"\[+\s*(?:المادة|مادة)\s+([0-9٠-٩]+)\s*\]"
)
UNBRACKETED_ARTICLE_RE = re.compile(
    r"(?<![\w\u0600-\u06ff\[])"
    r"(?:المادة|مادة)\s+([0-9٠-٩]+)"
    r"(?![\w\u0600-\u06ff])"
)


def _setting(settings: Settings, name: str, env_name: str, default: Any) -> Any:
    value = getattr(settings, name, None)
    if value is not None and value != "":
        return value
    return os.getenv(env_name, default)


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _unique_numbers(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = int(value)
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


def _canonicalize_citations(
    text: str,
    *,
    allowed_numbers: set[int],
) -> str:
    """Return canonical, idempotent ``[المادة N]`` citations.

    The function repairs harmless variants and duplicate opening brackets while
    leaving unretrieved article mentions unchanged so later validation can
    reject them. Applying it twice produces the same text as applying it once.
    """

    def bracketed(match: re.Match[str]) -> str:
        raw = match.group(1).translate(ARABIC_DIGITS)
        number = int(raw)
        if number not in allowed_numbers:
            return match.group(0)
        return f"[المادة {number}]"

    def unbracketed(match: re.Match[str]) -> str:
        raw = match.group(1).translate(ARABIC_DIGITS)
        number = int(raw)
        if number not in allowed_numbers:
            return match.group(0)
        return f"[المادة {number}]"

    normalized = BRACKETED_ARTICLE_RE.sub(bracketed, text)
    normalized = UNBRACKETED_ARTICLE_RE.sub(unbracketed, normalized)
    return normalized


class RequestedIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    question_part_ar: str = Field(min_length=1)


class ExcludedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_number: int = Field(ge=1)
    reason_ar: str = Field(min_length=1)


class IssueArticleSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    article_numbers: list[int]


class EvidenceSelectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_issues: list[RequestedIssue]
    selected_article_numbers: list[int]
    excluded_articles: list[ExcludedArticle]
    issue_support: list[IssueArticleSupport]
    missing_issue_ids: list[str]
    answerability: Literal["complete", "partial", "none"]
    missing_information_ar: list[str]


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_ar: str = Field(min_length=1)
    key_points: list[str]
    cited_article_numbers: list[int]
    limitations: list[str]
    covered_issue_ids: list[str]
    issue_support: list[IssueArticleSupport]


class GroundedAnswerGenerator:
    """Generate an Arabic answer from ``RetrievalResultV1`` only.

    The generation boundary performs no retrieval. It first selects a minimal
    evidence subset from the already retrieved articles, then generates and
    validates an answer using only that subset.
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
        self.selection_enabled = _truthy(
            _setting(
                self.settings,
                "openai_answer_evidence_selection_enabled",
                "OPENAI_ANSWER_EVIDENCE_SELECTION_ENABLED",
                "true",
            ),
            default=True,
        )
        self.selection_model = str(
            _setting(
                self.settings,
                "openai_answer_selection_model",
                "OPENAI_ANSWER_SELECTION_MODEL",
                self.model,
            )
        )
        self.selection_reasoning_effort = str(
            _setting(
                self.settings,
                "openai_answer_selection_reasoning_effort",
                "OPENAI_ANSWER_SELECTION_REASONING_EFFORT",
                "low",
            )
        )
        self.max_selected_articles = max(
            1,
            int(
                _setting(
                    self.settings,
                    "openai_answer_max_selected_articles",
                    "OPENAI_ANSWER_MAX_SELECTED_ARTICLES",
                    4,
                )
            ),
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
    def _selection_instructions() -> str:
        return r"""
You are the evidence-selection stage of a grounded Arabic legal-information
system. You receive one user question and only the provisions already
retrieved from the Jordanian Labor Law.

Your job is NOT to answer the user. Your job is to decompose the exact question
and select the smallest sufficient subset of retrieved articles for answering
it safely.

GENERAL SELECTION RULES

- Use only the supplied retrieved articles.
- Split the question into independent requested issues. Create stable issue IDs
  I1, I2, I3, ... in the order the user asked them.
- Select an article only when its supplied text is necessary to answer at least
  one requested issue.
- A retrieved article is not automatically relevant.
- Prefer a provision that directly and specifically governs the described
  actor, subject, and legal situation over a broader related provision.
- Use a general provision only when it supplies a necessary rule absent from
  the specific provision.
- When two articles duplicate the same answer, select the one that completely
  answers the requested issue and exclude the redundant article.
- Do not select an article for an unasked penalty, procedure, exception,
  definition, background rule, or neighboring legal topic.
- Never transfer a rule between actors. Verify that the worker, employer,
  inspector, ministry, union, court, or other actor governed by the article
  matches the actor in the user's question.
- For a multi-part question, map every issue to the exact article or articles
  supporting that issue.
- If a requested issue cannot be answered from the retrieved text, mark that
  issue as missing. Do not substitute a nearby article or infer from memory.
- Account for every retrieved article: it must be either selected or excluded.
- Keep selected_article_numbers in retrieval order.

ANSWERABILITY

- complete: every requested issue has sufficient selected evidence.
- partial: at least one issue has sufficient evidence and at least one does not.
- none: no requested issue can be answered safely from the retrieved text.

OUTPUT CONTENT

- requested_issues: all independent user-requested issues.
- selected_article_numbers: minimum sufficient article subset.
- excluded_articles: every non-selected retrieved article with a concise reason.
- issue_support: one entry for every issue; use an empty article list for a
  missing issue.
- missing_issue_ids: exactly the issues with empty support.
- missing_information_ar: specific missing evidence or facts; no generic
  disclaimer.
""".strip()

    @staticmethod
    def _answer_instructions() -> str:
        return r"""
You are the final grounded Arabic legal-information answer stage.

You receive:
1. the user's exact question;
2. an evidence-selection plan containing the requested issues; and
3. only the selected provisions from the Jordanian Labor Law.

Answer only from those selected provisions.

HARD GROUNDING AND RELEVANCE RULES

- Use only the selected provisions supplied in this call.
- Never use memory, general legal knowledge, web knowledge, or an excluded
  provision.
- Answer every supported requested issue and no unasked issue.
- Follow the issue-to-article mapping in the selection plan.
- Do not add a related penalty, procedure, exception, definition, or background
  merely because it appears in a selected article.
- Preserve every material actor, condition, exception, amount, percentage,
  duration, deadline, and procedural step stated in the supporting text.
- Never transfer a duty, right, restriction, or procedure from one actor to
  another.
- Prefer the exact statutory terminology and copy every number exactly.
- For partial answerability, answer only supported issues and state the specific
  missing issue in limitations. Do not fill the gap from outside knowledge.

ARABIC STYLE

- Use clear, natural Modern Standard Arabic.
- Start directly with the answer.
- Keep a simple answer to one concise paragraph of one to three sentences.
- For a multi-part question, use a compact paragraph or short separate points.
- Remove repetition and any sentence that does not answer a requested issue.
- Do not reproduce an entire article when a concise explanation is sufficient.
- Do not add a source heading, bibliography, or generic legal disclaimer.

CITATIONS

- Every legal claim must have an inline citation exactly as [المادة N].
- N must be a selected article number.
- Place the citation immediately after the sentence or clause it supports.
- cited_article_numbers must contain unique cited articles in first-use order.
- Every selected article assigned in issue_support must be visibly cited.
- Never cite an excluded or unselected article.

VERIFICATION FIELDS

- covered_issue_ids must contain every supported requested issue and no missing
  issue.
- issue_support must contain one entry for every covered issue and identify the
  selected article numbers that support it.
- Before returning JSON, verify that every covered issue is answered, every
  material actor and condition is preserved, and every answer sentence maps to
  a requested issue.

OUTPUT FIELDS

- answer_ar: final user-facing answer only.
- key_points: empty for a simple rule; use only when genuinely useful. Every
  non-empty point must contain an inline citation.
- limitations: empty for complete answerability; specific missing issue only
  for partial answerability.
""".strip()

    @staticmethod
    def _selection_schema() -> dict[str, Any]:
        issue_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issue_id": {"type": "string"},
                "question_part_ar": {"type": "string"},
            },
            "required": ["issue_id", "question_part_ar"],
        }
        support_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issue_id": {"type": "string"},
                "article_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["issue_id", "article_numbers"],
        }
        excluded_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "article_number": {"type": "integer"},
                "reason_ar": {"type": "string"},
            },
            "required": ["article_number", "reason_ar"],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "requested_issues": {
                    "type": "array",
                    "items": issue_schema,
                },
                "selected_article_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "excluded_articles": {
                    "type": "array",
                    "items": excluded_schema,
                },
                "issue_support": {
                    "type": "array",
                    "items": support_schema,
                },
                "missing_issue_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "answerability": {
                    "type": "string",
                    "enum": ["complete", "partial", "none"],
                },
                "missing_information_ar": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "requested_issues",
                "selected_article_numbers",
                "excluded_articles",
                "issue_support",
                "missing_issue_ids",
                "answerability",
                "missing_information_ar",
            ],
        }

    @staticmethod
    def _answer_schema() -> dict[str, Any]:
        support_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issue_id": {"type": "string"},
                "article_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["issue_id", "article_numbers"],
        }
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
                "covered_issue_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "issue_support": {
                    "type": "array",
                    "items": support_schema,
                },
            },
            "required": [
                "answer_ar",
                "key_points",
                "cited_article_numbers",
                "limitations",
                "covered_issue_ids",
                "issue_support",
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
        debug_details: dict[str, Any] | None = None,
        model_called: bool = False,
        usage: GenerationUsageV1 | None = None,
    ) -> GroundedAnswerResultV1:
        warnings = [warning] if warning else []
        debug_payload: dict[str, Any] | None = None
        if include_debug:
            debug_payload = {
                "decision": retrieval.decision.model_dump(mode="json"),
                "model_called": model_called,
            }
            if debug_details:
                debug_payload.update(debug_details)

        return GroundedAnswerResultV1(
            status=status,
            question=retrieval.question,
            answer_ar=answer_ar,
            grounded=False,
            warnings=warnings,
            usage=usage or GenerationUsageV1(),
            elapsed_ms=max(
                0,
                round((time.perf_counter() - started) * 1000),
            ),
            debug=debug_payload,
        )

    def _article_payload(
        self,
        articles: list[RetrievalEvidenceV1],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for rank, article in enumerate(articles, start=1):
            if article.article_number is None:
                continue
            payload.append(
                {
                    "retrieval_rank": rank,
                    "article_number": article.article_number,
                    "labels_ar": article.labels_ar,
                    "text": article.text[: self.max_article_chars],
                    "final_score": article.final_score,
                    "graph_supported": article.graph_supported,
                    "support_paths": article.support_paths[:5],
                }
            )
        return payload

    @staticmethod
    def _citation(article: RetrievalEvidenceV1) -> AnswerCitationV1:
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

    @staticmethod
    def _usage(response: Any) -> GenerationUsageV1:
        usage_obj = getattr(response, "usage", None)
        input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage_obj, "total_tokens", 0)
            or input_tokens + output_tokens
        )
        return GenerationUsageV1(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _sum_usage(*values: GenerationUsageV1) -> GenerationUsageV1:
        return GenerationUsageV1(
            input_tokens=sum(value.input_tokens for value in values),
            output_tokens=sum(value.output_tokens for value in values),
            total_tokens=sum(value.total_tokens for value in values),
        )

    def _fallback_selection_plan(
        self,
        retrieval: RetrievalResultV1,
        article_numbers: list[int],
    ) -> EvidenceSelectionPlan:
        return EvidenceSelectionPlan(
            requested_issues=[
                RequestedIssue(
                    issue_id="I1",
                    question_part_ar=retrieval.question,
                )
            ],
            selected_article_numbers=article_numbers,
            excluded_articles=[],
            issue_support=[
                IssueArticleSupport(
                    issue_id="I1",
                    article_numbers=article_numbers,
                )
            ],
            missing_issue_ids=[],
            answerability="complete",
            missing_information_ar=[],
        )

    def _validate_selection_plan(
        self,
        plan: EvidenceSelectionPlan,
        *,
        allowed_numbers: set[int],
        retrieval_order: list[int],
    ) -> list[str]:
        errors: list[str] = []
        issue_ids = [item.issue_id.strip() for item in plan.requested_issues]
        if not issue_ids or any(not value for value in issue_ids):
            errors.append("Selection plan contains no valid requested issues.")
        if len(issue_ids) != len(set(issue_ids)):
            errors.append("Selection plan contains duplicate issue IDs.")

        selected = _unique_numbers(plan.selected_article_numbers)
        excluded = _unique_numbers(
            [item.article_number for item in plan.excluded_articles]
        )
        selected_set = set(selected)
        excluded_set = set(excluded)

        invalid = sorted((selected_set | excluded_set) - allowed_numbers)
        if invalid:
            errors.append(
                "Selection plan referenced unretrieved articles: "
                + ", ".join(str(value) for value in invalid)
            )
        if selected_set & excluded_set:
            errors.append("An article is both selected and excluded.")
        if selected_set | excluded_set != allowed_numbers:
            missing_accounting = sorted(
                allowed_numbers - (selected_set | excluded_set)
            )
            errors.append(
                "Selection plan did not account for retrieved articles: "
                + ", ".join(str(value) for value in missing_accounting)
            )
        expected_order = [
            number for number in retrieval_order if number in selected_set
        ]
        if selected != expected_order:
            errors.append(
                "Selected articles are not in retrieval order."
            )
        if len(selected) > self.max_selected_articles:
            errors.append(
                "Selection exceeds OPENAI_ANSWER_MAX_SELECTED_ARTICLES."
            )

        support_ids = [item.issue_id for item in plan.issue_support]
        if len(support_ids) != len(set(support_ids)):
            errors.append("Selection plan contains duplicate issue support.")
        if set(support_ids) != set(issue_ids):
            errors.append(
                "Selection issue_support does not cover every requested issue."
            )

        missing_ids = set(plan.missing_issue_ids)
        if not missing_ids.issubset(set(issue_ids)):
            errors.append("Selection has unknown missing issue IDs.")

        support_union: set[int] = set()
        for support in plan.issue_support:
            article_set = set(_unique_numbers(support.article_numbers))
            if not article_set.issubset(selected_set):
                errors.append(
                    f"Issue {support.issue_id} uses an unselected article."
                )
            support_union.update(article_set)
            is_missing = support.issue_id in missing_ids
            if is_missing and article_set:
                errors.append(
                    f"Missing issue {support.issue_id} has article support."
                )
            if not is_missing and not article_set:
                errors.append(
                    f"Supported issue {support.issue_id} has no article."
                )

        if support_union != selected_set:
            errors.append(
                "Every selected article must support at least one issue."
            )

        supported_count = len(issue_ids) - len(missing_ids)
        if plan.answerability == "complete":
            if missing_ids or supported_count != len(issue_ids):
                errors.append(
                    "Complete answerability is inconsistent with missing issues."
                )
            if plan.missing_information_ar:
                errors.append(
                    "Complete selection must not contain missing information."
                )
        elif plan.answerability == "partial":
            if not missing_ids or supported_count < 1 or not selected:
                errors.append(
                    "Partial answerability requires supported and missing issues."
                )
            if not plan.missing_information_ar:
                errors.append(
                    "Partial selection requires specific missing information."
                )
        elif plan.answerability == "none":
            if selected or supported_count != 0 or missing_ids != set(issue_ids):
                errors.append(
                    "None answerability requires all issues to be missing."
                )
            if not plan.missing_information_ar:
                errors.append(
                    "None answerability requires specific missing information."
                )

        return errors

    def _validate_answer_coverage(
        self,
        draft: AnswerDraft,
        *,
        plan: EvidenceSelectionPlan,
        selected_numbers: set[int],
        cited_numbers: list[int],
    ) -> list[str]:
        errors: list[str] = []
        missing_ids = set(plan.missing_issue_ids)
        expected_covered = {
            item.issue_id
            for item in plan.requested_issues
            if item.issue_id not in missing_ids
        }
        covered = set(draft.covered_issue_ids)
        if len(draft.covered_issue_ids) != len(covered):
            errors.append("Answer contains duplicate covered issue IDs.")
        if covered != expected_covered:
            errors.append(
                "Answer covered_issue_ids do not match supported requested issues."
            )

        support_ids = [item.issue_id for item in draft.issue_support]
        if len(support_ids) != len(set(support_ids)):
            errors.append("Answer contains duplicate issue support entries.")
        if set(support_ids) != expected_covered:
            errors.append(
                "Answer issue_support does not cover every answered issue."
            )

        support_union: set[int] = set()
        selection_support = {
            item.issue_id: set(item.article_numbers)
            for item in plan.issue_support
        }
        for support in draft.issue_support:
            numbers = set(_unique_numbers(support.article_numbers))
            if not numbers:
                errors.append(
                    f"Answered issue {support.issue_id} has no article support."
                )
            if not numbers.issubset(selected_numbers):
                errors.append(
                    f"Answered issue {support.issue_id} uses an unselected article."
                )
            if not numbers.issubset(
                selection_support.get(support.issue_id, set())
            ):
                errors.append(
                    f"Answer changed the selected support for {support.issue_id}."
                )
            support_union.update(numbers)

        if support_union != set(cited_numbers):
            errors.append(
                "Cited articles must exactly match issue-support articles."
            )

        if plan.answerability == "complete" and draft.limitations:
            errors.append("Complete answer must not contain limitations.")
        if plan.answerability == "partial" and not draft.limitations:
            errors.append("Partial answer must state the specific limitation.")

        return errors

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
            return self._non_retrieval_result(
                retrieval,
                status="out_of_scope",
                answer_ar=(
                    reason
                    or "السؤال خارج نطاق قانون العمل الأردني المتاح في هذا النظام."
                ),
                started=started,
                include_debug=include_debug,
            )

        article_map = {
            int(article.article_number): article
            for article in retrieval.articles
            if article.article_number is not None
        }
        retrieval_order = list(article_map)
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
                warning=(
                    "OPENAI_ANSWER_ENABLED is false or no client is available."
                ),
                include_debug=include_debug,
            )

        selection_usage = GenerationUsageV1()
        selection_response: Any | None = None

        if self.selection_enabled:
            selection_payload = {
                "user_question": retrieval.question,
                "retrieved_articles": self._article_payload(
                    retrieval.articles
                ),
            }
            selection_response = self.client.responses.create(
                model=self.selection_model,
                instructions=self._selection_instructions(),
                input=json.dumps(
                    selection_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                reasoning={"effort": self.selection_reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_evidence_selection_plan",
                        "schema": self._selection_schema(),
                        "strict": True,
                    }
                },
                store=False,
            )
            selection_usage = self._usage(selection_response)
            selection_text = str(
                getattr(selection_response, "output_text", "") or ""
            )
            if not selection_text:
                return self._non_retrieval_result(
                    retrieval,
                    status="insufficient_evidence",
                    answer_ar=(
                        "تعذر تحديد المواد اللازمة للإجابة من النتائج "
                        "المسترجعة."
                    ),
                    started=started,
                    warning="Evidence selector returned no output.",
                    include_debug=include_debug,
                    model_called=True,
                    usage=selection_usage,
                    debug_details={"stage": "evidence_selection"},
                )
            try:
                plan = EvidenceSelectionPlan.model_validate(
                    json.loads(selection_text)
                )
            except Exception as exc:
                return self._non_retrieval_result(
                    retrieval,
                    status="insufficient_evidence",
                    answer_ar=(
                        "تعذر اعتماد خطة اختيار المواد اللازمة للإجابة."
                    ),
                    started=started,
                    warning=f"Invalid evidence selection output: {exc}",
                    include_debug=include_debug,
                    model_called=True,
                    usage=selection_usage,
                    debug_details={
                        "stage": "evidence_selection",
                        "raw_selection_output": selection_text,
                    },
                )
        else:
            plan = self._fallback_selection_plan(
                retrieval,
                retrieval_order,
            )

        selection_errors = self._validate_selection_plan(
            plan,
            allowed_numbers=allowed_numbers,
            retrieval_order=retrieval_order,
        )
        if selection_errors:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar=(
                    "تعذر اعتماد اختيار المواد اللازمة للإجابة بصورة آمنة."
                ),
                started=started,
                warning=" ".join(selection_errors),
                include_debug=include_debug,
                model_called=self.selection_enabled,
                usage=selection_usage,
                debug_details={
                    "stage": "evidence_selection_validation",
                    "selection_plan": plan.model_dump(mode="json"),
                    "selection_errors": selection_errors,
                },
            )

        if plan.answerability == "none":
            missing = " ".join(plan.missing_information_ar).strip()
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar=(
                    missing
                    or "لا تتضمن المواد المسترجعة حكماً يجيب عن السؤال."
                ),
                started=started,
                warning="Selected evidence cannot answer any requested issue.",
                include_debug=include_debug,
                model_called=self.selection_enabled,
                usage=selection_usage,
                debug_details={
                    "stage": "evidence_selection",
                    "selection_plan": plan.model_dump(mode="json"),
                },
            )

        selected_numbers = _unique_numbers(plan.selected_article_numbers)
        selected_set = set(selected_numbers)
        selected_articles = [
            article_map[number] for number in selected_numbers
        ]

        answer_payload = {
            "user_question": retrieval.question,
            "selection_plan": plan.model_dump(mode="json"),
            "selected_articles": self._article_payload(selected_articles),
        }
        answer_response = self.client.responses.create(
            model=self.model,
            instructions=self._answer_instructions(),
            input=json.dumps(
                answer_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            reasoning={"effort": self.reasoning_effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "verified_grounded_jordan_labor_answer",
                    "schema": self._answer_schema(),
                    "strict": True,
                }
            },
            store=False,
        )
        answer_usage = self._usage(answer_response)
        total_usage = self._sum_usage(selection_usage, answer_usage)
        output_text = str(
            getattr(answer_response, "output_text", "") or ""
        )
        if not output_text:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر إنشاء إجابة موثقة من المواد المختارة.",
                started=started,
                warning="OpenAI returned no final answer output.",
                include_debug=include_debug,
                model_called=True,
                usage=total_usage,
                debug_details={
                    "stage": "answer_generation",
                    "selection_plan": plan.model_dump(mode="json"),
                },
            )

        try:
            draft = AnswerDraft.model_validate(json.loads(output_text))
        except Exception as exc:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar="تعذر اعتماد بنية الإجابة المولدة.",
                started=started,
                warning=f"Invalid final answer output: {exc}",
                include_debug=include_debug,
                model_called=True,
                usage=total_usage,
                debug_details={
                    "stage": "answer_generation",
                    "selection_plan": plan.model_dump(mode="json"),
                    "raw_answer_output": output_text,
                },
            )

        normalized_answer = _canonicalize_citations(
            draft.answer_ar,
            allowed_numbers=selected_set,
        )
        normalized_key_points = [
            _canonicalize_citations(
                point,
                allowed_numbers=selected_set,
            )
            for point in draft.key_points
        ]

        cited_numbers = _unique_numbers(draft.cited_article_numbers)
        answer_inline_numbers = [
            int(value.translate(ARABIC_DIGITS))
            for value in INLINE_CITATION_RE.findall(normalized_answer)
        ]
        key_point_inline_numbers: list[int] = []
        uncited_key_points: list[int] = []
        for index, key_point in enumerate(normalized_key_points, start=1):
            point_numbers = [
                int(value.translate(ARABIC_DIGITS))
                for value in INLINE_CITATION_RE.findall(key_point)
            ]
            if key_point.strip() and not point_numbers:
                uncited_key_points.append(index)
            key_point_inline_numbers.extend(point_numbers)

        inline_numbers = _unique_numbers(
            answer_inline_numbers + key_point_inline_numbers
        )
        citation_repair_applied = False

        if (
            cited_numbers
            and not inline_numbers
            and len(cited_numbers) == 1
            and cited_numbers[0] in selected_set
            and normalized_answer.strip()
            and not normalized_key_points
        ):
            repaired_number = cited_numbers[0]
            normalized_answer = (
                normalized_answer.rstrip()
                + f" [المادة {repaired_number}]"
            )
            inline_numbers = [repaired_number]
            citation_repair_applied = True

        if (
            cited_numbers
            and inline_numbers
            and set(cited_numbers) == set(inline_numbers)
        ):
            cited_numbers = inline_numbers
        if not cited_numbers and inline_numbers:
            cited_numbers = inline_numbers

        invalid_numbers = sorted(
            (set(cited_numbers) | set(inline_numbers)) - selected_set
        )
        citation_mismatch = set(cited_numbers) != set(inline_numbers)
        coverage_errors = self._validate_answer_coverage(
            draft,
            plan=plan,
            selected_numbers=selected_set,
            cited_numbers=cited_numbers,
        )

        validation_errors: list[str] = []
        if invalid_numbers:
            validation_errors.append(
                "Answer cited unselected articles: "
                + ", ".join(str(value) for value in invalid_numbers)
            )
        if citation_mismatch:
            validation_errors.append(
                "Inline citations do not match cited_article_numbers."
            )
        if not cited_numbers:
            validation_errors.append(
                "Generated answer contains no valid citation."
            )
        if uncited_key_points:
            validation_errors.append(
                "Uncited key_points indexes: "
                + ", ".join(str(value) for value in uncited_key_points)
            )
        validation_errors.extend(coverage_errors)

        if validation_errors:
            return self._non_retrieval_result(
                retrieval,
                status="insufficient_evidence",
                answer_ar=(
                    "تعذر اعتماد الإجابة لأن توثيقها أو تغطيتها للسؤال "
                    "لم يطابق المواد المختارة."
                ),
                started=started,
                warning=" ".join(validation_errors),
                include_debug=include_debug,
                model_called=True,
                usage=total_usage,
                debug_details={
                    "stage": "answer_validation",
                    "allowed_article_numbers": sorted(allowed_numbers),
                    "selected_article_numbers": selected_numbers,
                    "excluded_article_numbers": [
                        item.article_number
                        for item in plan.excluded_articles
                    ],
                    "selection_plan": plan.model_dump(mode="json"),
                    "rejected_answer_draft": draft.model_dump(mode="json"),
                    "normalized_answer_ar": normalized_answer,
                    "normalized_key_points": normalized_key_points,
                    "inline_article_numbers": inline_numbers,
                    "structured_article_numbers": (
                        draft.cited_article_numbers
                    ),
                    "citation_repair_applied": citation_repair_applied,
                    "validation_errors": validation_errors,
                    "usage_breakdown": {
                        "selection": selection_usage.model_dump(mode="json"),
                        "answer": answer_usage.model_dump(mode="json"),
                    },
                },
            )

        citations = [
            self._citation(article_map[number])
            for number in cited_numbers
        ]

        debug: dict[str, Any] | None = None
        if include_debug:
            debug = {
                "model_called": True,
                "model_calls": 2 if self.selection_enabled else 1,
                "selection_model": (
                    self.selection_model if self.selection_enabled else ""
                ),
                "answer_model": self.model,
                "allowed_article_numbers": sorted(allowed_numbers),
                "selected_article_numbers": selected_numbers,
                "excluded_article_numbers": [
                    item.article_number for item in plan.excluded_articles
                ],
                "selection_plan": plan.model_dump(mode="json"),
                "answer_draft": draft.model_dump(mode="json"),
                "inline_article_numbers": inline_numbers,
                "normalized_answer_ar": normalized_answer,
                "normalized_key_points": normalized_key_points,
                "citation_repair_applied": citation_repair_applied,
                "usage_breakdown": {
                    "selection": selection_usage.model_dump(mode="json"),
                    "answer": answer_usage.model_dump(mode="json"),
                },
            }

        return GroundedAnswerResultV1(
            status="generated",
            question=retrieval.question,
            answer_ar=normalized_answer.strip(),
            key_points=[
                value.strip()
                for value in normalized_key_points
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
            usage=total_usage,
            elapsed_ms=max(
                0,
                round((time.perf_counter() - started) * 1000),
            ),
            debug=debug,
        )
