from __future__ import annotations

import json
import logging
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.legal_question_analysis import (
    LegalQuestionAnalysis,
    normalize_arabic,
)

logger = logging.getLogger(__name__)

_QUERY_STOPWORDS = {
    "انا", "ان", "او", "اي", "اذا", "الى", "التي", "الذي", "على",
    "عن", "في", "ما", "ماذا", "من", "هل", "هو", "هي", "هذا",
    "هذه", "كم", "كيف", "له", "لها", "لم", "لا", "مع", "بعد",
    "قبل", "يمكن", "يجب",
}


class AtomicLegalIssue(BaseModel):
    """One independent issue that should receive its own retrieval query."""

    model_config = ConfigDict(extra="forbid")

    issue_ar: str = Field(min_length=1)
    retrieval_query_ar: str = Field(min_length=1)
    actors: list[str] = Field(default_factory=list, max_length=6)
    conditions: list[str] = Field(default_factory=list, max_length=8)
    numbers: list[str] = Field(default_factory=list, max_length=8)
    requested_result_ar: str = ""


class LegalQueryPlan(BaseModel):
    """Strict output produced before any KG or vector retrieval."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["retrieve", "clarify", "abstain"]
    confidence: float = Field(ge=0.0, le=1.0)
    decision_reason: str = Field(min_length=1)
    normalized_question_ar: str = ""
    legal_domain: str = ""
    atomic_issues: list[AtomicLegalIssue] = Field(
        default_factory=list,
        max_length=3,
    )
    minimum_articles: int = Field(ge=0, le=3)
    maximum_articles: int = Field(ge=0, le=3)
    clarification_question_ar: str = ""

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        queries: list[str] = []
        for issue in self.atomic_issues:
            query = issue.retrieval_query_ar.strip()
            if query and query not in queries:
                queries.append(query)
        return tuple(queries)

    @property
    def issue_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for issue in self.atomic_issues:
            label = issue.issue_ar.strip()
            if label and label not in labels:
                labels.append(label)
        return tuple(labels)


class LegalQueryPlanner:
    """
    Optional pre-retrieval scope, ambiguity, and query-planning layer.

    The planner never selects article numbers and never generates a legal
    answer. Any API or validation failure returns ``None`` so the existing
    deterministic analyzer remains the fallback.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        enabled_value = os.getenv(
            "OPENAI_QUERY_PLANNER_ENABLED",
            "false",
        ).strip().lower()
        self.enabled = enabled_value not in {
            "0",
            "false",
            "no",
            "off",
        }

        configured_model = os.getenv(
            "OPENAI_QUERY_PLANNER_MODEL",
            "",
        ).strip()
        self.model = configured_model or str(
            getattr(settings, "openai_chat_model", "gpt-5-nano")
        ).strip()

        self.reasoning_effort = os.getenv(
            "OPENAI_QUERY_PLANNER_REASONING_EFFORT",
            getattr(settings, "openai_reasoning_effort", "low"),
        ).strip()

        self.route_confidence = self._bounded_float(
            os.getenv(
                "OPENAI_QUERY_PLANNER_ROUTE_CONFIDENCE",
                "0.80",
            ),
            default=0.80,
        )
        self.retrieve_override_confidence = self._bounded_float(
            os.getenv(
                "OPENAI_QUERY_PLANNER_RETRIEVE_OVERRIDE_CONFIDENCE",
                "0.90",
            ),
            default=0.90,
        )

        self.client: OpenAI | None = None

        if self.enabled:
            api_key = str(
                getattr(settings, "openai_api_key", "") or ""
            ).strip()
            if not api_key:
                logger.warning(
                    "OpenAI query planning disabled because "
                    "OPENAI_API_KEY is missing."
                )
                self.enabled = False
            else:
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=float(
                        getattr(settings, "openai_timeout_seconds", 120)
                    ),
                    max_retries=int(
                        getattr(settings, "openai_max_retries", 3)
                    ),
                )

    @staticmethod
    def _bounded_float(value: str, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return min(1.0, max(0.0, parsed))

    @staticmethod
    def _instructions() -> str:
        return """
You are the pre-retrieval query planner for a system whose knowledge base
contains only the Jordanian Labor Law represented in the supplied project.

You must not answer the question and must not select or mention article
numbers. Choose exactly one decision:

- retrieve: the question is within Jordanian labor law and contains enough
  facts to search for the governing provision.
- clarify: the question concerns labor law, but a legally decisive fact is
  missing, so selecting a provision would be unsafe.
- abstain: the question is outside Jordanian labor law.

For retrieve decisions:
1. Rewrite the question in precise Arabic legal language while preserving the
   user's facts, actors, numbers, deadlines, exceptions, and requested result.
2. Decompose it into the smallest set of independent legal issues explicitly
   requested by the user.
3. Produce one focused Arabic retrieval query for each independent issue.
4. Do not split background facts, explanations, or multiple details governed
   by the same legal rule into separate issues.
5. Use two or three issues only when distinct rules, procedures, remedies,
   consequences, actors, or chronological stages are explicitly requested.
6. Estimate the smallest plausible number of governing provisions. The range
   must be between one and three.

For clarify decisions:
- State the missing decisive fact and ask one concise Arabic clarification
  question.
- Return no atomic issues and article counts of zero.

For abstain decisions:
- Explain that the matter is outside Jordanian labor law.
- Return no atomic issues and article counts of zero.

Important distinctions:
- A question may contain labor-related words but still concern consumer law,
  company registration, academic decisions, tenancy, tax, family law,
  criminal law, or another unsupported domain. Such questions require
  abstention.
- A broad labor question is not automatically answerable. Clarify when the
  type of decision, procedural stage, violation, actor, contract, leave,
  deduction, dismissal reason, injury, remedy, or other decisive circumstance
  is missing.
- Handle Modern Standard Arabic, Jordanian colloquial Arabic, spelling errors,
  attached prefixes, and Arabic or Western digits.
- Never rely on article numbers and never provide legal advice.
""".strip()

    @staticmethod
    def _validate_plan(plan: LegalQueryPlan) -> LegalQueryPlan:
        if plan.decision == "retrieve":
            if not plan.atomic_issues:
                raise ValueError(
                    "A retrieve plan must contain at least one atomic issue."
                )
            if plan.minimum_articles < 1:
                raise ValueError(
                    "A retrieve plan must request at least one article."
                )
            if plan.maximum_articles < plan.minimum_articles:
                raise ValueError(
                    "maximum_articles cannot be smaller than minimum_articles."
                )
            if plan.maximum_articles < len(plan.retrieval_queries):
                raise ValueError(
                    "maximum_articles must cover all independent issues."
                )
        else:
            if plan.atomic_issues:
                raise ValueError(
                    "Clarify/abstain plans cannot contain retrieval issues."
                )
            if plan.minimum_articles != 0 or plan.maximum_articles != 0:
                raise ValueError(
                    "Clarify/abstain plans must use zero article counts."
                )
            if (
                plan.decision == "clarify"
                and not plan.clarification_question_ar.strip()
            ):
                raise ValueError(
                    "A clarification plan must include a question."
                )
        return plan

    def plan(self, question: str) -> LegalQueryPlan | None:
        if not self.enabled or self.client is None:
            return None

        try:
            schema = LegalQueryPlan.model_json_schema()
            response = self.client.responses.create(
                model=self.model,
                instructions=self._instructions(),
                input=json.dumps(
                    {"user_question": question},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                reasoning={"effort": self.reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "jordan_labor_legal_query_plan",
                        "schema": schema,
                        "strict": True,
                    }
                },
                store=False,
            )

            if not response.output_text:
                raise RuntimeError("OpenAI returned no query-planning output.")

            plan = LegalQueryPlan.model_validate(
                json.loads(response.output_text)
            )
            return self._validate_plan(plan)
        except Exception as exc:  # deterministic fallback
            logger.warning(
                "OpenAI query planning failed; using deterministic analysis. "
                "Error: %s",
                exc,
            )
            return None

    def merge_with_analysis(
        self,
        *,
        analysis: LegalQuestionAnalysis,
        plan: LegalQueryPlan | None,
    ) -> LegalQuestionAnalysis:
        """Merge a validated plan conservatively into deterministic analysis."""

        if plan is None or plan.confidence < self.route_confidence:
            return analysis

        deterministic_behavior = analysis.behavior
        planned_behavior = plan.decision

        # A deterministic abstention is never weakened by an LLM retrieve
        # decision. This keeps clear non-labor rules as a hard safety floor.
        if deterministic_behavior == "abstain" and planned_behavior == "retrieve":
            final_behavior = "abstain"
            final_reason = analysis.behavior_reason
        elif planned_behavior in {"clarify", "abstain"}:
            # A high-confidence LLM non-answer may add generalization beyond
            # the narrow deterministic keyword rules.
            if deterministic_behavior == "abstain":
                final_behavior = "abstain"
                final_reason = analysis.behavior_reason
            else:
                final_behavior = planned_behavior
                final_reason = plan.decision_reason.strip()
        elif deterministic_behavior == "clarify":
            # Recover a deterministic false clarification only when the plan
            # is especially confident and contains a concrete issue query.
            if (
                plan.confidence >= self.retrieve_override_confidence
                and plan.retrieval_queries
            ):
                final_behavior = "retrieve"
                final_reason = plan.decision_reason.strip()
            else:
                final_behavior = "clarify"
                final_reason = analysis.behavior_reason
        else:
            final_behavior = "retrieve"
            final_reason = plan.decision_reason.strip()

        if final_behavior != "retrieve":
            return LegalQuestionAnalysis(
                original_question=analysis.original_question,
                normalized_question=analysis.normalized_question,
                issue_ids=analysis.issue_ids,
                preferred_concepts=analysis.preferred_concepts,
                query_expansion_terms=analysis.query_expansion_terms,
                article_anchor_phrases=analysis.article_anchor_phrases,
                primary_article_anchor_phrases=(
                    analysis.primary_article_anchor_phrases
                ),
                meaningful_tokens=analysis.meaningful_tokens,
                numeric_tokens=analysis.numeric_tokens,
                max_final_articles=0,
                behavior=final_behavior,
                behavior_reason=final_reason,
                planner_queries=(),
                planner_issue_labels=(),
                planner_used=True,
                planner_confidence=plan.confidence,
                clarification_question=(
                    plan.clarification_question_ar.strip()
                ),
            )

        retrieval_queries = plan.retrieval_queries
        issue_labels = plan.issue_labels
        combined_text = " ".join(
            part
            for part in (
                analysis.original_question,
                plan.normalized_question_ar.strip(),
                *issue_labels,
                *retrieval_queries,
            )
            if part
        )
        combined_normalized = normalize_arabic(combined_text)

        meaningful_tokens = frozenset(
            set(analysis.meaningful_tokens)
            | {
                token
                for token in combined_normalized.split()
                if len(token) > 1 and token not in _QUERY_STOPWORDS
            }
        )
        numeric_tokens = frozenset(
            token
            for token in combined_normalized.split()
            if token.isdigit()
        )

        merged_expansions: list[str] = list(analysis.query_expansion_terms)
        for query in retrieval_queries:
            if query not in merged_expansions:
                merged_expansions.append(query)

        article_limit = max(
            1,
            min(
                3,
                plan.maximum_articles
                or len(retrieval_queries)
                or analysis.max_final_articles,
            ),
        )

        return LegalQuestionAnalysis(
            original_question=analysis.original_question,
            normalized_question=combined_normalized,
            issue_ids=analysis.issue_ids,
            preferred_concepts=analysis.preferred_concepts,
            query_expansion_terms=tuple(merged_expansions),
            article_anchor_phrases=analysis.article_anchor_phrases,
            primary_article_anchor_phrases=(
                analysis.primary_article_anchor_phrases
            ),
            meaningful_tokens=meaningful_tokens,
            numeric_tokens=numeric_tokens,
            max_final_articles=article_limit,
            behavior="retrieve",
            behavior_reason=final_reason,
            planner_queries=retrieval_queries,
            planner_issue_labels=issue_labels,
            planner_used=True,
            planner_confidence=plan.confidence,
            clarification_question="",
        )
