from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm_provider import build_pipeline_llm
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

    decision: Literal["retrieve", "abstain"]
    confidence: float = Field(ge=0.0, le=1.0)
    decision_reason: str = Field(min_length=1)
    normalized_question_ar: str = ""
    legal_domain: str = ""
    atomic_issues: list[AtomicLegalIssue] = Field(
        default_factory=list,
        max_length=5,
    )
    minimum_articles: int = Field(ge=0, le=5)
    maximum_articles: int = Field(ge=0, le=5)
    clarification_question_ar: str = ""

    @property
    def retrieval_issue_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return de-duplicated (issue label, retrieval query) pairs.

        Keeping the label and query together prevents independent de-duplication
        from misaligning planner issues in downstream issue-wise retrieval.
        """

        pairs: list[tuple[str, str]] = []
        seen_queries: set[str] = set()

        for issue in self.atomic_issues:
            label = issue.issue_ar.strip()
            query = issue.retrieval_query_ar.strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            pairs.append((label or query, query))

        return tuple(pairs)

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        return tuple(
            query
            for _, query in self.retrieval_issue_pairs
        )

    @property
    def issue_labels(self) -> tuple[str, ...]:
        return tuple(
            label
            for label, _ in self.retrieval_issue_pairs
        )


class LegalQueryPlanner:
    """
    Optional pre-retrieval scope, ambiguity, and query-planning layer.

    The planner never selects article numbers and never generates a legal
    answer. Any API or validation failure returns ``None`` so the existing
    deterministic analyzer remains the fallback.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        strict_value = os.getenv(
            "PIPELINE_STRICT_EVALUATION",
            "false",
        ).strip().lower()
        self.strict_evaluation = strict_value not in {
            "0", "false", "no", "off", ""
        }

        enabled_value = os.getenv(
            "PIPELINE_QUERY_PLANNER_ENABLED",
            os.getenv(
                "OPENAI_QUERY_PLANNER_ENABLED",
                "false",
            ),
        ).strip().lower()
        self.enabled = enabled_value not in {
            "0",
            "false",
            "no",
            "off",
        }

        # For fair end-to-end comparison, the planner must use the same
        # shared provider/model as the other LLM-dependent pipeline stages.
        self.provider = str(
            getattr(settings, "pipeline_llm_provider", "openai")
        ).strip().lower()
        self.model = str(
            getattr(settings, "pipeline_llm_model", "gpt-5-nano")
        ).strip()

        self.route_confidence = self._bounded_float(
            os.getenv(
                "PIPELINE_QUERY_PLANNER_ROUTE_CONFIDENCE",
                os.getenv(
                    "OPENAI_QUERY_PLANNER_ROUTE_CONFIDENCE",
                    "0.80",
                ),
            ),
            default=0.80,
        )
        self.retrieve_override_confidence = self._bounded_float(
            os.getenv(
                "PIPELINE_QUERY_PLANNER_RETRIEVE_OVERRIDE_CONFIDENCE",
                os.getenv(
                    "OPENAI_QUERY_PLANNER_RETRIEVE_OVERRIDE_CONFIDENCE",
                    "0.90",
                ),
            ),
            default=0.90,
        )

        verify_value = os.getenv(
            "PIPELINE_QUERY_PLANNER_VERIFY_NON_ANSWER",
            os.getenv(
                "OPENAI_QUERY_PLANNER_VERIFY_NON_ANSWER",
                "false",
            ),
        ).strip().lower()
        self.verify_non_answer = verify_value not in {
            "0",
            "false",
            "no",
            "off",
        }

        self.non_answer_verification_confidence = self._bounded_float(
            os.getenv(
                "PIPELINE_QUERY_PLANNER_NON_ANSWER_VERIFY_CONFIDENCE",
                os.getenv(
                    "OPENAI_QUERY_PLANNER_NON_ANSWER_VERIFY_CONFIDENCE",
                    "0.80",
                ),
            ),
            default=0.80,
        )

        verify_retrieve_value = os.getenv(
            "PIPELINE_QUERY_PLANNER_VERIFY_LOW_CONFIDENCE_RETRIEVE",
            os.getenv(
                "OPENAI_QUERY_PLANNER_VERIFY_LOW_CONFIDENCE_RETRIEVE",
                "false",
            ),
        ).strip().lower()
        self.verify_low_confidence_retrieve = (
            verify_retrieve_value not in {
                "0",
                "false",
                "no",
                "off",
            }
        )

        self.retrieve_verification_below = self._bounded_float(
            os.getenv(
                "PIPELINE_QUERY_PLANNER_RETRIEVE_VERIFY_BELOW",
                os.getenv(
                    "OPENAI_QUERY_PLANNER_RETRIEVE_VERIFY_BELOW",
                    "0.90",
                ),
            ),
            default=0.90,
        )

        self.llm = None

        # Diagnostic state only. This does not affect routing or retrieval.
        # It exposes the latest planner/provider failure instead of allowing
        # a silent deterministic fallback with no inspectable reason.
        self.last_error: str | None = None
        self.last_error_stage: str | None = None

        if self.enabled:
            try:
                self.llm = build_pipeline_llm(settings)
            except Exception as exc:
                logger.warning(
                    "Pipeline query planning disabled because the configured "
                    "LLM provider could not be initialized. Provider=%s "
                    "Model=%s Error=%s",
                    self.provider,
                    self.model,
                    exc,
                )
                self.enabled = False

    def diagnostic_state(self) -> dict[str, object]:
        """Return read-only planner diagnostics without changing routing behavior."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "llm_ready": self.llm is not None,
            "strict_evaluation": self.strict_evaluation,
            "last_error_stage": self.last_error_stage,
            "last_error": self.last_error,
        }

    @staticmethod
    def _bounded_float(value: str, *, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return min(1.0, max(0.0, parsed))

    @staticmethod
    def _instructions() -> str:
        return r"""
You are the routing and query-planning component for a Jordanian Labor Law
knowledge-graph retrieval system.

Return structured JSON that follows the supplied schema. Never answer the legal
question, provide legal advice, or select, infer, mention, or guess article
numbers.

Choose exactly one decision:

- retrieve
- abstain

ROUTING POLICY

Choose retrieve whenever the substantive legal issue may be governed by the
Jordanian Labor Law represented in this system.

This includes direct, general, ambiguous, incomplete, colloquial, misspelled,
paraphrased, numerical, and multi-issue labor-law questions.

KNOWLEDGE-BASE SCOPE

The knowledge base contains the complete Jordanian Labor Law represented as a
knowledge graph. It covers rules expressly contained in that law, including:

- the law's definitions, application, and commencement;
- employment relationships and employment contracts;
- wages and deductions made from wages;
- working hours, rest periods, and leave;
- occupational safety, occupational injuries, and compensation under Labor Law;
- labor inspection, violations, penalties, and closure procedures;
- vocational training and employment organization;
- employment of non-Jordanian workers and work permits;
- collective labor disputes, conciliation boards, and labor courts;
- trade unions and employers' unions, including their establishment,
  registration, legal personality, administration, funds, federations,
  liabilities, offences, and dissolution;
- authorities, procedures, remedies, and transitional provisions expressly
  stated in the Jordanian Labor Law.

The knowledge base does not contain the complete text or independent rules of
other Jordanian legal regimes, such as:

- Social Security Law and its retirement, pension, disability, unemployment,
  benefit-calculation, and appeal rules;
- banking and financial law, including loans, interest, bank accounts, and
  Central Bank complaint procedures;
- residence and immigration law, including residence permits, visas, family
  reunification, and Ministry of Interior procedures;
- tax, company, family, inheritance, tenancy, traffic, consumer-protection,
  academic, and unrelated medical rules.

Choose retrieve when the substantive answer can be supported by rules expressly
contained in the represented Jordanian Labor Law.

Choose abstain when the principal requested answer depends on independent rules
from a legal regime that is not contained in this knowledge base.

Do not classify a question only from isolated words such as worker, salary,
contract, compensation, bank, pension, residence, court, or union. Judge the
substantive legal issue requested by the user.

When uncertain, prefer retrieve unless the principal issue is clearly outside
the represented Labor Law.

The system does not ask clarification questions. When an in-scope question
lacks facts, retrieve the most useful generally applicable provisions that can
be identified from the available wording.

FOR ABSTAIN

- atomic_issues must be an empty list;
- minimum_articles and maximum_articles must both be 0;
- normalized_question_ar and clarification_question_ar must be empty strings;
- legal_domain must briefly identify the unsupported domain;
- decision_reason must be one concise Arabic sentence;
- all user-facing Arabic fields must be fully Arabic.

FOR RETRIEVE

- identify every independent legal issue, using one to five atomic issues;
- produce one focused Arabic retrieval query for each issue;
- preserve material actors, conditions, numbers, dates, deadlines, amounts,
  procedures, and requested consequences;
- do not select article numbers;
- minimum_articles must be between 1 and 5;
- maximum_articles must be between minimum_articles and 5;
- request multiple articles only when separate requested issues are likely
  governed by separate provisions;
- clarification_question_ar must be an empty string;
- decision_reason and all Arabic fields must be fully Arabic.

Return the smallest complete retrieval plan that can answer all requested legal
issues.
""".strip()

    @staticmethod
    def _route_verifier_instructions() -> str:
        return r"""
You verify routing for a Jordanian Labor Law knowledge-graph system.

Return a complete corrected plan using exactly one decision:

- retrieve
- abstain

The knowledge base contains the complete Jordanian Labor Law, including
employment contracts, wages, working time and leave, occupational safety and
injuries, inspection and penalties, non-Jordanian work permits, labor disputes,
labor courts, conciliation, and trade-union and employers'-union rules.

It does not contain the complete independent rules of Social Security Law,
banking law, residence and immigration law, tax law, company law, family and
inheritance law, tenancy law, traffic law, consumer-protection law, academic
regulations, or unrelated medical rules.

Choose retrieve when the requested answer can be supported by a rule expressly
contained in the represented Jordanian Labor Law.

Choose abstain when the principal requested answer depends on an independent
rule from another legal regime not contained in the knowledge base.

Do not classify from isolated words such as worker, salary, contract,
compensation, bank, pension, residence, court, or union. Judge the substantive
legal issue requested by the user.

When uncertain, prefer retrieve unless the principal issue is clearly outside
the represented Labor Law.

For retrieve, provide one to five focused Arabic atomic issues and request one
to five articles. For abstain, provide no issues and request zero articles.

Never answer the question, never mention or guess article numbers, and keep
clarification_question_ar empty.
""".strip()

    @staticmethod
    def _non_answer_verifier_instructions() -> str:
        """Backward-compatible alias used by older tests and tooling."""
        return LegalQueryPlanner._route_verifier_instructions()

    @staticmethod
    def _validate_plan(plan: LegalQueryPlan) -> LegalQueryPlan:
        if plan.clarification_question_ar.strip():
            raise ValueError(
                "clarification_question_ar must always be empty in the two-route planner."
            )

        if plan.decision == "abstain":
            if plan.atomic_issues:
                raise ValueError(
                    "An abstention plan must not contain atomic issues."
                )
            if plan.minimum_articles != 0 or plan.maximum_articles != 0:
                raise ValueError(
                    "An abstention plan must request zero articles."
                )
            return plan

        if not plan.atomic_issues:
            raise ValueError(
                "A retrieval plan must contain at least one atomic issue."
            )
        if plan.minimum_articles < 1:
            raise ValueError(
                "A retrieval plan must request at least one article."
            )
        if plan.maximum_articles < plan.minimum_articles:
            raise ValueError(
                "maximum_articles cannot be smaller than minimum_articles."
            )
        if plan.maximum_articles > 5:
            raise ValueError(
                "A retrieval plan cannot request more than five articles."
            )
        return plan

    def _request_plan(
        self,
        *,
        payload: dict,
        instructions: str,
        schema_name: str,
    ) -> LegalQueryPlan:
        if self.llm is None:
            raise RuntimeError(
                "Pipeline query-planner LLM is unavailable."
            )

        result = self.llm.generate_structured(
            instructions=instructions,
            payload=payload,
            response_model=LegalQueryPlan,
            schema_name=schema_name,
            max_output_tokens=int(
                getattr(
                    self.settings,
                    "planner_max_output_tokens",
                    3000,
                )
            ),
        )

        plan = LegalQueryPlan.model_validate(result.data)
        return self._validate_plan(plan)

    def plan(self, question: str) -> LegalQueryPlan | None:
        # Reset diagnostics for each new request.
        self.last_error = None
        self.last_error_stage = None

        if not self.enabled or self.llm is None:
            if not self.enabled:
                self.last_error_stage = "initialization"
                self.last_error = "Query planner is disabled."
            elif self.llm is None:
                self.last_error_stage = "initialization"
                self.last_error = "Pipeline LLM provider is not initialized."

            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: query planner is unavailable. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={self.last_error}"
                )
            return None

        try:
            first_plan = self._request_plan(
                payload={"user_question": question},
                instructions=self._instructions(),
                schema_name="jordan_labor_legal_query_plan",
            )
        except Exception as exc:  # deterministic fallback outside strict evaluation
            self.last_error_stage = "query_planning"
            self.last_error = f"{type(exc).__name__}: {exc}"

            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: query planner failed. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={self.last_error}"
                ) from exc

            logger.warning(
                "Pipeline query planning failed; using deterministic analysis. "
                "Provider=%s Model=%s Error=%s",
                self.provider,
                self.model,
                exc,
            )
            return None

        verification_reason = ""
        if first_plan.decision == "abstain":
            if self.verify_non_answer:
                verification_reason = "proposed_non_answer"
        elif (
            first_plan.decision == "retrieve"
            and self.verify_low_confidence_retrieve
            and first_plan.confidence < self.retrieve_verification_below
        ):
            verification_reason = "low_confidence_retrieve"

        if not verification_reason:
            return first_plan

        try:
            verified_plan = self._request_plan(
                payload={
                    "user_question": question,
                    "proposed_plan": first_plan.model_dump(),
                    "verification_reason": verification_reason,
                },
                instructions=self._route_verifier_instructions(),
                schema_name="jordan_labor_route_verification",
            )

            if (
                verified_plan.confidence
                >= self.non_answer_verification_confidence
            ):
                if verified_plan.decision != first_plan.decision:
                    logger.info(
                        "Query-planner route verifier changed route "
                        "from %s to %s (%s).",
                        first_plan.decision,
                        verified_plan.decision,
                        verification_reason,
                    )
                return verified_plan

            logger.info(
                "Query-planner route verifier confidence %.3f is below "
                "threshold %.3f; keeping the first plan.",
                verified_plan.confidence,
                self.non_answer_verification_confidence,
            )
            return first_plan
        except Exception as exc:
            self.last_error_stage = "route_verification"
            self.last_error = f"{type(exc).__name__}: {exc}"

            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: route verification failed. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={self.last_error}"
                ) from exc

            logger.warning(
                "Pipeline route verification failed; keeping the first "
                "validated plan. Provider=%s Model=%s Error=%s",
                self.provider,
                self.model,
                exc,
            )
            return first_plan

    def merge_with_analysis(
        self,
        *,
        analysis: LegalQuestionAnalysis,
        plan: LegalQueryPlan | None,
    ) -> LegalQuestionAnalysis:
        """Merge a validated plan conservatively into deterministic analysis."""

        if plan is None:
            return analysis

        deterministic_behavior = analysis.behavior
        planned_behavior = plan.decision

        # The final router has exactly two states. A deterministic abstention is
        # retained as a safety floor; otherwise the validated planner decides.
        if deterministic_behavior == "abstain":
            final_behavior = "abstain"
            final_reason = analysis.behavior_reason
        elif planned_behavior == "abstain":
            final_behavior = "abstain"
            final_reason = plan.decision_reason.strip()
        else:
            final_behavior = "retrieve"
            final_reason = plan.decision_reason.strip()

        if final_behavior == "abstain":
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
                behavior="abstain",
                behavior_reason=final_reason,
                planner_queries=(),
                planner_issue_labels=(),
                planner_used=True,
                planner_confidence=plan.confidence,
                clarification_question="",
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
                5,
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
