from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm_provider import LLMUsage, build_pipeline_llm
from app.legal_question_analysis import normalize_arabic

logger = logging.getLogger(__name__)


class AtomicLegalIssue(BaseModel):
    """One independent legal issue and its ontology-linking hints."""

    model_config = ConfigDict(extra="forbid")

    issue_ar: str = Field(min_length=1)
    retrieval_query_ar: str = Field(min_length=1)
    concept_hints_ar: list[str] = Field(default_factory=list, max_length=6)
    actors: list[str] = Field(default_factory=list, max_length=6)
    conditions: list[str] = Field(default_factory=list, max_length=8)
    numbers: list[str] = Field(default_factory=list, max_length=8)
    requested_result_ar: str = ""

    @property
    def embedding_text(self) -> str:
        parts = [self.issue_ar.strip(), self.retrieval_query_ar.strip()]
        parts.extend(value.strip() for value in self.concept_hints_ar if value.strip())
        return " ".join(dict.fromkeys(part for part in parts if part)).strip()


class LegalQueryPlan(BaseModel):
    """Strict pre-retrieval plan. It never contains article numbers."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["retrieve", "abstain"]
    confidence: float = Field(ge=0.0, le=1.0)
    decision_reason: str = Field(min_length=1)
    normalized_question_ar: str = ""
    legal_domain: str = ""
    atomic_issues: list[AtomicLegalIssue] = Field(default_factory=list, max_length=5)
    minimum_articles: int = Field(ge=0, le=5)
    maximum_articles: int = Field(ge=0, le=5)

    @property
    def issue_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((issue.issue_ar, issue.retrieval_query_ar) for issue in self.atomic_issues)


class LegalQueryPlanner:
    """LLM router + issue decomposer for the final GraphRAG architecture."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.pipeline_llm_provider
        self.model = settings.pipeline_llm_model
        self.llm = build_pipeline_llm(settings)
        self.last_usage = LLMUsage()
        self.last_verified = False

    @staticmethod
    def _instructions() -> str:
        return r"""
You are the query-planning component for an ontology-driven GraphRAG system over
Jordanian Labor Law.

Return structured JSON exactly matching the supplied schema. Never answer the
legal question and NEVER select, infer, mention, or guess legal article numbers.

Choose exactly one decision:
- retrieve: the substantive answer may be governed by Jordanian Labor Law.
- abstain: the principal answer depends on another legal regime not represented
  by this knowledge graph.

The represented Labor Law covers, among other things: law scope and
commencement, definitions, employment relationships and contracts, wages,
working time and leave, occupational safety and injuries, inspection and
penalties, vocational training and employment organization, non-Jordanian work
permits, worker protections, trade unions/employers' unions, collective labor
disputes, strikes/lockouts, conciliation and labor courts.

Independent Social Security Law, banking, residence/immigration, tax, company,
family/inheritance, tenancy, traffic, consumer, academic and unrelated medical
rules are outside scope unless the question only asks what Jordanian Labor Law
itself says about them.

For retrieve:
1. Produce 1-5 atomic_issues. Split separately requested rules/procedures/facts
   when they can require different evidence.
2. issue_ar: a short Arabic noun phrase naming the legal issue.
3. retrieval_query_ar: normally 3-12 legally material Arabic content words.
   Preserve actors, conditions, dates, durations, amounts and requested effects.
4. concept_hints_ar: 1-6 short Arabic semantic phrases likely to correspond to
   ontology/KG concepts. Use ordinary legal phrases, synonyms or paraphrases;
   do NOT invent ontology identifiers and do NOT mention article numbers.
   Example style: "نفاذ القانون", "النشر في الجريدة الرسمية",
   "تأخر دفع الأجر", "التزام دفع الأجور".
5. actors/conditions/numbers/requested_result_ar preserve material constraints.
6. minimum_articles and maximum_articles must be 1-5. They are planning bounds,
   not predictions of specific statutory article numbers.

For abstain:
- atomic_issues must be empty;
- minimum_articles=maximum_articles=0;
- legal_domain briefly names the unsupported domain.

All Arabic fields must be Arabic. Do not add generic boilerplate to retrieval
queries. Never answer the question and never output an article number.
""".strip()

    @staticmethod
    def _verifier_instructions() -> str:
        return r"""
You verify a proposed abstention for a Jordanian Labor Law GraphRAG system.
Return a complete corrected LegalQueryPlan.

Choose retrieve whenever the requested substantive answer can be supported by
Jordanian Labor Law itself, including its commencement, scope, definitions,
employment, wages, working time, leave, safety, inspections, penalties,
training, work permits, unions, collective disputes, strikes/lockouts,
conciliation or labor-court provisions.

Choose abstain only when the principal answer depends on a separate legal
regime not contained in the represented Labor Law.

If you change the decision to retrieve, create 1-5 focused atomic issues and
short Arabic concept_hints_ar. Never answer the question and never mention or
guess article numbers.
""".strip()

    @staticmethod
    def _validate_plan(plan: LegalQueryPlan) -> LegalQueryPlan:
        if plan.decision == "abstain":
            if plan.atomic_issues:
                raise ValueError("An abstention plan must not contain atomic issues.")
            if plan.minimum_articles != 0 or plan.maximum_articles != 0:
                raise ValueError("An abstention plan must request zero articles.")
            return plan

        if not plan.atomic_issues:
            raise ValueError("A retrieval plan must contain at least one atomic issue.")
        if plan.minimum_articles < 1:
            raise ValueError("A retrieval plan must request at least one article.")
        if plan.maximum_articles < plan.minimum_articles:
            raise ValueError("maximum_articles cannot be smaller than minimum_articles.")

        # Article-number leakage would compromise the retrieval experiment.
        article_marker = "المادة"
        for issue in plan.atomic_issues:
            combined = " ".join([
                issue.issue_ar,
                issue.retrieval_query_ar,
                *issue.concept_hints_ar,
            ])
            if article_marker in normalize_arabic(combined):
                # "المادة" can occur in ordinary phrases (e.g., مادة خطرة), so
                # only reject explicit مادة + digits patterns.
                import re
                if re.search(r"(?:المادة|مادة)\s*[0-9٠-٩۰-۹]+", combined):
                    raise ValueError("The planner must not output article numbers.")

        return plan

    def _request(self, *, question: str, instructions: str, schema_name: str) -> LegalQueryPlan:
        result = self.llm.generate_structured(
            instructions=instructions,
            payload={"question": question},
            response_model=LegalQueryPlan,
            schema_name=schema_name,
            max_output_tokens=self.settings.planner_max_output_tokens,
        )
        self.last_usage = result.usage
        return self._validate_plan(LegalQueryPlan.model_validate(result.data))

    def plan(self, question: str) -> LegalQueryPlan:
        self.last_verified = False
        try:
            plan = self._request(
                question=question,
                instructions=self._instructions(),
                schema_name="jordan_labor_graph_query_plan_v3",
            )

            if (
                plan.decision == "abstain"
                and self.settings.planner_verify_abstain
                and plan.confidence >= self.settings.planner_abstain_verify_confidence
            ):
                verified = self._request(
                    question=question,
                    instructions=self._verifier_instructions(),
                    schema_name="jordan_labor_graph_route_verification_v3",
                )
                plan = verified
                self.last_verified = True

            return plan
        except Exception as exc:
            raise RuntimeError(
                "Query planner failed. "
                f"Provider={self.provider} Model={self.model} "
                f"Error={type(exc).__name__}: {exc}"
            ) from exc
