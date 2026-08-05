from __future__ import annotations

import copy
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

    @classmethod
    def _strict_response_schema(cls) -> dict:
        """Return an OpenAI-compatible strict JSON Schema.

        Pydantic omits fields with defaults from ``required``. OpenAI strict
        structured outputs instead require every property of every object to
        appear in that object's ``required`` array. Optional semantic values
        are therefore represented by empty strings or empty lists, not by
        omitting the property.
        """

        schema = copy.deepcopy(LegalQueryPlan.model_json_schema())

        def normalize(node: object) -> None:
            if isinstance(node, dict):
                # Pydantic defaults are application-side conveniences, not
                # part of the strict response contract sent to OpenAI.
                node.pop("default", None)

                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties.keys())
                    node["additionalProperties"] = False

                for value in node.values():
                    normalize(value)
            elif isinstance(node, list):
                for value in node:
                    normalize(value)

        normalize(schema)
        return schema

    @staticmethod
    def _instructions() -> str:
        return r"""
You are a pre-retrieval routing and query-planning component for a legal
information system.

The system's knowledge base contains only the Jordanian Labor Law represented
in the supplied project.

You must not answer the user's legal question.
You must not provide legal advice.
You must not select, infer, mention, or guess article numbers.

Your task is limited to:

1. deciding whether retrieval should occur;
2. identifying whether clarification is required;
3. identifying whether the question is outside the represented legal scope;
4. generating concise Arabic retrieval queries when retrieval is appropriate.

Choose exactly one decision:

- retrieve
- clarify
- abstain

Apply the following decision process in the exact order shown.

STEP 1 — SCOPE

First determine whether the substance of the question is governed by the
Jordanian Labor Law represented in this system.

Choose abstain when the question principally concerns an unsupported legal or
non-legal domain, including consumer protection, sale of goods, company
registration, academic decisions, tenancy, taxation, family law, criminal law,
or another domain outside Jordanian Labor Law.

The presence of words such as worker, institution, contract, dispute, decision,
or compensation does not by itself make a question a labor-law question.

For abstain:

- atomic_issues must be an empty list;
- minimum_articles must be 0;
- maximum_articles must be 0;
- clarification_question_ar must be an empty string;
- normalized_question_ar must be an empty string;
- briefly identify the actual unsupported domain in legal_domain;
- explain the scope reason in one concise sentence.

STEP 2 — FACTUAL SUFFICIENCY

If the question is within Jordanian Labor Law, determine whether it includes
enough legally decisive facts to identify a governing rule safely.

Choose clarify when an omitted fact could change the applicable rule,
procedure, remedy, consequence, responsible actor, or governing provision.

Missing background details that do not affect the legal rule do not require
clarification.

Clarification is mandatory in the following situations:

- the user asks whether an unspecified decision is lawful without identifying
  the type of decision or its basis;
- the user asks for the next step in a multi-stage legal procedure without
  stating the current procedural stage;
- the user mentions a violation without identifying its type or responsible
  actor when different rules may apply;
- the user asks about dismissal, leave, deduction, injury, compensation,
  disciplinary action, union action, or another legal consequence while
  omitting a fact that determines which rule applies.

For clarify:

- atomic_issues must be an empty list;
- minimum_articles must be 0;
- maximum_articles must be 0;
- ask exactly one concise Arabic clarification question;
- the clarification question must request the missing decisive fact;
- do not generate retrieval queries;
- do not attempt to retrieve all possible procedural stages or legal outcomes.

STEP 3 — RETRIEVAL PLANNING

Choose retrieve only when:

- the question is within Jordanian Labor Law; and
- the question contains enough decisive facts to search for the governing
  provision without guessing.

For retrieve decisions:

1. Produce a concise normalized Arabic version of the question.
2. Preserve every explicit actor, number, percentage, duration, deadline,
   condition, exception, and requested result.
3. Do not add any fact, actor, institution, remedy, condition, procedure,
   document, compensation, exception, or possible outcome that the user did
   not state.
4. Decompose the question into the smallest possible set of independent legal
   issues explicitly requested by the user.
5. Do not create separate issues for background facts or for several details
   governed by the same legal rule.
6. Create separate issues only when the user explicitly asks about genuinely
   distinct rules, stages, duties, rights, procedures, remedies, or
   consequences.
7. Produce exactly one retrieval_query_ar for each atomic issue.
8. Each retrieval query must contain approximately 5 to 20 Arabic words.
9. Each retrieval query must contain legal search terms, not an explanation or
   a complete answer.
10. Do not include generic phrases such as:
    "وفق قانون العمل الأردني",
    "حدد الأحكام القانونية",
    "ما هي النصوص التنظيمية",
    or "الجهات المختصة".
11. Do not repeat the entire user question inside every retrieval query.
12. Do not generate alternative hypothetical interpretations.

ATOMIC ISSUE FIELDS

For each atomic issue:

- issue_ar must be a short description of one requested legal issue;
- retrieval_query_ar must be a concise search phrase;
- actors must contain only actors explicitly stated by the user;
- conditions must contain only conditions explicitly stated by the user;
- numbers must contain only numbers, percentages, durations, or deadlines
  explicitly stated by the user;
- requested_result_ar must state only what the user explicitly wants to know.

Use empty lists or an empty string when the corresponding information was not
explicitly stated.

Never infer additional actors or conditions merely because they may commonly
appear in the relevant legal procedure.

ARTICLE-COUNT ESTIMATION

Estimate the smallest number of governing provisions required.

- One atomic issue normally means:
  minimum_articles = 1
  maximum_articles = 1

- Two independent atomic issues normally mean:
  minimum_articles = 2
  maximum_articles = 2

- Three independent atomic issues normally mean:
  minimum_articles = 3
  maximum_articles = 3

Do not increase maximum_articles merely because neighboring, introductory,
definitional, or generally related provisions may exist.

The number of atomic issues and the article-count range must never exceed 3.

CONFIDENCE

Confidence represents confidence in the routing decision, not confidence in
the final legal answer.

Use:

- 0.90 to 1.00 when the scope and routing decision are explicit and
  unambiguous;
- 0.80 to 0.89 when the routing decision is strongly supported;
- 0.60 to 0.79 when some routing uncertainty remains;
- below 0.60 only when the routing decision itself is materially uncertain.

When uncertainty is caused by a missing legally decisive fact, choose clarify
instead of retrieve.

OUTPUT CONSISTENCY

For retrieve:

- atomic_issues must contain between 1 and 3 issues;
- minimum_articles must be at least 1;
- maximum_articles must be greater than or equal to minimum_articles;
- clarification_question_ar must be an empty string.

For clarify or abstain:

- atomic_issues must be empty;
- minimum_articles must be 0;
- maximum_articles must be 0.

Handle Modern Standard Arabic, Jordanian colloquial Arabic, spelling mistakes,
attached Arabic prefixes, Arabic digits, and Western digits.
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
            schema = self._strict_response_schema()
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
