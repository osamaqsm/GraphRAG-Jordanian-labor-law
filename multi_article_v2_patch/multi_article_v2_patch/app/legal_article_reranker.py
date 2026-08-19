from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm_provider import build_pipeline_llm

logger = logging.getLogger(__name__)




class BasicArticleSelection(BaseModel):
    """Original V1 selection schema used for single-issue questions."""

    model_config = ConfigDict(extra="forbid")

    behavior: Literal["retrieve", "abstain"]
    selected_article_numbers: list[int] = Field(max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class IssueCoverage(BaseModel):
    """Coverage judgment for one planner-decomposed atomic issue."""

    model_config = ConfigDict(extra="forbid")

    issue_index: int = Field(ge=1, le=5)
    covered: bool
    supporting_article_numbers: list[int] = Field(
        default_factory=list,
        max_length=1,
    )
    reason: str = ""


class ArticleSelection(BaseModel):
    """Strict model output for constrained legal-article selection."""

    model_config = ConfigDict(extra="forbid")

    behavior: Literal["retrieve", "abstain"]
    selected_article_numbers: list[int] = Field(
        max_length=5,
    )
    issue_coverage: list[IssueCoverage] = Field(
        default_factory=list,
        max_length=5,
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@dataclass(frozen=True, slots=True)
class CatalogArticle:
    article_number: int
    text: str
    deterministic_rank: int | None = None
    deterministic_score: float | None = None
    graph_supported: bool = False


class LegalArticleReranker:
    """
    Select the minimum complete article set from the supplied catalogue.

    Multi-issue questions also receive the planner's atomic issue plan. The
    reranker must report one coverage judgment per issue, and Python validates
    all cited article numbers against the supplied catalogue.
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
            "PIPELINE_RERANK_ENABLED",
            os.getenv(
                "OPENAI_RERANK_ENABLED",
                "true",
            ),
        ).strip().lower()
        self.enabled = enabled_value not in {
            "0",
            "false",
            "no",
            "off",
        }

        self.provider = str(
            getattr(settings, "pipeline_llm_provider", "openai")
        ).strip().lower()
        self.model = str(
            getattr(settings, "pipeline_llm_model", "gpt-5-nano")
        ).strip()

        self.max_article_chars = max(
            800,
            int(
                os.getenv(
                    "PIPELINE_RERANK_ARTICLE_CHAR_LIMIT",
                    os.getenv(
                        "OPENAI_RERANK_ARTICLE_CHAR_LIMIT",
                        "2500",
                    ),
                )
            ),
        )

        self.candidate_limit = int(
            getattr(settings, "reranker_candidate_limit", 12)
        )
        self.multi_issue_candidate_limit = int(
            getattr(settings, "reranker_multi_issue_candidate_limit", 15)
        )
        self.total_char_budget = int(
            getattr(settings, "reranker_total_char_budget", 12000)
        )

        self.llm = None

        if self.enabled:
            try:
                self.llm = build_pipeline_llm(settings)
            except Exception as exc:
                logger.warning(
                    "Pipeline article reranking disabled because the "
                    "configured LLM provider could not be initialized. "
                    "Provider=%s Model=%s Error=%s",
                    self.provider,
                    self.model,
                    exc,
                )
                self.enabled = False

    @staticmethod
    def _basic_instructions() -> str:
        return """
You are a constrained Jordanian Labor Law article selector.

You receive one user question and the exact text of every available article.
Your task is retrieval only, not legal advice and not answer generation.

Rules:
1. Select only article numbers present in the supplied catalogue.
2. Identify each independently requested legal fact, rule, procedure,
   consequence, stage, authority, condition, or penalty before selecting.
3. Select the smallest complete set that covers all requested parts. The
   five-article limit is a ceiling, never a target.
4. One article may cover several requested parts. When it does, do not add
   neighbouring or background articles merely to create one article per part.
5. Add a second or later article only when its text supplies an independently
   requested answer element that the already selected articles do not supply.
6. Before returning, verify issue by issue:
   - every requested part is covered by at least one selected article;
   - every selected article contributes a requested part;
   - removing any selected article would make the answer incomplete.
7. Order the principal substantive provisions first, followed only by necessary
   procedural, remedy, consequence, or penalty provisions.
8. Distinguish neighbouring provisions in the same chapter by their exact
   actors, facts, conditions, numbers, deadlines, procedures, and legal
   consequences. Shared topic or chapter proximity is not sufficient.
9. Do not select introductory, definitional, adjacent, or generally useful
   provisions unless the question expressly asks for what they contain.
10. Handle Arabic paraphrases, Jordanian colloquial wording, spelling errors,
    attached prefixes, and Western or Arabic digits.
11. Ignore deterministic ranking when exact statutory text supports a different
    article; ranking data is only a weak hint.
12. For every in-scope question, return behavior=retrieve and the minimal
    complete article set; never request clarification.
13. If the question is outside Jordanian labor law, return behavior=abstain and
    no articles.
14. Never rely on knowledge not contained in the supplied article catalogue.
15. Return no more than five article numbers.
""".strip()

    @staticmethod
    def _instructions() -> str:
        return """
You are a constrained Jordanian Labor Law article selector.

You receive:
- one user question;
- an optional issue_plan containing independently requested legal issues;
- a bounded catalogue containing exact article text.

Your task is retrieval only, not legal advice and not answer generation.

Rules:
1. Select only article numbers present in the supplied catalogue.
2. When issue_plan is non-empty, evaluate EVERY issue independently before
   selecting the final article set.
3. Select the smallest complete set that covers all requested issues. The
   five-article limit is a ceiling, never a target.
4. One article may cover several issues. Do not add neighbouring/background
   articles merely to create one article per issue.
5. For each issue_plan item, return exactly one issue_coverage item using the
   same issue_index. Set covered=true only when at least one supplied article
   directly provides the requested rule/fact/procedure/consequence.
6. When covered=true, supporting_article_numbers must contain exactly one best
   supporting article and that article must also be in selected_article_numbers.
7. When covered=false, supporting_article_numbers must be empty. Never pretend
   an issue is covered by a merely related article.
8. Before returning, verify issue by issue that each requested part is either
   covered by a selected article or explicitly marked uncovered.
9. Distinguish neighbouring provisions by exact actors, facts, conditions,
   numbers, deadlines, procedures and legal consequences. Shared topic or
   chapter proximity is not sufficient.
10. Handle Arabic paraphrases, Jordanian colloquial wording, spelling errors,
    attached prefixes, and Western or Arabic digits.
11. Deterministic ranking is only a weak hint. Exact statutory text controls.
12. For every in-scope question, return behavior=retrieve. If the question is
    outside Jordanian labor law, return behavior=abstain with no articles.
13. Never rely on knowledge outside the supplied catalogue.
14. Return no more than five selected article numbers.
""".strip()

    def _payload(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
        issue_pairs: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, Any]:
        candidate_limit = self.candidate_limit
        if len(issue_pairs) > 1:
            candidate_limit = max(
                candidate_limit,
                self.multi_issue_candidate_limit,
            )

        bounded_articles = list(articles[:candidate_limit])
        catalogue: list[dict[str, Any]] = []

        per_article_chars = self.max_article_chars
        if bounded_articles:
            per_article_chars = min(
                self.max_article_chars,
                max(600, self.total_char_budget // len(bounded_articles)),
            )

        for article in sorted(
            bounded_articles,
            key=lambda item: item.article_number,
        ):
            catalogue.append(
                {
                    "article_number": article.article_number,
                    "text": article.text[:per_article_chars],
                    "deterministic_rank": article.deterministic_rank,
                    "deterministic_score": (
                        round(article.deterministic_score, 6)
                        if article.deterministic_score is not None
                        else None
                    ),
                    "graph_supported": article.graph_supported,
                }
            )

        issue_plan = [
            {
                "issue_index": index,
                "issue_ar": label,
                "retrieval_query_ar": query,
            }
            for index, (label, query) in enumerate(issue_pairs, start=1)
        ]

        return {
            "question": question,
            "issue_plan": issue_plan,
            "catalogue": catalogue,
        }

    def select(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
        issue_pairs: tuple[tuple[str, str], ...] = (),
    ) -> ArticleSelection | None:
        if not articles:
            return None

        if not self.enabled or self.llm is None:
            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: article reranker is unavailable. "
                    f"Provider={self.provider} Model={self.model}"
                )
            return None

        candidate_limit = self.candidate_limit
        if len(issue_pairs) > 1:
            candidate_limit = max(
                candidate_limit,
                self.multi_issue_candidate_limit,
            )

        bounded_articles = list(articles[:candidate_limit])
        valid_numbers = {
            article.article_number
            for article in bounded_articles
        }

        try:
            max_output_tokens = int(
                getattr(
                    self.settings,
                    "reranker_max_output_tokens",
                    2000,
                )
            )

            if len(issue_pairs) <= 1:
                # Preserve the exact V1 reranker contract for simple questions.
                result = self.llm.generate_structured(
                    instructions=self._basic_instructions(),
                    payload={
                        "question": question,
                        "catalogue": self._payload(
                            question=question,
                            articles=bounded_articles,
                            issue_pairs=(),
                        )["catalogue"],
                    },
                    response_model=BasicArticleSelection,
                    schema_name="legal_article_selection",
                    max_output_tokens=max_output_tokens,
                )
                basic_selection = BasicArticleSelection.model_validate(
                    result.data
                )
                selection = ArticleSelection(
                    behavior=basic_selection.behavior,
                    selected_article_numbers=(
                        basic_selection.selected_article_numbers
                    ),
                    issue_coverage=[],
                    confidence=basic_selection.confidence,
                    reason=basic_selection.reason,
                )
            else:
                result = self.llm.generate_structured(
                    instructions=self._instructions(),
                    payload=self._payload(
                        question=question,
                        articles=bounded_articles,
                        issue_pairs=issue_pairs,
                    ),
                    response_model=ArticleSelection,
                    schema_name="legal_article_selection_v2",
                    max_output_tokens=max_output_tokens,
                )
                selection = ArticleSelection.model_validate(
                    result.data
                )

            ordered_unique = list(
                dict.fromkeys(
                    int(number)
                    for number in selection.selected_article_numbers
                )
            )

            if any(
                number not in valid_numbers
                for number in ordered_unique
            ):
                raise ValueError(
                    "The reranker returned an article number outside "
                    "the supplied catalogue."
                )

            valid_issue_indices = set(range(1, len(issue_pairs) + 1))
            seen_issue_indices: set[int] = set()
            normalized_coverage: list[IssueCoverage] = []

            for coverage in selection.issue_coverage:
                if coverage.issue_index not in valid_issue_indices:
                    continue
                if coverage.issue_index in seen_issue_indices:
                    continue
                seen_issue_indices.add(coverage.issue_index)

                support_numbers = list(
                    dict.fromkeys(
                        int(number)
                        for number in coverage.supporting_article_numbers
                    )
                )
                if any(number not in valid_numbers for number in support_numbers):
                    raise ValueError(
                        "Issue coverage referenced an article outside the "
                        "supplied catalogue."
                    )

                if coverage.covered and not support_numbers:
                    coverage = coverage.model_copy(
                        update={"covered": False}
                    )
                    support_numbers = []

                if not coverage.covered:
                    support_numbers = []

                if coverage.covered:
                    for number in support_numbers:
                        if number not in ordered_unique:
                            ordered_unique.append(number)

                normalized_coverage.append(
                    coverage.model_copy(
                        update={
                            "supporting_article_numbers": support_numbers[:1],
                        }
                    )
                )

            if selection.behavior == "retrieve" and not ordered_unique:
                raise ValueError(
                    "The reranker chose retrieve without an article."
                )

            if selection.behavior != "retrieve" and ordered_unique:
                raise ValueError(
                    "The reranker returned articles for a non-retrieval "
                    "decision."
                )

            return selection.model_copy(
                update={
                    "selected_article_numbers": ordered_unique[:5],
                    "issue_coverage": normalized_coverage,
                }
            )

        except Exception as exc:
            if self.strict_evaluation:
                raise RuntimeError(
                    "Strict evaluation aborted: article reranker failed. "
                    f"Provider={self.provider} Model={self.model} "
                    f"Error={type(exc).__name__}: {exc}"
                ) from exc

            logger.warning(
                "Pipeline article reranking failed; using deterministic "
                "ranking. Provider=%s Model=%s Error=%s",
                self.provider,
                self.model,
                exc,
            )
            return None
