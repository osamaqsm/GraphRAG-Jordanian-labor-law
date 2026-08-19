from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm_provider import build_pipeline_llm

logger = logging.getLogger(__name__)


class ArticleSelection(BaseModel):
    """Strict model output for constrained legal-article selection."""

    model_config = ConfigDict(extra="forbid")

    behavior: Literal["retrieve", "abstain"]
    selected_article_numbers: list[int] = Field(
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

    The model never generates database queries and cannot invent article
    numbers: Python validates every returned number against the catalogue.
    Any API, parsing, or validation failure returns an empty selection so the
    deterministic retriever remains the safe fallback.
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

        # Generic enable flag for the provider-neutral pipeline.  Fall back to
        # the historical OPENAI_* flag so existing configuration still works.
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

        # For fair end-to-end comparison, the reranker must use the same
        # provider/model as all other LLM-dependent pipeline stages.
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
    def _instructions() -> str:
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

    def _payload(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
    ) -> dict[str, Any]:
        catalogue: list[dict[str, Any]] = []

        for article in sorted(
            articles,
            key=lambda item: item.article_number,
        ):
            catalogue.append(
                {
                    "article_number": article.article_number,
                    "text": article.text[: self.max_article_chars],
                    "deterministic_rank": article.deterministic_rank,
                    "deterministic_score": (
                        round(article.deterministic_score, 6)
                        if article.deterministic_score is not None
                        else None
                    ),
                    "graph_supported": article.graph_supported,
                }
            )

        return {
            "question": question,
            "catalogue": catalogue,
        }

    def select(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
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

        valid_numbers = {
            article.article_number
            for article in articles
        }

        try:
            result = self.llm.generate_structured(
                instructions=self._instructions(),
                payload=self._payload(
                    question=question,
                    articles=articles,
                ),
                response_model=ArticleSelection,
                schema_name="legal_article_selection",
                max_output_tokens=int(
                    getattr(
                        self.settings,
                        "reranker_max_output_tokens",
                        2000,
                    )
                ),
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
                }
            )

        except Exception as exc:  # safe deterministic fallback outside strict evaluation
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