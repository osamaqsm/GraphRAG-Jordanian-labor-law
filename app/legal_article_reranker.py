from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

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
        enabled_value = os.getenv(
            "OPENAI_RERANK_ENABLED",
            "true",
        ).strip().lower()
        self.enabled = enabled_value not in {
            "0",
            "false",
            "no",
            "off",
        }

        self.model = os.getenv(
            "OPENAI_RERANK_MODEL",
            getattr(settings, "openai_chat_model", "gpt-5-nano"),
        ).strip()

        self.reasoning_effort = os.getenv(
            "OPENAI_RERANK_REASONING_EFFORT",
            getattr(settings, "openai_reasoning_effort", "low"),
        ).strip()

        self.max_article_chars = max(
            800,
            int(
                os.getenv(
                    "OPENAI_RERANK_ARTICLE_CHAR_LIMIT",
                    "2500",
                )
            ),
        )

        self.client: OpenAI | None = None

        if self.enabled:
            api_key = str(
                getattr(settings, "openai_api_key", "")
                or ""
            ).strip()

            if not api_key:
                logger.warning(
                    "OpenAI article reranking disabled because "
                    "OPENAI_API_KEY is missing."
                )
                self.enabled = False
            else:
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=float(
                        getattr(
                            settings,
                            "openai_timeout_seconds",
                            120,
                        )
                    ),
                    max_retries=int(
                        getattr(
                            settings,
                            "openai_max_retries",
                            3,
                        )
                    ),
                )

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
    ) -> str:
        catalogue = []

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

        return json.dumps(
            {
                "question": question,
                "catalogue": catalogue,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def select(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
    ) -> ArticleSelection | None:
        if not self.enabled or self.client is None or not articles:
            return None

        valid_numbers = {
            article.article_number
            for article in articles
        }

        try:
            schema = ArticleSelection.model_json_schema()
            response = self.client.responses.create(
                model=self.model,
                instructions=self._instructions(),
                input=self._payload(
                    question=question,
                    articles=articles,
                ),
                reasoning={
                    "effort": self.reasoning_effort,
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "legal_article_selection",
                        "schema": schema,
                        "strict": True,
                    }
                },
                store=False,
            )

            if not response.output_text:
                raise RuntimeError(
                    "OpenAI returned no reranking output."
                )

            selection = ArticleSelection.model_validate(
                json.loads(response.output_text)
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
        except Exception as exc:  # safe deterministic fallback
            logger.warning(
                "OpenAI article reranking failed; using deterministic "
                "ranking. Error: %s",
                exc,
            )
            return None
