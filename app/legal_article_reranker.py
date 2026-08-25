from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm_provider import build_pipeline_llm


class IssueCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_index: int = Field(ge=1, le=5)
    covered: bool
    supporting_article_numbers: list[int] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ArticleSelection(BaseModel):
    """Constrained evidence selection over graph-derived candidates only."""

    model_config = ConfigDict(extra="forbid")

    selected_article_numbers: list[int] = Field(default_factory=list, max_length=5)
    issue_coverage: list[IssueCoverage] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@dataclass(frozen=True, slots=True)
class CatalogArticle:
    article_number: int
    text: str
    graph_rank: int | None = None
    graph_score: float | None = None


class LegalArticleReranker:
    """One LLM call that filters a bounded graph-derived article catalogue.

    It cannot discover, request or inject an Article outside the supplied
    catalogue. An empty selection is a valid outcome when graph retrieval did
    not supply evidence that directly supports the in-scope question.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.pipeline_llm_provider
        self.model = settings.pipeline_llm_model
        self.llm = None

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = build_pipeline_llm(self.settings)
        return self.llm

    @staticmethod
    def _instructions() -> str:
        return r"""
You are the evidence-reranking component of a Jordanian Labor Law GraphRAG
system. The router has already decided that the question is in scope.

You receive:
- one user question;
- 1-5 atomic legal issues;
- a bounded catalogue of exact statutory Article text. Every catalogue Article
  was independently discovered through knowledge-graph traversal.

Your task is retrieval/evidence selection only. Do not answer the legal question.

RULES
1. Use ONLY article numbers present in the supplied catalogue.
2. Evaluate every atomic issue independently against the exact statutory text.
3. Set covered=true only when supplied text directly supports the requested
   legal fact/rule/procedure/condition/consequence. Shared topic or proximity is
   not enough.
4. supporting_article_numbers must contain every supplied Article independently
   necessary for that issue, up to five.
5. Every supporting Article must appear in selected_article_numbers.
6. Select the smallest complete set of supplied Articles that directly supports
   the covered issues.
7. If none of the supplied Articles supports an issue, set covered=false and use
   an empty supporting_article_numbers list.
8. If the graph-derived catalogue does not contain sufficient evidence for ANY
   requested issue, selected_article_numbers MUST be an empty list. Do not force
   a selection and do not invent an Article from outside the catalogue.
9. Graph rank/score is a weak hint only. Exact statutory text controls.
10. Never rely on outside legal knowledge or article numbering.
11. Return no more than five selected Article numbers.
""".strip()

    def _payload(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
        issue_pairs: tuple[tuple[str, str], ...],
    ) -> dict[str, Any]:
        bounded = list(articles[: self.settings.reranker_candidate_limit])
        per_article_chars = self.settings.reranker_article_char_limit
        if bounded:
            per_article_chars = min(
                per_article_chars,
                max(600, self.settings.reranker_total_char_budget // len(bounded)),
            )

        return {
            "question": question,
            "issue_plan": [
                {
                    "issue_index": index,
                    "issue_ar": label,
                    "retrieval_query_ar": query,
                }
                for index, (label, query) in enumerate(issue_pairs, start=1)
            ],
            "catalogue": [
                {
                    "article_number": article.article_number,
                    "text": article.text[:per_article_chars],
                    "graph_rank": article.graph_rank,
                    "graph_score": (
                        round(article.graph_score, 6)
                        if article.graph_score is not None
                        else None
                    ),
                }
                for article in bounded
            ],
        }

    def select(
        self,
        *,
        question: str,
        articles: list[CatalogArticle],
        issue_pairs: tuple[tuple[str, str], ...],
    ) -> ArticleSelection:
        if not articles:
            return ArticleSelection(
                selected_article_numbers=[],
                issue_coverage=[
                    IssueCoverage(
                        issue_index=index,
                        covered=False,
                        supporting_article_numbers=[],
                        confidence=1.0,
                        reason="No graph-derived article candidate was supplied.",
                    )
                    for index, _ in enumerate(issue_pairs, start=1)
                ],
                confidence=1.0,
                reason="Graph retrieval returned no candidate article.",
            )

        bounded = list(articles[: self.settings.reranker_candidate_limit])
        valid_numbers = {article.article_number for article in bounded}

        try:
            result = self._ensure_llm().generate_structured(
                instructions=self._instructions(),
                payload=self._payload(
                    question=question,
                    articles=bounded,
                    issue_pairs=issue_pairs,
                ),
                response_model=ArticleSelection,
                schema_name="graph_article_evidence_selection_v3",
                max_output_tokens=self.settings.reranker_max_output_tokens,
            )
            selection = ArticleSelection.model_validate(result.data)

            selected = list(dict.fromkeys(int(n) for n in selection.selected_article_numbers))
            if any(number not in valid_numbers for number in selected):
                raise ValueError("Reranker referenced an Article outside the graph catalogue.")

            coverage_by_index: dict[int, IssueCoverage] = {}
            for coverage in selection.issue_coverage:
                if coverage.issue_index < 1 or coverage.issue_index > len(issue_pairs):
                    continue
                if coverage.issue_index in coverage_by_index:
                    continue
                support = list(
                    dict.fromkeys(int(n) for n in coverage.supporting_article_numbers)
                )
                if any(number not in valid_numbers for number in support):
                    raise ValueError(
                        "Issue coverage referenced an Article outside the graph catalogue."
                    )
                if not coverage.covered or not support:
                    coverage = coverage.model_copy(
                        update={"covered": False, "supporting_article_numbers": []}
                    )
                    support = []
                else:
                    for number in support:
                        if number not in selected:
                            selected.append(number)
                coverage_by_index[coverage.issue_index] = coverage.model_copy(
                    update={"supporting_article_numbers": support[:5]}
                )

            normalized_coverage = [
                coverage_by_index.get(
                    index,
                    IssueCoverage(
                        issue_index=index,
                        covered=False,
                        supporting_article_numbers=[],
                        confidence=0.0,
                        reason="The reranker did not return a coverage item for this issue.",
                    ),
                )
                for index in range(1, len(issue_pairs) + 1)
            ]

            # Do not keep an Article unless it supports at least one covered
            # issue. This removes topical-but-irrelevant graph candidates.
            supported = {
                n
                for coverage in normalized_coverage
                if coverage.covered
                for n in coverage.supporting_article_numbers
            }
            selected = [number for number in selected if number in supported][:5]

            return selection.model_copy(
                update={
                    "selected_article_numbers": selected,
                    "issue_coverage": normalized_coverage,
                }
            )
        except Exception as exc:
            raise RuntimeError(
                "Article reranker failed. "
                f"Provider={self.provider} Model={self.model} "
                f"Error={type(exc).__name__}: {exc}"
            ) from exc
