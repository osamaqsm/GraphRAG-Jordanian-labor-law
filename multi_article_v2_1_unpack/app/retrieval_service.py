from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import math
import re
from typing import Any

from weaviate.classes.query import (
    Filter,
    MetadataQuery,
)

from app.config import Settings
from app.graph_retrieval import (
    GraphExpansionResult,
    GraphTraversalService,
    select_graph_seeds,
)
from app.legal_article_reranker import (
    CatalogArticle,
    LegalArticleReranker,
)
from app.legal_query_planner import LegalQueryPlanner
from app.legal_question_analysis import (
    LegalQuestionAnalysis,
    analyze_legal_question,
    article_question_relevance,
    normalize_arabic,
)
from app.retrieval_models import (
    RetrievalHit,
    RetrievalPreview,
)


RETURN_PROPERTIES = [
    "uri",
    "localName",
    "nodeKind",
    "labelsAr",
    "labelsEn",
    "commentsAr",
    "commentsEn",
    "articleNumber",
    "searchableText",
]


NON_CONCEPT_NODE_KINDS = {
    "Article",
    "Paragraph",
    "Definition",
    "Law",
    "Ontology",
    "OntologyClass",
    "ObjectProperty",
    "DatatypeProperty",
    "AnnotationProperty",
    "Resource",
}


# Stage 7.5-A hybrid article ranking.  Graph evidence is important, but it
# must not suppress a strong direct article or paragraph match.  Each source
# is converted to a comparable rank-based score before the weighted sum.
DIRECT_ARTICLE_WEIGHT = 0.30
GRAPH_ARTICLE_WEIGHT = 0.15
PARAGRAPH_ARTICLE_WEIGHT = 0.10
QUESTION_ARTICLE_WEIGHT = 0.45
SOURCE_AGREEMENT_BONUS = 0.04
ANCHOR_BM25_WEIGHT = 3.0
PLANNER_BM25_WEIGHT = 2.0
ISSUE_CATALOG_CANDIDATES = 12
ISSUE_TOP_MARGIN = 0.08
ISSUE_TOP_BONUS = 0.22
MULTI_ARTICLE_COVERAGE_BONUS = 0.50
ISSUE_COVERAGE_MIN_RELEVANCE = 0.10
FINAL_ARTICLE_LIMIT = 5


_ISSUE_QUERY_BOILERPLATE_RE = re.compile(
    r"(?:ما\s+هي\s+احكام|ما\s+هي|ما\s+حكم|هل\s+يوجد|"
    r"وفق\s+قانون\s+العمل\s+الاردني|في\s+قانون\s+العمل\s+الاردني|"
    r"ضمن\s+قانون\s+العمل\s+الاردني|قانون\s+العمل\s+الاردني)",
    flags=re.IGNORECASE,
)


def _focused_issue_text(label: str, query: str) -> str:
    """Create a compact retrieval phrase for one atomic legal issue.

    Planner queries are useful for decomposition but some models naturally
    phrase them as full questions (for example, "ما هي أحكام ... وفق قانون
    العمل الأردني؟"). Those generic words are actively harmful to BM25 and
    semantic retrieval because they occur across the statute. Keep the concise
    issue label and only retain non-boilerplate content from the query.
    """

    normalized_label = normalize_arabic(label)
    normalized_query = normalize_arabic(query)
    cleaned_query = _ISSUE_QUERY_BOILERPLATE_RE.sub(
        " ",
        normalized_query,
    )
    cleaned_query = " ".join(cleaned_query.split())

    parts: list[str] = []
    for value in (normalized_label, cleaned_query):
        if value and value not in parts:
            parts.append(value)

    return " ".join(parts).strip() or normalized_query or normalized_label


def _atomic_issue_analysis(label: str, query: str) -> LegalQuestionAnalysis:
    """Build retrieval analysis from the atomic issue itself, not V1 profiles.

    The deterministic V1 IssueProfile catalogue intentionally covers common
    single-issue questions, but it is not exhaustive enough to define every
    planner-created atomic issue. Re-running that analyzer on a new planner
    query can therefore collapse a specific issue back to a generic legal
    question. This V2.1 analysis remains article-number agnostic and uses the
    planner decomposition directly.
    """

    focused = _focused_issue_text(label, query)
    normalized = normalize_arabic(focused)
    tokens = frozenset(
        token
        for token in normalized.split()
        if len(token) > 1
    )
    numeric_tokens = frozenset(
        token
        for token in normalized.split()
        if token.isdigit()
    )

    return LegalQuestionAnalysis(
        original_question=focused,
        normalized_question=normalized,
        issue_ids=(),
        preferred_concepts=(),
        query_expansion_terms=(),
        article_anchor_phrases=(focused,) if focused else (),
        primary_article_anchor_phrases=(focused,) if focused else (),
        meaningful_tokens=tokens,
        numeric_tokens=numeric_tokens,
        max_final_articles=1,
        behavior="retrieve",
        behavior_reason="Atomic issue retrieval",
        planner_queries=(),
        planner_issue_labels=(label.strip() or focused,),
        planner_used=True,
        planner_confidence=1.0,
        clarification_question="",
    )


def _list_value(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            str(item)
            for item in value
        ]

    return [str(value)]


def _first_text(
    properties: dict[str, Any],
) -> str:
    arabic_comments = _list_value(
        properties.get("commentsAr")
    )

    if arabic_comments:
        return arabic_comments[0][:2500]

    english_comments = _list_value(
        properties.get("commentsEn")
    )

    if english_comments:
        return english_comments[0][:2500]

    searchable_text = str(
        properties.get(
            "searchableText",
            "",
        )
        or ""
    )

    return searchable_text[:2500]


def _make_hit(
    properties: dict[str, Any],
) -> RetrievalHit:
    article_number_value = properties.get(
        "articleNumber"
    )

    article_number: int | None = None

    if article_number_value is not None:
        article_number = int(
            article_number_value
        )

    return RetrievalHit(
        uri=str(
            properties.get(
                "uri",
                "",
            )
        ),
        local_name=str(
            properties.get(
                "localName",
                "",
            )
        ),
        node_kind=str(
            properties.get(
                "nodeKind",
                "",
            )
        ),
        labels_ar=_list_value(
            properties.get("labelsAr")
        ),
        labels_en=_list_value(
            properties.get("labelsEn")
        ),
        article_number=article_number,
        text_preview=_first_text(
            properties
        ),
    )


class RetrievalService:
    """
    Graph-aware retrieval service.

    Ranking priority:
        1. Graph-supported legal articles
        2. Paragraph-promoted articles
        3. Direct vector retrieval
        4. Direct BM25 retrieval
    """

    def __init__(
        self,
        client: Any,
        settings: Settings,
    ) -> None:
        self.client = client
        self.settings = settings

        self.nodes = client.collections.use(
            settings.weaviate_node_collection
        )

        self.graph = GraphTraversalService(
            client=client,
            settings=settings,
        )

        self.query_planner = LegalQueryPlanner(
            settings=settings,
        )
        self.article_reranker = LegalArticleReranker(
            settings=settings,
        )
        self._article_catalog_cache: list[RetrievalHit] | None = None
        self.last_issue_coverage_debug: dict[str, Any] = {}

    @staticmethod
    def _merge_hit_groups(
        groups: list[list[RetrievalHit]],
        *,
        limit: int | None = None,
    ) -> list[RetrievalHit]:
        """Interleave issue-specific rankings while de-duplicating by URI.

        Interleaving prevents the strongest issue from consuming the whole
        candidate pool before weaker atomic issues contribute candidates.
        """

        merged: list[RetrievalHit] = []
        by_uri: dict[str, RetrievalHit] = {}
        max_group_size = max((len(group) for group in groups), default=0)

        for rank_index in range(max_group_size):
            for group in groups:
                if rank_index >= len(group):
                    continue
                source = group[rank_index]
                if not source.uri:
                    continue

                existing = by_uri.get(source.uri)
                if existing is None:
                    existing = replace(source)
                    existing.labels_ar = list(source.labels_ar)
                    existing.labels_en = list(source.labels_en)
                    existing.support_paths = list(source.support_paths)
                    by_uri[source.uri] = existing
                    merged.append(existing)
                else:
                    existing.fused_score = max(
                        existing.fused_score, source.fused_score
                    )
                    existing.direct_score = max(
                        existing.direct_score, source.direct_score
                    )
                    existing.graph_score = max(
                        existing.graph_score, source.graph_score
                    )
                    existing.paragraph_score = max(
                        existing.paragraph_score, source.paragraph_score
                    )
                    existing.final_score = max(
                        existing.final_score, source.final_score
                    )
                    existing.graph_supported = (
                        existing.graph_supported or source.graph_supported
                    )
                    if source.support_paths:
                        known = {repr(path) for path in existing.support_paths}
                        for path in source.support_paths:
                            if repr(path) not in known:
                                existing.support_paths.append(path)

                if limit is not None and len(merged) >= limit:
                    return merged

        return merged

    @staticmethod
    def _merge_expansions(
        expansions: list[GraphExpansionResult],
    ) -> GraphExpansionResult:
        article_evidence: dict[str, dict[str, Any]] = {}
        related_concepts: dict[str, float] = {}

        for expansion in expansions:
            for uri, evidence in expansion.article_evidence.items():
                target = article_evidence.setdefault(
                    uri,
                    {"score": 0.0, "paths": []},
                )
                target["score"] = max(
                    float(target["score"]),
                    float(evidence.get("score", 0.0)),
                )
                known = {repr(path) for path in target["paths"]}
                for path in evidence.get("paths", []):
                    if repr(path) not in known:
                        target["paths"].append(path)

            for uri, score in expansion.related_concepts.items():
                related_concepts[uri] = max(
                    related_concepts.get(uri, 0.0),
                    float(score),
                )

        return GraphExpansionResult(
            article_evidence=article_evidence,
            related_concepts=related_concepts,
        )

    @staticmethod
    def _concept_filter() -> Any:
        """
        Exclude document nodes and OWL schema resources
        before searching, rather than filtering afterward.
        """

        exclusions = [
            Filter.not_(
                Filter
                .by_property("nodeKind")
                .equal(node_kind)
            )
            for node_kind in sorted(
                NON_CONCEPT_NODE_KINDS
            )
        ]

        return Filter.all_of(
            exclusions
        )

    def _vector_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        filters: Any | None = None,
    ) -> list[Any]:
        response = self.nodes.query.near_vector(
            near_vector=query_vector,
            target_vector="default",
            filters=filters,
            limit=limit,
            return_properties=RETURN_PROPERTIES,
            return_metadata=MetadataQuery(
                distance=True
            ),
        )

        return list(response.objects)

    def _bm25_search(
        self,
        question: str,
        *,
        limit: int,
        filters: Any | None = None,
        concept_mode: bool = False,
    ) -> list[Any]:
        if concept_mode:
            query_properties = [
                "labelsAr^5",
                "labelsEn^2",
                "searchableText",
            ]
        else:
            query_properties = [
                "labelsAr^4",
                "commentsAr^2",
                "labelsEn^2",
                "searchableText",
            ]

        response = self.nodes.query.bm25(
            query=question,
            query_properties=query_properties,
            filters=filters,
            limit=limit,
            return_properties=RETURN_PROPERTIES,
            return_metadata=MetadataQuery(
                score=True
            ),
        )

        return list(response.objects)

    def _anchor_search_objects(
        self,
        analysis: LegalQuestionAnalysis,
        *,
        filters: Any,
        limit: int,
    ) -> list[Any]:
        """
        Run a few focused statutory-anchor BM25 searches and interleave
        their results by rank.

        Interleaving lets an article accumulate evidence when it appears
        for multiple legal aspects, while preventing one long BM25 query
        from being dominated by generic words such as worker or employer.
        """

        query_groups: list[list[Any]] = []
        per_query_limit = max(
            5,
            min(limit, 12),
        )

        for anchor_query in analysis.anchor_queries:
            objects = self._bm25_search(
                anchor_query,
                limit=per_query_limit,
                filters=filters,
            )

            if objects:
                query_groups.append(objects)

        if not query_groups:
            return []

        interleaved: list[Any] = []
        max_group_size = max(
            len(group)
            for group in query_groups
        )

        for rank_index in range(max_group_size):
            for group in query_groups:
                if rank_index < len(group):
                    interleaved.append(
                        group[rank_index]
                    )

        return interleaved

    def _planner_search_objects(
        self,
        analysis: LegalQuestionAnalysis,
        *,
        filters: Any,
        limit: int,
        concept_mode: bool = False,
    ) -> list[Any]:
        """Run each LLM-generated atomic query independently and interleave."""

        if not analysis.planner_queries:
            return []

        query_groups: list[list[Any]] = []
        per_query_limit = max(5, min(limit, 12))

        for planner_query in analysis.planner_queries:
            objects = self._bm25_search(
                planner_query,
                limit=per_query_limit,
                filters=filters,
                concept_mode=concept_mode,
            )
            if objects:
                query_groups.append(objects)

        if not query_groups:
            return []

        interleaved: list[Any] = []
        seen_uris: set[str] = set()
        max_group_size = max(len(group) for group in query_groups)

        for rank_index in range(max_group_size):
            for group in query_groups:
                if rank_index >= len(group):
                    continue
                item = group[rank_index]
                uri = str(dict(item.properties).get("uri", ""))
                if uri and uri in seen_uris:
                    continue
                if uri:
                    seen_uris.add(uri)
                interleaved.append(item)

        return interleaved

    def _fuse(
        self,
        vector_objects: Iterable[Any],
        bm25_objects: Iterable[Any],
        *,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        anchor_objects: Iterable[Any] = (),
        anchor_weight: float = 0.0,
        planner_objects: Iterable[Any] = (),
        planner_weight: float = 0.0,
    ) -> list[RetrievalHit]:
        """
        Weighted Reciprocal Rank Fusion.

        Vector and BM25 scores cannot safely be added
        directly because they use different scales.
        """

        hits_by_uri: dict[
            str,
            RetrievalHit,
        ] = {}

        rrf_k = self.settings.retrieval_rrf_k

        for rank, item in enumerate(
            vector_objects,
            start=1,
        ):
            properties = dict(
                item.properties
            )

            uri = str(
                properties.get(
                    "uri",
                    "",
                )
            )

            if not uri:
                continue

            hit = hits_by_uri.get(uri)

            if hit is None:
                hit = _make_hit(properties)
                hits_by_uri[uri] = hit

            hit.vector_rank = rank

            distance = getattr(
                item.metadata,
                "distance",
                None,
            )

            if distance is not None:
                hit.vector_distance = float(
                    distance
                )

            hit.fused_score += (
                vector_weight
                /
                (rrf_k + rank)
            )

        for rank, item in enumerate(
            bm25_objects,
            start=1,
        ):
            properties = dict(
                item.properties
            )

            uri = str(
                properties.get(
                    "uri",
                    "",
                )
            )

            if not uri:
                continue

            hit = hits_by_uri.get(uri)

            if hit is None:
                hit = _make_hit(properties)
                hits_by_uri[uri] = hit

            hit.bm25_rank = rank

            score = getattr(
                item.metadata,
                "score",
                None,
            )

            if score is not None:
                hit.bm25_score = float(
                    score
                )

            hit.fused_score += (
                bm25_weight
                /
                (rrf_k + rank)
            )

        # A dedicated issue-anchor BM25 query is used only for legal
        # article/paragraph candidates. It recovers provisions whose exact
        # statutory wording is diluted in the broader natural-language query.
        for rank, item in enumerate(
            anchor_objects,
            start=1,
        ):
            properties = dict(
                item.properties
            )

            uri = str(
                properties.get(
                    "uri",
                    "",
                )
            )

            if not uri:
                continue

            hit = hits_by_uri.get(uri)

            if hit is None:
                hit = _make_hit(properties)
                hits_by_uri[uri] = hit

            score = getattr(
                item.metadata,
                "score",
                None,
            )

            if (
                score is not None
                and hit.bm25_score is None
            ):
                hit.bm25_score = float(
                    score
                )

            hit.fused_score += (
                anchor_weight
                /
                (rrf_k + rank)
            )


        for rank, item in enumerate(
            planner_objects,
            start=1,
        ):
            properties = dict(item.properties)
            uri = str(properties.get("uri", ""))
            if not uri:
                continue

            hit = hits_by_uri.get(uri)
            if hit is None:
                hit = _make_hit(properties)
                hits_by_uri[uri] = hit

            score = getattr(item.metadata, "score", None)
            if score is not None and hit.bm25_score is None:
                hit.bm25_score = float(score)

            hit.fused_score += planner_weight / (rrf_k + rank)

        return sorted(
            hits_by_uri.values(),
            key=lambda hit: (
                -hit.fused_score,
                (
                    hit.vector_distance
                    if hit.vector_distance
                    is not None
                    else float("inf")
                ),
                hit.uri,
            ),
        )

    def _all_article_catalog_hits(
        self,
    ) -> list[RetrievalHit]:
        """Fetch and cache the complete 142-article catalogue."""

        if self._article_catalog_cache is not None:
            return self._article_catalog_cache

        response = self.nodes.query.fetch_objects(
            filters=(
                Filter
                .by_property("nodeKind")
                .equal("Article")
            ),
            limit=500,
            return_properties=RETURN_PROPERTIES,
        )

        hits = [
            _make_hit(dict(item.properties))
            for item in response.objects
        ]

        hits = [
            hit
            for hit in hits
            if hit.article_number is not None
        ]

        hits.sort(
            key=lambda hit: (
                hit.article_number
                if hit.article_number is not None
                else 999999
            )
        )

        self._article_catalog_cache = hits
        return hits

    def _issue_article_candidates(
        self,
        analysis: LegalQuestionAnalysis,
    ) -> list[RetrievalHit]:
        """
        Score the complete statutory article catalogue for candidate recovery.

        The current law contains only 142 articles, so a local deterministic
        scan is inexpensive and prevents a highly relevant provision from
        disappearing merely because it missed the vector/BM25 cut-off.  No
        article number is encoded in the analyzer or in this method.
        """

        scored: list[tuple[float, RetrievalHit]] = []

        for cached_hit in self._all_article_catalog_hits():

            score = article_question_relevance(
                analysis,
                " ".join(
                    [
                        *cached_hit.labels_ar,
                        *cached_hit.labels_en,
                        cached_hit.text_preview,
                    ]
                ),
            )

            if score <= 0.0:
                continue

            # Never mutate the cached full-law catalogue. Multi-issue retrieval
            # scores the same article independently for several atomic queries.
            hit = replace(cached_hit)
            hit.direct_score = score
            scored.append((score, hit))

        scored.sort(
            key=lambda item: (
                -item[0],
                (
                    item[1].article_number
                    if item[1].article_number is not None
                    else 999999
                ),
            )
        )

        return [
            hit
            for _, hit in scored[:ISSUE_CATALOG_CANDIDATES]
        ]

    def _fetch_nodes_by_uris(
        self,
        uris: list[str],
    ) -> dict[str, RetrievalHit]:
        """
        Retrieve node details for exact RDF URIs.
        """

        unique_uris = sorted(
            set(uris)
        )

        if not unique_uris:
            return {}

        response = self.nodes.query.fetch_objects(
            filters=(
                Filter
                .by_property("uri")
                .contains_any(unique_uris)
            ),
            limit=len(unique_uris),
            return_properties=RETURN_PROPERTIES,
        )

        result: dict[
            str,
            RetrievalHit,
        ] = {}

        for item in response.objects:
            hit = _make_hit(
                dict(item.properties)
            )

            if hit.uri:
                result[hit.uri] = hit

        return result

    def _fetch_nodes_by_local_names(
        self,
        local_names: list[str],
    ) -> dict[str, RetrievalHit]:
        """Retrieve exact ontology/KG concepts by their local names."""

        unique_names = sorted(
            {
                name
                for name in local_names
                if name
            }
        )

        if not unique_names:
            return {}

        response = self.nodes.query.fetch_objects(
            filters=(
                Filter
                .by_property("localName")
                .contains_any(unique_names)
            ),
            limit=len(unique_names),
            return_properties=RETURN_PROPERTIES,
        )

        result: dict[str, RetrievalHit] = {}

        for item in response.objects:
            hit = _make_hit(
                dict(item.properties)
            )

            if hit.local_name:
                result[hit.local_name] = hit

        return result

    def search_concepts(
        self,
        question: str,
        query_vector: list[float],
        analysis: LegalQuestionAnalysis,
    ) -> list[RetrievalHit]:
        """
        Search domain individuals only.

        The filter is applied before vector/BM25 ranking.
        """

        concept_filter = (
            self._concept_filter()
        )

        vector_objects = self._vector_search(
            query_vector,
            limit=(
                self.settings
                .retrieval_vector_candidates
            ),
            filters=concept_filter,
        )

        bm25_objects = self._bm25_search(
            analysis.bm25_query,
            limit=(
                self.settings
                .retrieval_bm25_candidates
            ),
            filters=concept_filter,
            concept_mode=True,
        )

        planner_objects = self._planner_search_objects(
            analysis,
            filters=concept_filter,
            limit=self.settings.retrieval_bm25_candidates,
            concept_mode=True,
        )

        results = self._fuse(
            vector_objects,
            bm25_objects,
            vector_weight=1.5,
            bm25_weight=1.0,
            planner_objects=planner_objects,
            planner_weight=PLANNER_BM25_WEIGHT,
        )

        # Add exact, analyzer-validated ontology concepts even when vector
        # or BM25 retrieval ranks them below the diagnostic cutoff. The
        # analyzer supplies concept names only, never article numbers.
        preferred_node_map = (
            self._fetch_nodes_by_local_names(
                list(analysis.preferred_concepts)
            )
        )

        hits_by_uri = {
            hit.uri: hit
            for hit in results
        }

        for preferred_hit in (
            preferred_node_map.values()
        ):
            hits_by_uri.setdefault(
                preferred_hit.uri,
                preferred_hit,
            )

        merged_results = list(
            hits_by_uri.values()
        )

        # Put analyzer-validated concepts and safe graph seeds first. The
        # remaining concepts stay available for diagnostics but do not
        # control graph expansion.
        strong_seeds = select_graph_seeds(
            concept_hits=merged_results,
            limit=(
                self.settings
                .retrieval_concept_top_k
            ),
            preferred_local_names=(
                analysis.preferred_concepts
            ),
        )

        strong_seed_uris = {
            hit.uri
            for hit in strong_seeds
        }

        preferred_order = {
            local_name: index
            for index, local_name in enumerate(
                analysis.preferred_concepts
            )
        }

        preferred_non_seeds = sorted(
            (
                hit
                for hit in merged_results
                if (
                    hit.uri not in strong_seed_uris
                    and hit.local_name
                    in preferred_order
                )
            ),
            key=lambda hit: (
                preferred_order[hit.local_name],
                hit.uri,
            ),
        )

        preferred_uris = {
            hit.uri
            for hit in preferred_non_seeds
        }

        remaining_results = sorted(
            (
                hit
                for hit in merged_results
                if (
                    hit.uri not in strong_seed_uris
                    and hit.uri not in preferred_uris
                )
            ),
            key=lambda hit: (
                -hit.fused_score,
                (
                    hit.vector_distance
                    if hit.vector_distance
                    is not None
                    else float("inf")
                ),
                hit.uri,
            ),
        )

        ordered_results = [
            *strong_seeds,
            *preferred_non_seeds,
            *remaining_results,
        ]

        return ordered_results[
            :self.settings
            .retrieval_concept_top_k
        ]

    def search_articles_direct(
        self,
        question: str,
        query_vector: list[float],
        analysis: LegalQuestionAnalysis,
    ) -> list[RetrievalHit]:
        article_filter = (
            Filter
            .by_property("nodeKind")
            .equal("Article")
        )

        vector_objects = self._vector_search(
            query_vector,
            limit=(
                self.settings
                .retrieval_vector_candidates
            ),
            filters=article_filter,
        )

        bm25_objects = self._bm25_search(
            analysis.bm25_query,
            limit=(
                self.settings
                .retrieval_bm25_candidates
            ),
            filters=article_filter,
        )

        anchor_objects = self._anchor_search_objects(
            analysis,
            filters=article_filter,
            limit=(
                self.settings
                .retrieval_bm25_candidates
            ),
        )

        planner_objects = self._planner_search_objects(
            analysis,
            filters=article_filter,
            limit=self.settings.retrieval_bm25_candidates,
        )

        results = self._fuse(
            vector_objects,
            bm25_objects,
            vector_weight=(
                self.settings
                .retrieval_vector_weight
            ),
            bm25_weight=(
                self.settings
                .retrieval_bm25_weight
            ),
            anchor_objects=anchor_objects,
            anchor_weight=ANCHOR_BM25_WEIGHT,
            planner_objects=planner_objects,
            planner_weight=PLANNER_BM25_WEIGHT,
        )

        for hit in results:
            hit.direct_score = (
                hit.fused_score
            )

        # Preserve the normal vector/BM25 ordering, but append the best
        # deterministic full-catalog matches when they were not retrieved.
        # Their question relevance is evaluated again during final ranking.
        hits_by_uri = {
            hit.uri: hit
            for hit in results
        }

        for catalog_hit in self._issue_article_candidates(analysis):
            hits_by_uri.setdefault(
                catalog_hit.uri,
                catalog_hit,
            )

        return list(hits_by_uri.values())

    def search_paragraphs(
        self,
        question: str,
        query_vector: list[float],
        analysis: LegalQuestionAnalysis,
    ) -> list[RetrievalHit]:
        paragraph_filter = (
            Filter
            .by_property("nodeKind")
            .equal("Paragraph")
        )

        vector_objects = self._vector_search(
            query_vector,
            limit=(
                self.settings
                .retrieval_paragraph_candidates
            ),
            filters=paragraph_filter,
        )

        bm25_objects = self._bm25_search(
            analysis.bm25_query,
            limit=(
                self.settings
                .retrieval_paragraph_candidates
            ),
            filters=paragraph_filter,
        )

        anchor_objects: list[Any] = []

        if analysis.anchor_query:
            anchor_objects = self._bm25_search(
                analysis.anchor_query,
                limit=(
                    self.settings
                    .retrieval_paragraph_candidates
                ),
                filters=paragraph_filter,
            )

        planner_objects = self._planner_search_objects(
            analysis,
            filters=paragraph_filter,
            limit=self.settings.retrieval_paragraph_candidates,
        )

        return self._fuse(
            vector_objects,
            bm25_objects,
            vector_weight=1.5,
            bm25_weight=1.0,
            anchor_objects=anchor_objects,
            anchor_weight=ANCHOR_BM25_WEIGHT,
            planner_objects=planner_objects,
            planner_weight=PLANNER_BM25_WEIGHT,
        )[
            :self.settings
            .retrieval_paragraph_candidates
        ]

    def search_mixed(
        self,
        question: str,
        query_vector: list[float],
    ) -> list[RetrievalHit]:
        """
        Broad search retained for diagnostics only.
        """

        vector_objects = self._vector_search(
            query_vector,
            limit=(
                self.settings
                .retrieval_vector_candidates
            ),
        )

        bm25_objects = self._bm25_search(
            question,
            limit=(
                self.settings
                .retrieval_bm25_candidates
            ),
        )

        return self._fuse(
            vector_objects,
            bm25_objects,
        )

    def _expanded_concepts(
        self,
        expansion: GraphExpansionResult,
    ) -> list[RetrievalHit]:
        node_map = self._fetch_nodes_by_uris(
            list(
                expansion.related_concepts
            )
        )

        hits: list[RetrievalHit] = []

        for uri, score in (
            expansion
            .related_concepts
            .items()
        ):
            hit = node_map.get(uri)

            if hit is None:
                continue

            if (
                hit.node_kind
                in NON_CONCEPT_NODE_KINDS
            ):
                continue

            hit.graph_score = score
            hit.final_score = score
            hit.graph_supported = True

            hits.append(hit)

        return sorted(
            hits,
            key=lambda hit: (
                -hit.graph_score,
                hit.uri,
            ),
        )[
            :self.settings
            .retrieval_concept_top_k
        ]

    @staticmethod
    def _rank_score(
        rank: int | None,
    ) -> float:
        """
        Convert a positive 1-based rank into a score in (0, 1].

        Rank-based normalization is used because graph path scores,
        vector/BM25 RRF scores, and paragraph scores have incompatible
        numeric scales.
        """

        if rank is None or rank < 1:
            return 0.0

        return 1.0 / math.log2(rank + 1.0)

    @staticmethod
    def _article_normalized_text(
        hit: RetrievalHit,
    ) -> str:
        return normalize_arabic(
            " ".join(
                [
                    *hit.labels_ar,
                    *hit.labels_en,
                    hit.text_preview,
                ]
            )
        )

    def _select_complementary_articles(
        self,
        ranked: list[RetrievalHit],
        analysis: LegalQuestionAnalysis,
        final_limit: int,
    ) -> list[RetrievalHit]:
        """
        Select supporting provisions that add uncovered legal evidence.

        Multi-provision questions normally contain a principal rule and a
        separate remedy, consequence, or procedure.  Selecting the top N
        articles by one global score can return two provisions that repeat
        the same aspect.  This method keeps the highest-ranked primary
        article, then rewards candidates that cover statutory anchors not
        already covered by the selected articles.

        The method is article-number agnostic and activates only when the
        recognized issue explicitly permits more than one final article.
        """

        if (
            final_limit <= 1
            or len(ranked) <= 1
            or not analysis.article_anchor_phrases
        ):
            return ranked[:final_limit]

        normalized_anchors = tuple(
            normalize_arabic(anchor)
            for anchor in analysis.article_anchor_phrases
            if normalize_arabic(anchor)
        )

        if not normalized_anchors:
            return ranked[:final_limit]

        selected: list[RetrievalHit] = [ranked[0]]
        selected_uris = {ranked[0].uri}

        covered_anchors = {
            anchor
            for anchor in normalized_anchors
            if anchor in self._article_normalized_text(ranked[0])
        }

        while (
            len(selected) < final_limit
            and len(selected_uris) < len(ranked)
        ):
            uncovered_anchors = [
                anchor
                for anchor in normalized_anchors
                if anchor not in covered_anchors
            ]

            best_hit: RetrievalHit | None = None
            best_key: tuple[int, float, float, float, int] | None = None

            for hit in ranked:
                if hit.uri in selected_uris:
                    continue

                normalized_text = self._article_normalized_text(hit)

                if uncovered_anchors:
                    new_coverage = (
                        sum(
                            anchor in normalized_text
                            for anchor in uncovered_anchors
                        )
                        / len(uncovered_anchors)
                    )
                else:
                    new_coverage = 0.0

                complementary_score = (
                    hit.final_score
                    + MULTI_ARTICLE_COVERAGE_BONUS
                    * new_coverage
                )

                key = (
                    int(new_coverage > 0.0),
                    new_coverage,
                    complementary_score,
                    hit.final_score,
                    -(
                        hit.article_number
                        if hit.article_number is not None
                        else 999999
                    ),
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_hit = hit

            if best_hit is None:
                break

            selected.append(best_hit)
            selected_uris.add(best_hit.uri)

            best_text = self._article_normalized_text(best_hit)
            covered_anchors.update(
                anchor
                for anchor in normalized_anchors
                if anchor in best_text
            )

        return selected

    def _llm_article_selection(
        self,
        *,
        ranked: list[RetrievalHit],
        article_hits: dict[str, RetrievalHit],
        expansion: GraphExpansionResult,
        analysis: LegalQuestionAnalysis,
        issue_candidate_numbers: tuple[tuple[int, ...], ...] = (),
    ) -> list[RetrievalHit] | None:
        """Choose a complete article set from an issue-balanced catalogue.

        Multi-issue questions reserve candidate slots for every atomic issue
        before the remaining globally ranked articles are added. The reranker
        then returns explicit issue coverage. If an issue is still uncovered,
        the best independently retrieved candidate for that issue is used as a
        deterministic coverage repair.
        """

        issue_pairs = tuple(
            (label, query)
            for label, query in zip(
                analysis.planner_issue_labels,
                analysis.planner_queries,
                strict=False,
            )
            if query
        )

        base_limit = max(
            5,
            int(getattr(self.settings, "reranker_candidate_limit", 12)),
        )
        candidate_limit = base_limit
        if len(issue_pairs) > 1:
            candidate_limit = max(
                base_limit,
                int(
                    getattr(
                        self.settings,
                        "reranker_multi_issue_candidate_limit",
                        15,
                    )
                ),
            )
        candidate_limit = min(30, candidate_limit)

        ranked_by_number = {
            int(hit.article_number): hit
            for hit in ranked
            if hit.article_number is not None
        }

        candidate_hits: list[RetrievalHit] = []
        seen_numbers: set[int] = set()

        # First interleave independently retrieved issue candidates so every
        # atomic issue gets representation in the reranker catalogue.
        max_issue_candidates = max(
            (len(values) for values in issue_candidate_numbers),
            default=0,
        )
        for candidate_rank in range(max_issue_candidates):
            for values in issue_candidate_numbers:
                if candidate_rank >= len(values):
                    continue
                number = int(values[candidate_rank])
                hit = ranked_by_number.get(number)
                if hit is None or number in seen_numbers:
                    continue
                seen_numbers.add(number)
                candidate_hits.append(hit)
                if len(candidate_hits) >= candidate_limit:
                    break
            if len(candidate_hits) >= candidate_limit:
                break

        # Fill the remaining slots with the strongest global ranking.
        if len(candidate_hits) < candidate_limit:
            for hit in ranked:
                if hit.article_number is None:
                    continue
                number = int(hit.article_number)
                if number in seen_numbers:
                    continue
                seen_numbers.add(number)
                candidate_hits.append(hit)
                if len(candidate_hits) >= candidate_limit:
                    break

        if not candidate_hits:
            return None

        global_rank_by_number = {
            int(hit.article_number): rank
            for rank, hit in enumerate(ranked, start=1)
            if hit.article_number is not None
        }

        catalog_articles: list[CatalogArticle] = []
        for hit in candidate_hits:
            number = int(hit.article_number)
            catalog_articles.append(
                CatalogArticle(
                    article_number=number,
                    text=" ".join(
                        [
                            *hit.labels_ar,
                            *hit.labels_en,
                            hit.text_preview,
                        ]
                    ),
                    deterministic_rank=global_rank_by_number.get(number),
                    deterministic_score=hit.final_score,
                    graph_supported=hit.graph_supported,
                )
            )

        selection = self.article_reranker.select(
            question=analysis.reranker_question,
            articles=catalog_articles,
            issue_pairs=issue_pairs,
        )

        if selection is None:
            return None

        if selection.behavior != "retrieve":
            if selection.confidence >= 0.90:
                return []
            return None

        max_articles = min(
            FINAL_ARTICLE_LIMIT,
            self.settings.retrieval_article_top_k,
        )

        selected_numbers = list(
            dict.fromkeys(
                int(number)
                for number in selection.selected_article_numbers
            )
        )

        coverage_by_index = {
            item.issue_index: item
            for item in selection.issue_coverage
        }

        # For multi-issue questions, form a required coverage set first. A
        # model-declared supporting article is preferred; otherwise use the
        # strongest independently retrieved candidate for that issue.
        required_numbers: list[int] = []
        coverage_debug: list[dict[str, Any]] = []

        if issue_pairs:
            for issue_index, (label, query) in enumerate(
                issue_pairs,
                start=1,
            ):
                coverage = coverage_by_index.get(issue_index)
                support_number: int | None = None
                source = "uncovered"

                if (
                    coverage is not None
                    and coverage.covered
                    and coverage.supporting_article_numbers
                ):
                    support_number = int(
                        coverage.supporting_article_numbers[0]
                    )
                    source = "reranker_coverage"

                if support_number is None:
                    values = (
                        issue_candidate_numbers[issue_index - 1]
                        if issue_index - 1 < len(issue_candidate_numbers)
                        else ()
                    )
                    repair_catalog: list[CatalogArticle] = []
                    for candidate_number in values:
                        hit = ranked_by_number.get(int(candidate_number))
                        if hit is None:
                            continue
                        repair_catalog.append(
                            CatalogArticle(
                                article_number=int(candidate_number),
                                text=" ".join(
                                    [
                                        *hit.labels_ar,
                                        *hit.labels_en,
                                        hit.text_preview,
                                    ]
                                ),
                                deterministic_rank=None,
                                deterministic_score=hit.final_score,
                                graph_supported=hit.graph_supported,
                            )
                        )

                    # A missing issue is never "repaired" merely by taking
                    # the first retrieved article. The same constrained LLM
                    # must verify a candidate against the atomic issue text.
                    if repair_catalog:
                        repair_selection = self.article_reranker.select(
                            question=_focused_issue_text(label, query),
                            articles=repair_catalog,
                            issue_pairs=(),
                        )
                        if (
                            repair_selection is not None
                            and repair_selection.behavior == "retrieve"
                            and repair_selection.selected_article_numbers
                        ):
                            candidate_number = int(
                                repair_selection.selected_article_numbers[0]
                            )
                            if candidate_number in {
                                item.article_number for item in repair_catalog
                            }:
                                support_number = candidate_number
                                source = "targeted_reranker_repair"

                if (
                    support_number is not None
                    and support_number not in required_numbers
                ):
                    required_numbers.append(support_number)

                coverage_debug.append(
                    {
                        "issue_index": issue_index,
                        "issue_ar": label,
                        "retrieval_query_ar": query,
                        "support_article": support_number,
                        "source": source,
                    }
                )

        final_numbers: list[int] = []
        for number in [*required_numbers, *selected_numbers]:
            if number not in final_numbers:
                final_numbers.append(number)
            if len(final_numbers) >= max_articles:
                break

        if not final_numbers:
            return None

        candidate_by_number = {
            int(hit.article_number): hit
            for hit in candidate_hits
            if hit.article_number is not None
        }

        selected_hits: list[RetrievalHit] = []
        for index, number in enumerate(final_numbers):
            hit = candidate_by_number.get(int(number))
            if hit is None:
                hit = ranked_by_number.get(int(number))
            if hit is None:
                raise RuntimeError(
                    f"Selected article {number} is outside the ranked candidate map."
                )

            evidence = expansion.article_evidence.get(hit.uri)
            if evidence is not None:
                hit.graph_score = float(evidence["score"])
                hit.graph_supported = True
                hit.support_paths = list(evidence["paths"])

            hit.final_score = max(
                hit.final_score,
                1.0 - 0.01 * index,
            )
            selected_hits.append(hit)

        self.last_issue_coverage_debug = {
            "issue_pairs": [
                {"issue_ar": label, "retrieval_query_ar": query}
                for label, query in issue_pairs
            ],
            "issue_candidate_numbers": [
                list(values)
                for values in issue_candidate_numbers
            ],
            "reranker_selected_numbers": selected_numbers,
            "coverage": coverage_debug,
            "final_numbers": final_numbers,
        }

        return selected_hits

    def _rank_articles(
        self,
        direct_hits: list[RetrievalHit],
        paragraph_hits: list[RetrievalHit],
        expansion: GraphExpansionResult,
        analysis: LegalQuestionAnalysis,
        *,
        allow_llm_selection: bool = True,
        final_limit_override: int | None = None,
        issue_candidate_numbers: tuple[tuple[int, ...], ...] = (),
    ) -> list[RetrievalHit]:
        """
        Rank the union of graph, direct-article, and paragraph evidence.

        Stage 7.5-A intentionally preserves direct candidates even when the
        graph has evidence.  Graph support is a strong feature, not a hard
        gate, because an incorrect graph seed must not remove a directly
        matching legal article from the result set.
        """

        article_hits: dict[
            str,
            RetrievalHit,
        ] = {
            hit.uri: hit
            for hit in direct_hits
        }

        direct_ranks = {
            hit.uri: rank
            for rank, hit in enumerate(
                direct_hits,
                start=1,
            )
        }

        paragraph_parent_map = (
            self.graph.find_parent_articles(
                [
                    hit.uri
                    for hit in paragraph_hits
                ]
            )
        )

        paragraph_scores: dict[
            str,
            float,
        ] = {}

        for rank, paragraph in enumerate(
            paragraph_hits,
            start=1,
        ):
            parent_article_uri = (
                paragraph_parent_map.get(
                    paragraph.uri
                )
            )

            if parent_article_uri is None:
                continue

            contribution = (
                1.0 / (10 + rank)
            )

            paragraph_scores[
                parent_article_uri
            ] = (
                paragraph_scores.get(
                    parent_article_uri,
                    0.0,
                )
                + contribution
            )

        paragraph_ranks = {
            uri: rank
            for rank, (uri, _) in enumerate(
                sorted(
                    paragraph_scores.items(),
                    key=lambda item: (
                        -item[1],
                        item[0],
                    ),
                ),
                start=1,
            )
        }

        graph_ranks = {
            uri: rank
            for rank, (uri, evidence) in enumerate(
                sorted(
                    expansion.article_evidence.items(),
                    key=lambda item: (
                        -float(
                            item[1]["score"]
                        ),
                        item[0],
                    ),
                ),
                start=1,
            )
        }

        all_article_uris = set(
            article_hits
        )

        all_article_uris.update(
            paragraph_scores
        )

        all_article_uris.update(
            expansion.article_evidence
        )

        missing_article_uris = [
            uri
            for uri in all_article_uris
            if uri not in article_hits
        ]

        article_hits.update(
            self._fetch_nodes_by_uris(
                missing_article_uris
            )
        )

        question_scores: dict[str, float] = {}

        for uri, hit in article_hits.items():
            evidence = (
                expansion
                .article_evidence
                .get(uri)
            )

            if evidence is not None:
                hit.graph_score = float(
                    evidence["score"]
                )

                hit.graph_supported = True

                hit.support_paths = list(
                    evidence["paths"]
                )

            hit.paragraph_score = (
                paragraph_scores.get(
                    uri,
                    0.0,
                )
            )

            direct_component = self._rank_score(
                direct_ranks.get(uri)
            )

            graph_component = self._rank_score(
                graph_ranks.get(uri)
            )

            paragraph_component = self._rank_score(
                paragraph_ranks.get(uri)
            )

            question_component = (
                article_question_relevance(
                    analysis,
                    " ".join(
                        [
                            *hit.labels_ar,
                            *hit.labels_en,
                            hit.text_preview,
                        ]
                    ),
                )
            )

            question_scores[uri] = question_component

            evidence_source_count = sum(
                component > 0.0
                for component in (
                    direct_component,
                    graph_component,
                    paragraph_component,
                )
            )

            agreement_bonus = (
                SOURCE_AGREEMENT_BONUS
                * max(
                    0,
                    evidence_source_count - 1,
                )
            )

            hit.final_score = (
                DIRECT_ARTICLE_WEIGHT
                * direct_component
                + GRAPH_ARTICLE_WEIGHT
                * graph_component
                + PARAGRAPH_ARTICLE_WEIGHT
                * paragraph_component
                + QUESTION_ARTICLE_WEIGHT
                * question_component
                + agreement_bonus
            )

        # A clear full-text winner receives a bounded confidence bonus.
        # The margin requirement prevents ambiguous issues (for example,
        # closely related leave provisions) from being forced to the top.
        ordered_question_scores = sorted(
            question_scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        if ordered_question_scores:
            top_uri, top_score = ordered_question_scores[0]
            second_score = (
                ordered_question_scores[1][1]
                if len(ordered_question_scores) > 1
                else 0.0
            )

            if (
                top_score >= 0.25
                and top_score - second_score >= ISSUE_TOP_MARGIN
                and top_uri in article_hits
            ):
                article_hits[top_uri].final_score += ISSUE_TOP_BONUS

        ranked = sorted(
            article_hits.values(),
            key=lambda hit: (
                -hit.final_score,
                -self._rank_score(
                    direct_ranks.get(hit.uri)
                ),
                -self._rank_score(
                    graph_ranks.get(hit.uri)
                ),
                -self._rank_score(
                    paragraph_ranks.get(hit.uri)
                ),
                (
                    hit.article_number
                    if hit.article_number
                    is not None
                    else 999999
                ),
            ),
        )

        if allow_llm_selection:
            llm_selected = self._llm_article_selection(
                ranked=ranked,
                article_hits=article_hits,
                expansion=expansion,
                analysis=analysis,
                issue_candidate_numbers=issue_candidate_numbers,
            )

            if llm_selected is not None:
                return llm_selected

        requested_limit = max(
            1,
            analysis.max_final_articles,
            len(analysis.planner_queries),
        )

        if final_limit_override is not None:
            final_limit = max(1, int(final_limit_override))
        else:
            final_limit = min(
                FINAL_ARTICLE_LIMIT,
                requested_limit,
                self.settings.retrieval_article_top_k,
            )

        return self._select_complementary_articles(
            ranked=ranked,
            analysis=analysis,
            final_limit=final_limit,
        )

    def preview(
        self,
        question: str,
        query_vector: list[float],
        *,
        embedding_model: str,
        embedding_dimensions: int,
        embedding_input_tokens: int,
        analysis: LegalQuestionAnalysis | None = None,
        issue_query_vectors: tuple[
            tuple[str, str, list[float]], ...
        ] = (),
    ) -> RetrievalPreview:
        # Stage 8-A: callers may provide the already-routed analysis so the
        # planner runs exactly once. Existing callers that omit it preserve
        # the previous retrieval behavior.
        if analysis is None:
            analysis = analyze_legal_question(
                question
            )
            query_plan = self.query_planner.plan(question)
            analysis = self.query_planner.merge_with_analysis(
                analysis=analysis,
                plan=query_plan,
            )

        self.last_issue_coverage_debug = {}

        if analysis.behavior != "retrieve":
            return RetrievalPreview(
                question=question,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                embedding_input_tokens=embedding_input_tokens,
                concept_hits=[],
                expanded_concept_hits=[],
                article_hits=[],
                raw_mixed_hits=[],
            )

        # -------------------------------------------------------------
        # Global retrieval path (preserves the original V1 behavior).
        # -------------------------------------------------------------
        global_concepts = self.search_concepts(
            question,
            query_vector,
            analysis,
        )
        global_direct_hits = self.search_articles_direct(
            question,
            query_vector,
            analysis,
        )
        global_paragraph_hits = self.search_paragraphs(
            question,
            query_vector,
            analysis,
        )

        concept_groups: list[list[RetrievalHit]] = [global_concepts]
        direct_groups: list[list[RetrievalHit]] = [global_direct_hits]
        paragraph_groups: list[list[RetrievalHit]] = [global_paragraph_hits]
        expansion_groups: list[GraphExpansionResult] = []
        issue_candidate_numbers: list[tuple[int, ...]] = []
        issue_analyses: list[LegalQuestionAnalysis] = []

        # -------------------------------------------------------------
        # V2: independently retrieve evidence for every atomic issue.
        # Each issue gets its own embedding, BM25 query, concept linking,
        # KG traversal, paragraph search, and deterministic article ranking.
        # -------------------------------------------------------------
        per_issue_limit = int(
            getattr(
                self.settings,
                "retrieval_issue_candidates_per_issue",
                3,
            )
        )

        for _label, issue_query, issue_vector in issue_query_vectors:
            issue_focus = _focused_issue_text(_label, issue_query)
            issue_analysis = _atomic_issue_analysis(_label, issue_query)
            issue_analyses.append(issue_analysis)

            issue_concepts = self.search_concepts(
                issue_focus,
                issue_vector,
                issue_analysis,
            )
            issue_expansion = self.graph.expand_from_concepts(
                issue_concepts,
                preferred_local_names=issue_analysis.preferred_concepts,
            )
            issue_direct_hits = self.search_articles_direct(
                issue_focus,
                issue_vector,
                issue_analysis,
            )
            issue_paragraph_hits = self.search_paragraphs(
                issue_focus,
                issue_vector,
                issue_analysis,
            )

            issue_ranked = self._rank_articles(
                direct_hits=issue_direct_hits,
                paragraph_hits=issue_paragraph_hits,
                expansion=issue_expansion,
                analysis=issue_analysis,
                allow_llm_selection=False,
                final_limit_override=per_issue_limit,
            )

            # Keep the strongest bounded candidates for the issue and let the
            # constrained reranker verify legal support from article text. Do
            # not use a weak lexical threshold as a proof of coverage.
            strong_issue_numbers = [
                int(hit.article_number)
                for hit in issue_ranked
                if hit.article_number is not None
            ][:per_issue_limit]

            issue_candidate_numbers.append(
                tuple(strong_issue_numbers)
            )
            concept_groups.append(issue_concepts)
            direct_groups.append(issue_direct_hits)
            paragraph_groups.append(issue_paragraph_hits)
            expansion_groups.append(issue_expansion)

        # Interleave groups so no one issue dominates the shared ranking.
        concept_limit = max(
            self.settings.retrieval_concept_top_k,
            len(issue_query_vectors) * 3,
        )
        concept_hits = self._merge_hit_groups(
            concept_groups,
            limit=concept_limit,
        )

        # Run one combined graph expansion as well. For genuine multi-issue
        # questions the seed ceiling grows with the issue count, while the
        # original two-seed behavior remains unchanged for simple questions.
        combined_preferred = list(analysis.preferred_concepts)
        for issue_analysis in issue_analyses:
            for concept in issue_analysis.preferred_concepts:
                if concept not in combined_preferred:
                    combined_preferred.append(concept)

        seed_limit_override: int | None = None
        if len(issue_query_vectors) > 1:
            seed_limit_override = min(
                int(
                    getattr(
                        self.settings,
                        "retrieval_issue_graph_seed_limit",
                        5,
                    )
                ),
                max(2, len(issue_query_vectors)),
            )

        combined_expansion = self.graph.expand_from_concepts(
            concept_hits,
            preferred_local_names=tuple(combined_preferred),
            seed_limit_override=seed_limit_override,
        )
        expansion_groups.insert(0, combined_expansion)
        expansion = self._merge_expansions(expansion_groups)

        expanded_concepts = self._expanded_concepts(
            expansion
        )

        direct_article_hits = self._merge_hit_groups(
            direct_groups,
            limit=max(80, len(issue_query_vectors) * 20),
        )
        paragraph_hits = self._merge_hit_groups(
            paragraph_groups,
            limit=max(60, len(issue_query_vectors) * 20),
        )

        article_hits = self._rank_articles(
            direct_hits=direct_article_hits,
            paragraph_hits=paragraph_hits,
            expansion=expansion,
            analysis=analysis,
            issue_candidate_numbers=tuple(issue_candidate_numbers),
        )

        mixed_hits = self.search_mixed(
            question,
            query_vector,
        )

        if issue_query_vectors:
            self.last_issue_coverage_debug.setdefault(
                "issue_query_vectors",
                [
                    {
                        "issue_ar": label,
                        "retrieval_query_ar": query,
                        "focused_retrieval_text": _focused_issue_text(label, query),
                        "embedding_dimensions": len(vector),
                    }
                    for label, query, vector in issue_query_vectors
                ],
            )

        return RetrievalPreview(
            question=question,
            embedding_model=embedding_model,
            embedding_dimensions=(
                embedding_dimensions
            ),
            embedding_input_tokens=(
                embedding_input_tokens
            ),
            concept_hits=concept_hits,
            expanded_concept_hits=(
                expanded_concepts
            ),
            article_hits=article_hits,
            raw_mixed_hits=mixed_hits[:20],
        )
