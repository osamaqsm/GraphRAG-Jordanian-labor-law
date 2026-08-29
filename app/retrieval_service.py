from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

from weaviate.classes.query import Filter, MetadataQuery

from app.config import Settings
from app.graph_retrieval import GraphExpansionResult, GraphTraversalService
from app.legal_article_reranker import CatalogArticle, LegalArticleReranker
from app.legal_query_planner import AtomicLegalIssue, LegalQueryPlan, LegalQueryPlanner
from app.legal_question_analysis import article_question_relevance
from app.retrieval_models import RetrievalHit, RetrievalPreview


RETURN_PROPERTIES = [
    "uri",
    "localName",
    "nodeKind",
    "labelsAr",
    "labelsEn",
    "aliasesAr",
    "aliasesEn",
    "commentsAr",
    "commentsEn",
    "articleNumber",
    "retrievalEligible",
    "searchableText",
]


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _first_text(properties: dict[str, Any]) -> str:
    arabic = _list_value(properties.get("commentsAr"))
    if arabic:
        return arabic[0][:5000]
    english = _list_value(properties.get("commentsEn"))
    if english:
        return english[0][:5000]
    return str(properties.get("searchableText", "") or "")[:5000]


def _make_hit(properties: dict[str, Any]) -> RetrievalHit:
    article_number_value = properties.get("articleNumber")
    return RetrievalHit(
        uri=str(properties.get("uri", "") or ""),
        local_name=str(properties.get("localName", "") or ""),
        node_kind=str(properties.get("nodeKind", "") or ""),
        labels_ar=_list_value(properties.get("labelsAr")),
        labels_en=_list_value(properties.get("labelsEn")),
        aliases_ar=_list_value(properties.get("aliasesAr")),
        aliases_en=_list_value(properties.get("aliasesEn")),
        article_number=(int(article_number_value) if article_number_value is not None else None),
        text_preview=_first_text(properties),
    )


class RetrievalService:
    """Final graph-only legal-evidence retrieval service.

    Vector/BM25 search is restricted to retrievalEligible semantic concept
    individuals. Article candidates are created exclusively from KG traversal.
    """

    def __init__(self, client: Any, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.nodes = client.collections.use(settings.weaviate_node_collection)
        self.graph = GraphTraversalService(client=client, settings=settings)
        self.query_planner = LegalQueryPlanner(settings=settings)
        self.article_reranker = LegalArticleReranker(settings=settings)

    @staticmethod
    def _concept_filter() -> Any:
        return Filter.by_property("retrievalEligible").equal(True)

    def _vector_search(self, vector: list[float], *, limit: int) -> list[Any]:
        response = self.nodes.query.near_vector(
            near_vector=vector,
            target_vector="default",
            filters=self._concept_filter(),
            limit=limit,
            return_properties=RETURN_PROPERTIES,
            return_metadata=MetadataQuery(distance=True),
        )
        return list(response.objects)

    def _bm25_search(self, query: str, *, limit: int) -> list[Any]:
        response = self.nodes.query.bm25(
            query=query,
            query_properties=[
                "labelsAr^6",
                "aliasesAr^5",
                "labelsEn^2",
                "aliasesEn^2",
                "searchableText",
            ],
            filters=self._concept_filter(),
            limit=limit,
            return_properties=RETURN_PROPERTIES,
            return_metadata=MetadataQuery(score=True),
        )
        return list(response.objects)

    def _hint_search(self, hints: Iterable[str], *, limit: int) -> list[Any]:
        groups: list[list[Any]] = []
        per_hint = max(4, min(limit, 10))
        for hint in hints:
            hint = str(hint).strip()
            if not hint:
                continue
            values = self._bm25_search(hint, limit=per_hint)
            if values:
                groups.append(values)

        interleaved: list[Any] = []
        seen: set[str] = set()
        max_size = max((len(group) for group in groups), default=0)
        for rank_index in range(max_size):
            for group in groups:
                if rank_index >= len(group):
                    continue
                item = group[rank_index]
                uri = str(dict(item.properties).get("uri", ""))
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                interleaved.append(item)
        return interleaved[:limit]

    def _fuse_concepts(
        self,
        vector_objects: Iterable[Any],
        bm25_objects: Iterable[Any],
        hint_objects: Iterable[Any],
    ) -> list[RetrievalHit]:
        hits: dict[str, RetrievalHit] = {}
        rrf_k = self.settings.concept_rrf_k

        for rank, item in enumerate(vector_objects, start=1):
            properties = dict(item.properties)
            uri = str(properties.get("uri", ""))
            if not uri:
                continue
            hit = hits.setdefault(uri, _make_hit(properties))
            hit.vector_rank = rank
            distance = getattr(item.metadata, "distance", None)
            if distance is not None:
                hit.vector_distance = float(distance)
            hit.fused_score += 1.0 / (rrf_k + rank)

        for rank, item in enumerate(bm25_objects, start=1):
            properties = dict(item.properties)
            uri = str(properties.get("uri", ""))
            if not uri:
                continue
            hit = hits.setdefault(uri, _make_hit(properties))
            hit.bm25_rank = rank
            score = getattr(item.metadata, "score", None)
            if score is not None:
                hit.bm25_score = float(score)
            hit.fused_score += 1.0 / (rrf_k + rank)

        for rank, item in enumerate(hint_objects, start=1):
            properties = dict(item.properties)
            uri = str(properties.get("uri", ""))
            if not uri:
                continue
            hit = hits.setdefault(uri, _make_hit(properties))
            hit.hint_rank = rank
            score = getattr(item.metadata, "score", None)
            if score is not None:
                hit.hint_score = float(score)
            hit.fused_score += 1.5 / (rrf_k + rank)

        degrees = self.graph.semantic_degrees(hits)

        for hit in hits.values():
            if hit.vector_rank is not None:
                rank_component = 1.0 / math.sqrt(hit.vector_rank)
                distance_component = max(
                    0.0,
                    min(1.0, 1.0 - float(hit.vector_distance or 0.0)),
                )
                hit.semantic_score = 0.5 * rank_component + 0.5 * distance_component

            if hit.bm25_rank is not None:
                hit.lexical_score = 1.0 / math.sqrt(hit.bm25_rank)

            if hit.hint_rank is not None:
                hit.planner_hint_score = 1.0 / math.sqrt(hit.hint_rank)

            degree = degrees.get(hit.uri, 0)
            hit.specificity_score = 1.0 / (1.0 + math.log1p(degree))

            components: list[tuple[float, float]] = [
                (self.settings.concept_specificity_weight, hit.specificity_score),
            ]
            if hit.vector_rank is not None:
                components.append((self.settings.concept_semantic_weight, hit.semantic_score))
            if hit.bm25_rank is not None:
                components.append((self.settings.concept_lexical_weight, hit.lexical_score))
            if hit.hint_rank is not None:
                components.append((self.settings.concept_hint_weight, hit.planner_hint_score))

            weight_sum = sum(weight for weight, _ in components) or 1.0
            hit.seed_score = sum(weight * value for weight, value in components) / weight_sum
            hit.final_score = hit.seed_score

        return sorted(
            hits.values(),
            key=lambda hit: (-hit.seed_score, -hit.fused_score, hit.uri),
        )[: self.settings.concept_top_k]

    def search_concepts(
        self,
        issue: AtomicLegalIssue,
        issue_vector: list[float],
    ) -> list[RetrievalHit]:
        vector_objects = self._vector_search(
            issue_vector,
            limit=self.settings.concept_vector_candidates,
        )
        bm25_objects = self._bm25_search(
            issue.retrieval_query_ar,
            limit=self.settings.concept_bm25_candidates,
        )
        hint_objects = self._hint_search(
            issue.concept_hints_ar,
            limit=self.settings.concept_bm25_candidates,
        )
        return self._fuse_concepts(vector_objects, bm25_objects, hint_objects)

    def _fetch_nodes_by_uris(self, uris: Iterable[str]) -> dict[str, RetrievalHit]:
        unique = sorted({str(uri) for uri in uris if uri})
        if not unique:
            return {}
        response = self.nodes.query.fetch_objects(
            filters=Filter.by_property("uri").contains_any(unique),
            limit=len(unique),
            return_properties=RETURN_PROPERTIES,
        )
        result: dict[str, RetrievalHit] = {}
        for item in response.objects:
            hit = _make_hit(dict(item.properties))
            if hit.uri:
                result[hit.uri] = hit
        return result

    def _rank_graph_articles(
        self,
        *,
        issue: AtomicLegalIssue,
        expansion: GraphExpansionResult,
    ) -> list[RetrievalHit]:
        node_map = self._fetch_nodes_by_uris(expansion.article_evidence)
        if not node_map:
            return []

        max_graph = max(
            (float(value["score"]) for value in expansion.article_evidence.values()),
            default=1.0,
        ) or 1.0
        issue_text = " ".join(
            [issue.issue_ar, issue.retrieval_query_ar, *issue.concept_hints_ar]
        )

        ranked: list[RetrievalHit] = []
        for uri, evidence in expansion.article_evidence.items():
            hit = node_map.get(uri)
            if hit is None or hit.node_kind != "Article" or hit.article_number is None:
                continue
            hit.graph_score = float(evidence["score"])
            hit.graph_supported = True
            hit.support_paths = list(evidence["paths"])
            hit.issue_relevance = article_question_relevance(
                issue_text,
                " ".join([*hit.labels_ar, *hit.labels_en, hit.text_preview]),
            )
            normalized_graph = hit.graph_score / max_graph
            path_bonus = 0.03 * min(3, max(0, len(hit.support_paths) - 1))
            hit.final_score = min(
                1.25,
                self.settings.article_graph_weight * normalized_graph
                + self.settings.article_issue_relevance_weight * hit.issue_relevance
                + path_bonus,
            )
            ranked.append(hit)

        return sorted(
            ranked,
            key=lambda hit: (-hit.final_score, -hit.graph_score, hit.article_number or 999999),
        )

    def _expanded_concepts(self, expansions: list[GraphExpansionResult]) -> list[RetrievalHit]:
        scores: dict[str, float] = {}
        for expansion in expansions:
            for uri, score in expansion.related_concepts.items():
                scores[uri] = max(scores.get(uri, 0.0), float(score))
        node_map = self._fetch_nodes_by_uris(scores)
        result: list[RetrievalHit] = []
        for uri, score in scores.items():
            hit = node_map.get(uri)
            if hit is None or hit.node_kind in {
                "Article", "Paragraph", "Definition", "Law", "Ontology",
                "OntologyClass", "ObjectProperty", "DatatypeProperty",
                "AnnotationProperty", "Resource",
            }:
                continue
            hit.graph_score = score
            hit.final_score = score
            hit.graph_supported = True
            result.append(hit)
        return sorted(result, key=lambda hit: (-hit.graph_score, hit.uri))[: self.settings.concept_top_k]

    @staticmethod
    def _merge_concepts(groups: list[list[RetrievalHit]], limit: int) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for group in groups:
            for source in group:
                existing = merged.get(source.uri)
                if existing is None:
                    merged[source.uri] = replace(source)
                elif source.seed_score > existing.seed_score:
                    merged[source.uri] = replace(source)
        return sorted(
            merged.values(),
            key=lambda hit: (-hit.seed_score, hit.uri),
        )[:limit]

    @staticmethod
    def _merge_article_candidates(
        issue_rankings: list[list[RetrievalHit]],
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        issue_coverage_count: dict[str, int] = {}

        for ranking in issue_rankings:
            seen_issue: set[str] = set()
            for source in ranking:
                if source.uri not in seen_issue:
                    issue_coverage_count[source.uri] = issue_coverage_count.get(source.uri, 0) + 1
                    seen_issue.add(source.uri)

                existing = merged.get(source.uri)
                if existing is None:
                    clone = replace(source)
                    clone.labels_ar = list(source.labels_ar)
                    clone.labels_en = list(source.labels_en)
                    clone.aliases_ar = list(source.aliases_ar)
                    clone.aliases_en = list(source.aliases_en)
                    clone.support_paths = list(source.support_paths)
                    merged[source.uri] = clone
                else:
                    existing.graph_score = max(existing.graph_score, source.graph_score)
                    existing.issue_relevance = max(existing.issue_relevance, source.issue_relevance)
                    existing.final_score = max(existing.final_score, source.final_score)
                    known = {
                        (p.get("seed_uri"), p.get("article_uri"), tuple(
                            (s.get("source_uri"), s.get("relation"), s.get("target_uri"))
                            for s in p.get("steps", [])
                        ))
                        for p in existing.support_paths
                    }
                    for path in source.support_paths:
                        key = (
                            path.get("seed_uri"),
                            path.get("article_uri"),
                            tuple(
                                (s.get("source_uri"), s.get("relation"), s.get("target_uri"))
                                for s in path.get("steps", [])
                            ),
                        )
                        if key not in known:
                            existing.support_paths.append(path)
                            known.add(key)

        for uri, hit in merged.items():
            hit.final_score += 0.05 * max(0, issue_coverage_count.get(uri, 1) - 1)

        return sorted(
            merged.values(),
            key=lambda hit: (-hit.final_score, -hit.graph_score, hit.article_number or 999999),
        )

    def retrieve_plan(
        self,
        *,
        question: str,
        plan: LegalQueryPlan,
        issue_vectors: tuple[list[float], ...],
        embedding_model: str,
        embedding_dimensions: int,
        embedding_input_tokens: int,
    ) -> RetrievalPreview:
        if plan.decision != "retrieve":
            return RetrievalPreview(
                question=question,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                embedding_input_tokens=embedding_input_tokens,
                concept_hits=[],
                expanded_concept_hits=[],
                article_candidates=[],
                article_hits=[],
            )

        if len(issue_vectors) != len(plan.atomic_issues):
            raise ValueError("Issue-vector count does not match planner atomic issues.")

        concept_groups: list[list[RetrievalHit]] = []
        expansions: list[GraphExpansionResult] = []
        issue_rankings: list[list[RetrievalHit]] = []
        issue_debug: list[dict[str, Any]] = []
        selected_seeds: list[RetrievalHit] = []

        for index, (issue, vector) in enumerate(
            zip(plan.atomic_issues, issue_vectors, strict=True),
            start=1,
        ):
            concepts = self.search_concepts(issue, vector)
            expansion = self.graph.expand_from_concepts(concepts)
            ranking = self._rank_graph_articles(issue=issue, expansion=expansion)
            bounded = ranking[: self.settings.issue_article_candidates]

            concept_groups.append(concepts)
            expansions.append(expansion)
            issue_rankings.append(bounded)
            selected_seeds.extend(expansion.seeds)

            issue_debug.append(
                {
                    "issue_index": index,
                    "issue_ar": issue.issue_ar,
                    "retrieval_query_ar": issue.retrieval_query_ar,
                    "concept_hints_ar": list(issue.concept_hints_ar),
                    "selected_seed_local_names": [seed.local_name for seed in expansion.seeds],
                    "graph_candidate_article_numbers": [
                        hit.article_number for hit in bounded if hit.article_number is not None
                    ],
                }
            )

        candidates = self._merge_article_candidates(issue_rankings)
        candidates = candidates[: max(self.settings.reranker_candidate_limit, self.settings.article_top_k)]

        catalog = [
            CatalogArticle(
                article_number=int(hit.article_number),
                text=" ".join([*hit.labels_ar, *hit.labels_en, hit.text_preview]),
                graph_rank=rank,
                graph_score=hit.final_score,
            )
            for rank, hit in enumerate(candidates, start=1)
            if hit.article_number is not None
        ]

        selection = self.article_reranker.select(
            question=question,
            articles=catalog,
            issue_pairs=plan.issue_pairs,
        )
        candidate_by_number = {
            int(hit.article_number): hit
            for hit in candidates
            if hit.article_number is not None
        }
        selected_articles = [
            candidate_by_number[number]
            for number in selection.selected_article_numbers
            if number in candidate_by_number
        ][: self.settings.article_top_k]

        issue_debug.append(
            {
                "reranker_selected_article_numbers": list(selection.selected_article_numbers),
                "reranker_issue_coverage": [
                    item.model_dump(mode="json") for item in selection.issue_coverage
                ],
                "reranker_confidence": selection.confidence,
                "reranker_reason": selection.reason,
            }
        )

        concept_hits = self._merge_concepts(
            concept_groups,
            limit=max(self.settings.concept_top_k, len(plan.atomic_issues) * 4),
        )
        expanded = self._expanded_concepts(expansions)

        # De-duplicate selected seeds by URI, retaining the strongest score.
        seed_map: dict[str, RetrievalHit] = {}
        for seed in selected_seeds:
            if seed.uri not in seed_map or seed.seed_score > seed_map[seed.uri].seed_score:
                seed_map[seed.uri] = seed
        seeds = sorted(seed_map.values(), key=lambda hit: (-hit.seed_score, hit.uri))

        return RetrievalPreview(
            question=question,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_input_tokens=embedding_input_tokens,
            concept_hits=concept_hits,
            expanded_concept_hits=expanded,
            article_candidates=candidates,
            article_hits=selected_articles,
            selected_seeds=seeds,
            issue_debug=issue_debug,
        )
