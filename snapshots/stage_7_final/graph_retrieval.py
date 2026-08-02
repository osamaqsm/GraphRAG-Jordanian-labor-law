from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from weaviate.classes.query import Filter

from app.config import Settings
from app.retrieval_models import RetrievalHit


LAW_NAMESPACE = (
    "http://example.org/jordan-labor-law#"
)


ARTICLE_URI_PATTERN = re.compile(
    r"^http://example\.org/"
    r"jordan-labor-law#article_\d+$"
)


PARAGRAPH_URI_PATTERN = re.compile(
    r"^(http://example\.org/"
    r"jordan-labor-law#article_\d+)"
    r"_paragraph_\d+$"
)


# Generic actors are useful evidence, but they are poor
# initial graph seeds because they connect to many rules.
GENERIC_SEED_KINDS = {
    "Employer",
    "Worker",
    "Party",
    "LegalActor",
    "Person",
    "Organization",
}


def select_graph_seeds(
    concept_hits: list[RetrievalHit],
    limit: int,
    preferred_local_names: Iterable[str] | None = None,
) -> list[RetrievalHit]:
    """
    Select high-confidence, issue-specific concepts for graph traversal.

    Generic actors and consequence nodes are excluded from the initial
    frontier because they connect to many unrelated legal topics. Strong
    lexical evidence is preferred; semantic ranking is used only when no
    useful BM25 evidence is available.
    """

    if limit < 1:
        return []

    # Keep the initial graph frontier intentionally small. Broad frontiers
    # create unrelated paths even when every individual RDF edge is valid.
    effective_limit = min(limit, 2)

    preferred_order = {
        local_name: index
        for index, local_name in enumerate(
            preferred_local_names or (),
        )
    }

    candidates = [
        hit
        for hit in concept_hits
        if (
            hit.node_kind not in GENERIC_SEED_KINDS
            and (
                "consequence" not in hit.local_name.lower()
                or hit.local_name in preferred_order
            )
        )
    ]

    if not candidates:
        return []

    # Stage 7.5-B: when the controlled legal-question analyzer identifies
    # issue-specific ontology concepts, use only those concepts for the
    # initial graph frontier. This prevents an unrelated lexical concept
    # from overriding a clearly identified legal issue.
    preferred_candidates = [
        hit
        for hit in candidates
        if hit.local_name in preferred_order
    ]

    if preferred_candidates:
        preferred_candidates.sort(
            key=lambda hit: (
                preferred_order[hit.local_name],
                -float(hit.bm25_score or 0.0),
                -float(hit.fused_score),
                hit.uri,
            )
        )

        return preferred_candidates[:effective_limit]

    bm25_scores = [
        float(hit.bm25_score)
        for hit in candidates
        if hit.bm25_score is not None
    ]

    if bm25_scores:
        best_bm25_score = max(bm25_scores)

        # A strict threshold prevents generic lexical overlap, such as
        # "employer", from becoming a graph seed for an unrelated issue.
        minimum_bm25_score = best_bm25_score * 0.75

        strong_candidates = [
            hit
            for hit in candidates
            if (
                hit.bm25_score is not None
                and float(hit.bm25_score)
                >= minimum_bm25_score
            )
        ]

        strong_candidates.sort(
            key=lambda hit: (
                -float(hit.bm25_score or 0.0),
                -float(hit.fused_score),
                (
                    hit.vector_distance
                    if hit.vector_distance is not None
                    else float("inf")
                ),
                hit.uri,
            )
        )

        if strong_candidates:
            return strong_candidates[:effective_limit]

    # Semantic fallback for questions with little or no lexical overlap.
    # Only one fallback seed is used to avoid uncontrolled graph expansion.
    candidates.sort(
        key=lambda hit: (
            -float(hit.fused_score),
            (
                hit.vector_distance
                if hit.vector_distance is not None
                else float("inf")
            ),
            hit.uri,
        )
    )

    return candidates[:1]


# Only traverse meaningful domain relations.
# This avoids following rdf:type, labels and OWL schema edges.
RELATION_WEIGHTS: dict[str, float] = {
    # Direct evidence links.
    "supportedByArticle": 10.0,
    "regulatedBy": 8.0,
    "regulates": 5.0,

    # Core relations that describe a legal issue.
    "hasCondition": 3.0,
    "violatesRight": 3.0,
    "breachesObligation": 3.0,
    "resultsIn": 2.5,
    "basedOnEvent": 2.5,
}


EDGE_RETURN_PROPERTIES = [
    "sourceUri",
    "predicateUri",
    "predicateLocalName",
    "objectKind",
    "targetUri",
]


@dataclass(slots=True)
class GraphExpansionResult:
    """
    Articles and related concepts discovered by traversing
    the KG from the initial concept candidates.
    """

    article_evidence: dict[
        str,
        dict[str, Any],
    ]

    related_concepts: dict[
        str,
        float,
    ]


class GraphTraversalService:
    """
    Traverse URI-to-URI triples stored in the edge collection.
    """

    def __init__(
        self,
        client: Any,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.edges = client.collections.use(
            settings.weaviate_edge_collection
        )

    @staticmethod
    def _is_article_uri(
        uri: str,
    ) -> bool:
        return bool(
            ARTICLE_URI_PATTERN.match(uri)
        )

    def _query_edges(
        self,
        *,
        property_name: str,
        uris: list[str],
    ) -> list[dict[str, Any]]:
        """
        Retrieve URI edges where either sourceUri or targetUri
        matches one of the supplied URIs.
        """

        if not uris:
            return []

        filters = (
            Filter
            .by_property(property_name)
            .contains_any(uris)
            &
            Filter
            .by_property("objectKind")
            .equal("uri")
            &
            Filter
            .by_property("predicateLocalName")
            .contains_any(
                sorted(RELATION_WEIGHTS)
            )
        )

        response = self.edges.query.fetch_objects(
            filters=filters,
            limit=(
                self.settings
                .retrieval_graph_edges_limit
            ),
            return_properties=(
                EDGE_RETURN_PROPERTIES
            ),
        )

        return [
            dict(item.properties)
            for item in response.objects
        ]

    def outgoing_edges(
        self,
        uris: list[str],
    ) -> list[dict[str, Any]]:
        return self._query_edges(
            property_name="sourceUri",
            uris=uris,
        )

    def incoming_edges(
        self,
        uris: list[str],
    ) -> list[dict[str, Any]]:
        return self._query_edges(
            property_name="targetUri",
            uris=uris,
        )

    def find_parent_articles(
        self,
        paragraph_uris: list[str],
    ) -> dict[str, str]:
        """
        Map paragraph URI -> parent Article URI.

        Primary method:
            Article --hasParagraph--> Paragraph

        Fallback:
            Infer article URI from the paragraph URI format.
        """

        if not paragraph_uris:
            return {}

        result: dict[str, str] = {}

        response = self.edges.query.fetch_objects(
            filters=(
                Filter
                .by_property("targetUri")
                .contains_any(paragraph_uris)
                &
                Filter
                .by_property("predicateLocalName")
                .equal("hasParagraph")
                &
                Filter
                .by_property("objectKind")
                .equal("uri")
            ),
            limit=(
                self.settings
                .retrieval_graph_edges_limit
            ),
            return_properties=[
                "sourceUri",
                "targetUri",
            ],
        )

        for item in response.objects:
            properties = dict(
                item.properties
            )

            paragraph_uri = str(
                properties.get(
                    "targetUri",
                    "",
                )
            )

            article_uri = str(
                properties.get(
                    "sourceUri",
                    "",
                )
            )

            if paragraph_uri and article_uri:
                result[paragraph_uri] = (
                    article_uri
                )

        # Safe fallback when the explicit relationship
        # is missing from a future KG export.
        for paragraph_uri in paragraph_uris:
            if paragraph_uri in result:
                continue

            match = PARAGRAPH_URI_PATTERN.match(
                paragraph_uri
            )

            if match is not None:
                result[paragraph_uri] = (
                    match.group(1)
                )

        return result

    def expand_from_concepts(
        self,
        concept_hits: list[RetrievalHit],
        preferred_local_names: Iterable[str] | None = None,
    ) -> GraphExpansionResult:
        """
        Perform up to two hops from issue-specific concepts.

        Example:

        delay_exceeds_seven_days_condition
            <- hasCondition -
        wage_delay_violation
            - supportedByArticle ->
        article_46
        """

        issue_specific_seeds = select_graph_seeds(
            concept_hits=concept_hits,
            limit=(
                self.settings
                .retrieval_graph_seed_count
            ),
            preferred_local_names=(
                preferred_local_names
            ),
        )

        article_evidence: dict[
            str,
            dict[str, Any],
        ] = {}

        related_concepts: dict[
            str,
            float,
        ] = {}

        for seed_rank, seed in enumerate(
            issue_specific_seeds,
            start=1,
        ):
            seed_label = (
                seed.labels_ar[0]
                if seed.labels_ar
                else (
                    seed.labels_en[0]
                    if seed.labels_en
                    else seed.local_name
                )
            )

            # Each frontier item stores its best path
            # from the current seed.
            frontier: dict[
                str,
                dict[str, Any],
            ] = {
                seed.uri: {
                    "score": (
                        1.0 / seed_rank
                    ),
                    "steps": [],
                }
            }

            visited = {
                seed.uri
            }

            for _ in range(
                self.settings
                .retrieval_graph_max_hops
            ):
                if not frontier:
                    break

                frontier_uris = list(
                    frontier
                )

                outgoing = self.outgoing_edges(
                    frontier_uris
                )

                incoming = self.incoming_edges(
                    frontier_uris
                )

                next_frontier: dict[
                    str,
                    dict[str, Any],
                ] = {}

                directional_edges = [
                    ("outgoing", outgoing),
                    ("incoming", incoming),
                ]

                # Count unique article neighbors for each frontier node once
                # per hop.  This supports degree-normalized graph evidence
                # without repeatedly scanning all edges for every candidate.
                article_neighbors_by_current_uri: dict[
                    str,
                    set[str],
                ] = {}

                for candidate_direction, candidate_edges in (
                    directional_edges
                ):
                    for candidate in candidate_edges:
                        if candidate_direction == "outgoing":
                            candidate_current_uri = str(
                                candidate.get(
                                    "sourceUri",
                                    "",
                                )
                            )
                            candidate_neighbor_uri = str(
                                candidate.get(
                                    "targetUri",
                                    "",
                                )
                            )
                        else:
                            candidate_current_uri = str(
                                candidate.get(
                                    "targetUri",
                                    "",
                                )
                            )
                            candidate_neighbor_uri = str(
                                candidate.get(
                                    "sourceUri",
                                    "",
                                )
                            )

                        if (
                            candidate_current_uri
                            in frontier
                            and self._is_article_uri(
                                candidate_neighbor_uri
                            )
                        ):
                            article_neighbors_by_current_uri.setdefault(
                                candidate_current_uri,
                                set(),
                            ).add(
                                candidate_neighbor_uri
                            )

                for direction, edge_list in (
                    directional_edges
                ):
                    for edge in edge_list:
                        source_uri = str(
                            edge.get(
                                "sourceUri",
                                "",
                            )
                        )

                        target_uri = str(
                            edge.get(
                                "targetUri",
                                "",
                            )
                        )

                        relation = str(
                            edge.get(
                                "predicateLocalName",
                                "",
                            )
                        )

                        if (
                            not source_uri
                            or not target_uri
                            or relation
                            not in RELATION_WEIGHTS
                        ):
                            continue

                        if direction == "outgoing":
                            current_uri = source_uri
                            neighbor_uri = target_uri
                        else:
                            current_uri = target_uri
                            neighbor_uri = source_uri

                        current_state = (
                            frontier.get(
                                current_uri
                            )
                        )

                        if current_state is None:
                            continue

                        path_score = (
                            float(
                                current_state[
                                    "score"
                                ]
                            )
                            * RELATION_WEIGHTS[
                                relation
                            ]
                            * 0.65
                        )

                        # A broad concept can legitimately point to an entire
                        # chapter.  Penalize article evidence from high-degree
                        # frontier nodes so one generic concept does not make
                        # every connected article equally strong.
                        if self._is_article_uri(
                            neighbor_uri
                        ):
                            article_degree = max(
                                1,
                                len(
                                    article_neighbors_by_current_uri.get(
                                        current_uri,
                                        set(),
                                    )
                                ),
                            )

                            degree_penalty = math.log2(
                                2.0 + article_degree
                            )

                            path_score /= degree_penalty

                        step = {
                            "direction": direction,
                            "source_uri": source_uri,
                            "relation": relation,
                            "target_uri": target_uri,
                        }

                        path_steps = [
                            *current_state["steps"],
                            step,
                        ]

                        if self._is_article_uri(
                            neighbor_uri
                        ):
                            evidence = (
                                article_evidence
                                .setdefault(
                                    neighbor_uri,
                                    {
                                        "score": 0.0,
                                        "paths": [],
                                    },
                                )
                            )

                            evidence["score"] = max(
                                float(
                                    evidence["score"]
                                ),
                                path_score,
                            )

                            if (
                                len(evidence["paths"])
                                <
                                self.settings
                                .retrieval_graph_paths_per_article
                            ):
                                evidence[
                                    "paths"
                                ].append(
                                    {
                                        "seed_uri": (
                                            seed.uri
                                        ),
                                        "seed_label": (
                                            seed_label
                                        ),
                                        "article_uri": (
                                            neighbor_uri
                                        ),
                                        "score": (
                                            path_score
                                        ),
                                        "steps": (
                                            path_steps
                                        ),
                                    }
                                )

                            continue

                        if not neighbor_uri.startswith(
                            LAW_NAMESPACE
                        ):
                            continue

                        existing_related_score = (
                            related_concepts.get(
                                neighbor_uri,
                                0.0,
                            )
                        )

                        related_concepts[
                            neighbor_uri
                        ] = max(
                            existing_related_score,
                            path_score,
                        )

                        if neighbor_uri in visited:
                            continue

                        previous_state = (
                            next_frontier.get(
                                neighbor_uri
                            )
                        )

                        if (
                            previous_state is None
                            or path_score
                            > previous_state["score"]
                        ):
                            next_frontier[
                                neighbor_uri
                            ] = {
                                "score": path_score,
                                "steps": path_steps,
                            }

                visited.update(
                    next_frontier
                )

                frontier = next_frontier

        for seed in concept_hits:
            related_concepts.pop(
                seed.uri,
                None,
            )

        return GraphExpansionResult(
            article_evidence=article_evidence,
            related_concepts=related_concepts,
        )