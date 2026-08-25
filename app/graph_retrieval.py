from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from weaviate.classes.query import Filter

from app.config import Settings
from app.retrieval_models import RetrievalHit


LAW_NAMESPACE = "http://example.org/jordan-labor-law#"
ARTICLE_URI_PATTERN = re.compile(
    r"^http://example\.org/jordan-labor-law#article_\d+$"
)

# Direct concept <-> legal-source evidence bridges. `hasArticle` is
# intentionally absent: traversing Law -> all 142 Articles would be a broad
# statutory catalogue scan rather than semantic graph retrieval.
EVIDENCE_RELATION_WEIGHTS: dict[str, float] = {
    "supportedByArticle": 1.00,
    "regulatedBy": 0.95,
    "regulates": 0.90,
    # Precise bridge for definition concepts reached through `definedIn`.
    "hasDefinition": 0.90,
}

# Semantic relations used to move between legal concepts before reaching an
# evidence bridge. These are traversed in either direction with path provenance.
SEMANTIC_RELATION_WEIGHTS: dict[str, float] = {
    "hasCondition": 0.88,
    "violatesRight": 0.86,
    "breachesObligation": 0.86,
    "resultsIn": 0.82,
    "basedOnEvent": 0.82,
    "hasRight": 0.74,
    "hasObligation": 0.74,
    "defines": 0.78,
    "definedIn": 0.78,
    "isPartyTo": 0.68,
    "hasEmployer": 0.64,
    "hasWorker": 0.64,
    "committedBy": 0.62,
    "sufferedBy": 0.62,
}

RELATION_WEIGHTS = {
    **EVIDENCE_RELATION_WEIGHTS,
    **SEMANTIC_RELATION_WEIGHTS,
}
TRAVERSABLE_RELATIONS = frozenset(RELATION_WEIGHTS)

EDGE_RETURN_PROPERTIES = [
    "sourceUri",
    "predicateUri",
    "predicateLocalName",
    "objectKind",
    "targetUri",
]


def select_graph_seeds(
    concept_hits: list[RetrievalHit],
    settings: Settings,
) -> list[RetrievalHit]:
    """Select several high-confidence semantic seeds using absolute + relative thresholds."""

    ranked = sorted(
        (hit for hit in concept_hits if hit.seed_score > 0.0),
        key=lambda hit: (-hit.seed_score, hit.uri),
    )
    if not ranked:
        return []

    top_score = ranked[0].seed_score
    threshold = max(
        settings.graph_seed_min_score,
        top_score * settings.graph_seed_relative_threshold,
    )
    selected = [hit for hit in ranked if hit.seed_score >= threshold]
    if not selected:
        selected = ranked[:1]
    return selected[: settings.graph_seed_count]


@dataclass(slots=True)
class GraphExpansionResult:
    article_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    related_concepts: dict[str, float] = field(default_factory=dict)
    seeds: list[RetrievalHit] = field(default_factory=list)


class GraphTraversalService:
    """Relation-aware traversal over URI-to-URI RDF triples in Weaviate."""

    def __init__(self, client: Any, settings: Settings) -> None:
        self.settings = settings
        self.edges = client.collections.use(settings.weaviate_edge_collection)

    @staticmethod
    def is_article_uri(uri: str) -> bool:
        return bool(ARTICLE_URI_PATTERN.match(uri))

    def _query_edges(self, *, property_name: str, uris: list[str]) -> list[dict[str, Any]]:
        if not uris:
            return []
        filters = (
            Filter.by_property(property_name).contains_any(uris)
            & Filter.by_property("objectKind").equal("uri")
            & Filter.by_property("predicateLocalName").contains_any(
                sorted(TRAVERSABLE_RELATIONS)
            )
        )
        response = self.edges.query.fetch_objects(
            filters=filters,
            limit=self.settings.graph_edges_limit,
            return_properties=EDGE_RETURN_PROPERTIES,
        )
        return [dict(item.properties) for item in response.objects]

    def outgoing_edges(self, uris: list[str]) -> list[dict[str, Any]]:
        return self._query_edges(property_name="sourceUri", uris=uris)

    def incoming_edges(self, uris: list[str]) -> list[dict[str, Any]]:
        return self._query_edges(property_name="targetUri", uris=uris)

    def semantic_degrees(self, uris: Iterable[str]) -> dict[str, int]:
        """Count unique traversable graph neighbors for concept-specificity scoring."""

        unique = sorted({uri for uri in uris if uri})
        degrees = {uri: 0 for uri in unique}
        neighbors: dict[str, set[str]] = {uri: set() for uri in unique}
        for direction, edges in (
            ("outgoing", self.outgoing_edges(unique)),
            ("incoming", self.incoming_edges(unique)),
        ):
            for edge in edges:
                source = str(edge.get("sourceUri", ""))
                target = str(edge.get("targetUri", ""))
                if direction == "outgoing" and source in neighbors and target:
                    neighbors[source].add(target)
                elif direction == "incoming" and target in neighbors and source:
                    neighbors[target].add(source)
        for uri, values in neighbors.items():
            degrees[uri] = len(values)
        return degrees

    def expand_from_concepts(self, concept_hits: list[RetrievalHit]) -> GraphExpansionResult:
        seeds = select_graph_seeds(concept_hits, self.settings)
        article_evidence: dict[str, dict[str, Any]] = {}
        related_concepts: dict[str, float] = {}

        for seed in seeds:
            seed_label = (
                seed.labels_ar[0]
                if seed.labels_ar
                else (seed.labels_en[0] if seed.labels_en else seed.local_name)
            )
            frontier: dict[str, dict[str, Any]] = {
                seed.uri: {"score": max(seed.seed_score, 1e-6), "steps": []}
            }
            best_seen: dict[str, float] = {seed.uri: max(seed.seed_score, 1e-6)}

            for _hop in range(1, self.settings.graph_max_hops + 1):
                if not frontier:
                    break

                frontier_uris = list(frontier)
                outgoing = self.outgoing_edges(frontier_uris)
                incoming = self.incoming_edges(frontier_uris)
                directional_edges = [("outgoing", outgoing), ("incoming", incoming)]

                # Degree normalization only for Article neighbors from the same
                # frontier concept. It prevents broad concepts from dominating.
                article_neighbors: dict[str, set[str]] = {}
                for direction, edge_list in directional_edges:
                    for edge in edge_list:
                        source = str(edge.get("sourceUri", ""))
                        target = str(edge.get("targetUri", ""))
                        current = source if direction == "outgoing" else target
                        neighbor = target if direction == "outgoing" else source
                        if current in frontier and self.is_article_uri(neighbor):
                            article_neighbors.setdefault(current, set()).add(neighbor)

                next_frontier: dict[str, dict[str, Any]] = {}
                for direction, edge_list in directional_edges:
                    for edge in edge_list:
                        source = str(edge.get("sourceUri", ""))
                        target = str(edge.get("targetUri", ""))
                        relation = str(edge.get("predicateLocalName", ""))
                        if not source or not target or relation not in RELATION_WEIGHTS:
                            continue

                        current = source if direction == "outgoing" else target
                        neighbor = target if direction == "outgoing" else source
                        state = frontier.get(current)
                        if state is None:
                            continue

                        path_score = (
                            float(state["score"])
                            * RELATION_WEIGHTS[relation]
                            * self.settings.graph_hop_decay
                        )

                        if self.is_article_uri(neighbor):
                            degree = max(1, len(article_neighbors.get(current, set())))
                            # Mild graph-degree penalty; one precise bridge remains
                            # almost unchanged while chapter-like concepts are reduced.
                            path_score /= math.sqrt(1.0 + math.log1p(degree))

                        step = {
                            "direction": direction,
                            "source_uri": source,
                            "relation": relation,
                            "target_uri": target,
                        }
                        path_steps = [*state["steps"], step]

                        if self.is_article_uri(neighbor):
                            evidence = article_evidence.setdefault(
                                neighbor,
                                {"score": 0.0, "paths": []},
                            )
                            evidence["score"] = max(float(evidence["score"]), path_score)
                            if len(evidence["paths"]) < self.settings.graph_paths_per_article:
                                evidence["paths"].append(
                                    {
                                        "seed_uri": seed.uri,
                                        "seed_label": seed_label,
                                        "seed_score": seed.seed_score,
                                        "article_uri": neighbor,
                                        "score": path_score,
                                        "steps": path_steps,
                                    }
                                )
                            continue

                        if not neighbor.startswith(LAW_NAMESPACE):
                            continue

                        related_concepts[neighbor] = max(
                            related_concepts.get(neighbor, 0.0),
                            path_score,
                        )
                        if path_score <= best_seen.get(neighbor, -1.0):
                            continue
                        best_seen[neighbor] = path_score
                        next_frontier[neighbor] = {
                            "score": path_score,
                            "steps": path_steps,
                        }

                frontier = next_frontier

        for seed in seeds:
            related_concepts.pop(seed.uri, None)

        return GraphExpansionResult(
            article_evidence=article_evidence,
            related_concepts=related_concepts,
            seeds=seeds,
        )
