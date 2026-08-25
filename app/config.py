from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the final ontology-driven GraphRAG pipeline.

    The legal-evidence retrieval architecture is fixed: semantic/lexical search
    is allowed only over retrieval-eligible ontology/KG concepts, and legal
    articles can enter the evidence set only through graph traversal.
    """

    # Application
    app_name: str = "Jordan Legal KG API"
    app_version: str = "1.0.0-graphrag"
    app_environment: str = "development"

    # One shared evaluated model for every LLM-dependent stage.
    pipeline_llm_provider: Literal[
        "openai", "anthropic", "google", "opencode", "cohere", "ollama"
    ] = "openai"
    pipeline_llm_model: str = "gpt-5-nano"

    # Fixed LLM output budgets across compared models.
    planner_max_output_tokens: int = Field(default=3000, ge=128, le=16000)
    reranker_max_output_tokens: int = Field(default=2000, ge=128, le=16000)
    generator_max_output_tokens: int = Field(default=3000, ge=128, le=32000)

    # Planner policy. Only proposed abstentions are verified; retrieve decisions
    # proceed directly to concept linking and graph traversal.
    planner_verify_abstain: bool = True
    planner_abstain_verify_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    # OpenAI remains the fixed embedding provider and evaluation-judge provider.
    openai_api_key: str = Field(min_length=1)
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = Field(default=120.0, ge=1.0)
    openai_max_retries: int = Field(default=3, ge=0, le=10)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # Optional evaluated-model providers.
    anthropic_api_key: str = ""
    anthropic_timeout_seconds: float = Field(default=120.0, ge=1.0)
    anthropic_max_retries: int = Field(default=3, ge=0, le=10)

    google_api_key: str = ""
    google_timeout_seconds: float = Field(default=120.0, ge=1.0)

    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/go"
    opencode_timeout_seconds: float = Field(default=120.0, ge=1.0)

    cohere_api_key: str = ""
    cohere_base_url: str = "https://api.cohere.com"
    cohere_timeout_seconds: float = Field(default=120.0, ge=1.0)

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_timeout_seconds: float = Field(default=600.0, ge=1.0)
    ollama_num_ctx: int = Field(default=8192, ge=2048, le=131072)

    # Provider-neutral reranker input limits.
    reranker_candidate_limit: int = Field(default=12, ge=5, le=30)
    reranker_total_char_budget: int = Field(default=12000, ge=4000, le=50000)
    reranker_article_char_limit: int = Field(default=2500, ge=800, le=10000)

    # Weaviate.
    weaviate_api_key: str = Field(min_length=1)
    weaviate_http_host: str = "weaviate"
    weaviate_http_port: int = 8080
    weaviate_grpc_host: str = "weaviate"
    weaviate_grpc_port: int = 50051
    weaviate_connection_attempts: int = Field(default=30, ge=1)
    weaviate_connection_delay_seconds: float = Field(default=2.0, ge=0.1)
    weaviate_node_collection: str = "JordanLaborLegalNode"
    weaviate_edge_collection: str = "JordanLaborLegalEdge"
    weaviate_node_batch_size: int = Field(default=25, ge=1, le=200)
    weaviate_edge_batch_size: int = Field(default=100, ge=1, le=500)

    # Concept linking. These searches NEVER introduce legal Article/Paragraph
    # nodes; the collection is positively filtered by retrievalEligible=true.
    concept_vector_candidates: int = Field(default=40, ge=1, le=200)
    concept_bm25_candidates: int = Field(default=40, ge=1, le=200)
    concept_top_k: int = Field(default=16, ge=1, le=50)
    concept_rrf_k: int = Field(default=60, ge=1, le=1000)

    concept_semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    concept_lexical_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    concept_hint_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    concept_specificity_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    graph_seed_count: int = Field(default=4, ge=1, le=12)
    graph_seed_min_score: float = Field(default=0.28, ge=0.0, le=1.0)
    graph_seed_relative_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # Relation-aware graph traversal.
    graph_max_hops: int = Field(default=2, ge=1, le=3)
    graph_edges_limit: int = Field(default=1500, ge=50, le=5000)
    graph_paths_per_article: int = Field(default=5, ge=1, le=20)
    graph_hop_decay: float = Field(default=0.72, ge=0.1, le=1.0)

    # Graph-derived article candidate ranking.
    issue_article_candidates: int = Field(default=8, ge=1, le=20)
    article_top_k: int = Field(default=5, ge=1, le=10)
    article_graph_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    article_issue_relevance_weight: float = Field(default=0.25, ge=0.0, le=1.0)

    # KG file.
    kg_ttl_path: str = "/app/data/jordan_labor_law_full_knowledge_graph.ttl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
