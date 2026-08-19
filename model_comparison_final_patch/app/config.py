from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "Jordan Legal KG API"
    app_version: str = "0.1.0"
    app_environment: str = "development"

    # ---------------------------------------------------------
    # LLM providers
    # ---------------------------------------------------------

    # Shared provider/model used by all LLM-dependent pipeline stages:
    # query planner, route verifier, article reranker, answer generator,
    # and citation-repair retry.
    pipeline_llm_provider: Literal[
        "openai",
        "anthropic",
        "google",
        "ollama",
    ] = "openai"

    pipeline_llm_model: str = "gpt-5-nano"

    # ---------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------

    # Still required even when Anthropic is used for the pipeline because
    # OpenAI embeddings and the fixed evaluation judge may still use it.
    openai_api_key: str = Field(
        min_length=20
    )

    # Kept for backward compatibility with older code paths.
    # New pipeline LLM code should use pipeline_llm_model.
    openai_chat_model: str = "gpt-5-nano"

    openai_embedding_model: str = (
        "text-embedding-3-small"
    )

    openai_reasoning_effort: str = "low"

    openai_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
    )

    openai_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    # ---------------------------------------------------------
    # Anthropic
    # ---------------------------------------------------------

    # Optional while parsing settings so the existing OpenAI-only setup
    # continues to work. The Anthropic adapter should validate that this
    # key is present when pipeline_llm_provider == "anthropic".
    anthropic_api_key: str = ""

    anthropic_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
    )

    anthropic_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    # ---------------------------------------------------------
    # Google Gemini
    # ---------------------------------------------------------

    # Required only when PIPELINE_LLM_PROVIDER=google.
    google_api_key: str = ""

    google_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
    )

    # ---------------------------------------------------------
    # Ollama (local Qwen / Aya)
    # ---------------------------------------------------------

    # Docker Desktop exposes the host through host.docker.internal.
    # The API container calls the host Ollama server through this URL.
    ollama_base_url: str = "http://host.docker.internal:11434"

    # Local 8B models can be slower than hosted APIs. This is only a
    # transport ceiling; latency is still measured normally by the benchmark.
    ollama_timeout_seconds: float = Field(
        default=600.0,
        ge=1.0,
    )

    # Use one common context ceiling for both Qwen3:8b and Aya:8b/Aya-Expanse:8b.
    # The published Ollama Aya-Expanse 8B tag exposes an 8K context window, so
    # 8192 is the largest safe common value for the local-model comparison.
    ollama_num_ctx: int = Field(
        default=8192,
        ge=2048,
        le=131072,
    )

    # ---------------------------------------------------------
    # LLM output limits
    # ---------------------------------------------------------

    planner_max_output_tokens: int = Field(
        default=3000,
        ge=128,
        le=16000,
    )

    reranker_max_output_tokens: int = Field(
        default=2000,
        ge=128,
        le=16000,
    )

    generator_max_output_tokens: int = Field(
        default=3000,
        ge=128,
        le=32000,
    )

    # Provider-neutral reranker input controls. The LLM reranker receives only
    # the strongest deterministic/hybrid candidates, never the complete 142-
    # article statute. This makes the exact same reranker stage executable by
    # GPT, Gemini, Qwen 8B, and Aya 8B.
    reranker_candidate_limit: int = Field(
        default=12,
        ge=5,
        le=30,
    )

    # Total article-text character budget sent to the reranker. The per-article
    # limit is allocated dynamically from this shared budget.
    reranker_total_char_budget: int = Field(
        default=12000,
        ge=4000,
        le=50000,
    )

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
    )

    # ---------------------------------------------------------
    # Weaviate connection
    # ---------------------------------------------------------

    weaviate_api_key: str = Field(
        min_length=1
    )

    weaviate_http_host: str = "weaviate"
    weaviate_http_port: int = 8080

    weaviate_grpc_host: str = "weaviate"
    weaviate_grpc_port: int = 50051

    weaviate_connection_attempts: int = Field(
        default=30,
        ge=1,
    )

    weaviate_connection_delay_seconds: float = Field(
        default=2.0,
        ge=0.1,
    )

    # ---------------------------------------------------------
    # Weaviate collections
    # ---------------------------------------------------------

    weaviate_node_collection: str = (
        "JordanLaborLegalNode"
    )

    weaviate_edge_collection: str = (
        "JordanLaborLegalEdge"
    )

    weaviate_node_batch_size: int = Field(
        default=25,
        ge=1,
        le=200,
    )

    weaviate_edge_batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
    )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    retrieval_vector_candidates: int = Field(
        default=40,
        ge=1,
        le=200,
    )

    retrieval_bm25_candidates: int = Field(
        default=40,
        ge=1,
        le=200,
    )

    retrieval_concept_top_k: int = Field(
        default=12,
        ge=1,
        le=50,
    )

    retrieval_article_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    retrieval_rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
    )

    retrieval_paragraph_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    retrieval_graph_seed_count: int = Field(
        default=3,
        ge=1,
        le=20,
    )

    retrieval_graph_max_hops: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    retrieval_graph_edges_limit: int = Field(
        default=1000,
        ge=50,
        le=5000,
    )

    retrieval_graph_paths_per_article: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    retrieval_vector_weight: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
    )

    retrieval_bm25_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=10.0,
    )

    # ---------------------------------------------------------
    # KG file
    # ---------------------------------------------------------

    kg_ttl_path: str = (
        "/app/data/"
        "jordan_labor_law_full_knowledge_graph.ttl"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Create and cache one Settings object.
    """

    return Settings()