from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval_contract import RetrievalResultV2


GENERATION_CONTRACT_VERSION = "generation.v2"


class FrozenGenerationModel(BaseModel):
    """Versioned model used by the generation-only boundary."""

    model_config = ConfigDict(extra="forbid")


class GenerationRequestV2(FrozenGenerationModel):
    retrieval: RetrievalResultV2
    include_debug: bool = False


class AnswerCitationV2(FrozenGenerationModel):
    article_number: int = Field(ge=1)
    label_ar: str
    uri: str
    excerpt: str = ""


class GenerationUsageV2(FrozenGenerationModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class GroundedAnswerResultV2(FrozenGenerationModel):
    """
    Generation result derived only from a completed retrieval.v2 object.

    The result does not expose retrieval methods and cannot trigger retrieval.
    """

    schema_version: Literal["generation.v2"] = GENERATION_CONTRACT_VERSION
    status: Literal[
        "generated",
        "out_of_scope",
        "insufficient_evidence",
    ]
    question: str
    retrieval_schema_version: Literal["retrieval.v2"] = "retrieval.v2"
    answer_ar: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[AnswerCitationV2] = Field(default_factory=list)
    cited_article_numbers: list[int] = Field(default_factory=list)
    grounded: bool = False
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    usage: GenerationUsageV2 = Field(default_factory=GenerationUsageV2)
    elapsed_ms: int = Field(default=0, ge=0)
    debug: dict[str, Any] | None = None
