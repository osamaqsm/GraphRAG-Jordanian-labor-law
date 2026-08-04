from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval_contract import RetrievalResultV1


GENERATION_CONTRACT_VERSION = "generation.v1"


class FrozenGenerationModel(BaseModel):
    """Versioned model used by the generation-only boundary."""

    model_config = ConfigDict(extra="forbid")


class GenerationRequestV1(FrozenGenerationModel):
    retrieval: RetrievalResultV1
    include_debug: bool = False


class AnswerCitationV1(FrozenGenerationModel):
    article_number: int = Field(ge=1)
    label_ar: str
    uri: str
    excerpt: str = ""


class GenerationUsageV1(FrozenGenerationModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class GroundedAnswerResultV1(FrozenGenerationModel):
    """
    Generation result derived only from a completed retrieval.v1 object.

    The result does not expose retrieval methods and cannot trigger retrieval.
    """

    schema_version: Literal["generation.v1"] = GENERATION_CONTRACT_VERSION
    status: Literal[
        "generated",
        "clarification_required",
        "out_of_scope",
        "insufficient_evidence",
    ]
    question: str
    retrieval_schema_version: Literal["retrieval.v1"] = "retrieval.v1"
    answer_ar: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[AnswerCitationV1] = Field(default_factory=list)
    cited_article_numbers: list[int] = Field(default_factory=list)
    grounded: bool = False
    warnings: list[str] = Field(default_factory=list)
    model: str = ""
    usage: GenerationUsageV1 = Field(default_factory=GenerationUsageV1)
    elapsed_ms: int = Field(default=0, ge=0)
    debug: dict[str, Any] | None = None
