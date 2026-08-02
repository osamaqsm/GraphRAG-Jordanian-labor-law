from __future__ import annotations

import math
from dataclasses import dataclass

from openai import OpenAI

from app.config import Settings


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """
    Result returned from one OpenAI embedding request.
    """

    vectors: list[list[float]]
    input_tokens: int
    dimensions: int


class OpenAIService:
    """
    Wrapper around the OpenAI Python client.

    This class will later also contain the gpt-5-nano
    question parser and grounded-answer generator.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingBatchResult:
        """
        Generate one embedding for every supplied text.

        The returned order is guaranteed to match the input
        order by sorting the API response using its index.
        """

        if not texts:
            raise ValueError(
                "At least one text is required."
            )

        normalized_texts: list[str] = []

        for position, text in enumerate(texts):
            normalized = text.strip()

            if not normalized:
                raise ValueError(
                    "Embedding text cannot be empty. "
                    f"Empty value at position {position}."
                )

            normalized_texts.append(normalized)

        response = self.client.embeddings.create(
            model=(
                self.settings.openai_embedding_model
            ),
            input=normalized_texts,
            encoding_format="float",
        )

        ordered_items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        vectors = [
            [
                float(value)
                for value in item.embedding
            ]
            for item in ordered_items
        ]

        if len(vectors) != len(normalized_texts):
            raise RuntimeError(
                "OpenAI returned a different number of "
                "embeddings than requested."
            )

        dimensions = len(vectors[0])

        if dimensions == 0:
            raise RuntimeError(
                "OpenAI returned an empty embedding."
            )

        for vector_index, vector in enumerate(vectors):
            if len(vector) != dimensions:
                raise RuntimeError(
                    "Embedding dimensions are inconsistent. "
                    f"Vector {vector_index} has "
                    f"{len(vector)} dimensions; expected "
                    f"{dimensions}."
                )

            if not all(
                math.isfinite(value)
                for value in vector
            ):
                raise RuntimeError(
                    "OpenAI returned a vector containing "
                    "a non-finite numeric value."
                )

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = 0

        if usage is not None:
            input_tokens = int(
                getattr(
                    usage,
                    "total_tokens",
                    0,
                )
                or getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
                or 0
            )

        return EmbeddingBatchResult(
            vectors=vectors,
            input_tokens=input_tokens,
            dimensions=dimensions,
        )

    def close(self) -> None:
        """
        Close the underlying HTTP client.
        """

        self.client.close()