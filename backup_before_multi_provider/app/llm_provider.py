from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel


@dataclass(slots=True)
class LLMUsage:
    """
    Provider-neutral token-usage container.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class StructuredLLMResult:
    """
    Provider-neutral structured LLM response.
    """

    data: dict[str, Any]
    usage: LLMUsage
    provider: str
    model: str
    raw_text: str


def _strip_json_fences(text: str) -> str:
    """
    Remove optional Markdown JSON fences without changing JSON content.
    """

    value = text.strip()

    if value.startswith("```json"):
        value = value[len("```json") :].strip()
    elif value.startswith("```"):
        value = value[3:].strip()

    if value.endswith("```"):
        value = value[:-3].strip()

    return value


def _extract_json_object(text: str) -> str:
    """
    Return the JSON object from a response.

    Normally the model should return JSON only. This small defensive fallback
    handles accidental surrounding text without changing the semantic content.
    """

    value = _strip_json_fences(text)

    if value.startswith("{") and value.endswith("}"):
        return value

    start = value.find("{")
    end = value.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return value[start : end + 1]


def _openai_strict_schema(value: Any) -> Any:
    """
    Normalize a Pydantic JSON Schema for OpenAI strict structured outputs.

    OpenAI strict mode requires every property of an object to be listed in
    ``required``. Pydantic omits fields with defaults from ``required`` even
    when the model expects the field to be present in our pipeline contract.
    This provider-level normalization keeps the Pydantic model unchanged while
    producing a strict schema that works consistently for planner, reranker,
    generator, and any future structured stage.
    """
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}

        for key, item in value.items():
            # ``default`` is a Pydantic annotation, not part of the output
            # contract we want the model to rely on in strict mode.
            if key == "default":
                continue
            normalized[key] = _openai_strict_schema(item)

        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties.keys())

        return normalized

    if isinstance(value, list):
        return [_openai_strict_schema(item) for item in value]

    return value


class StructuredLLMProvider:
    """
    Unified structured-output adapter for OpenAI and Anthropic.

    The rest of the GraphRAG pipeline should depend only on this class.
    Provider-specific API details stay here.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openai_timeout_seconds: float = 120.0,
        anthropic_timeout_seconds: float = 120.0,
        openai_max_retries: int = 3,
        anthropic_max_retries: int = 3,
    ) -> None:
        provider = provider.strip().lower()

        if provider not in {"openai", "anthropic"}:
            raise ValueError(
                "pipeline_llm_provider must be either "
                "'openai' or 'anthropic'."
            )

        if not model.strip():
            raise ValueError("pipeline_llm_model must not be empty.")

        self.provider = provider
        self.model = model.strip()

        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None

        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when "
                    "PIPELINE_LLM_PROVIDER=openai."
                )

            self._openai_client = OpenAI(
                api_key=openai_api_key,
                timeout=openai_timeout_seconds,
                max_retries=openai_max_retries,
            )

        else:
            if not anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when "
                    "PIPELINE_LLM_PROVIDER=anthropic."
                )

            self._anthropic_client = Anthropic(
                api_key=anthropic_api_key,
                timeout=anthropic_timeout_seconds,
                max_retries=anthropic_max_retries,
            )

    def generate_structured(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        """
        Generate and validate one structured response.

        response_model:
            Pydantic model expected by the calling stage, e.g.
            LegalQueryPlan, ArticleSelection, or AnswerDraft.
        """

        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero.")

        if self.provider == "openai":
            return self._call_openai(
                instructions=instructions,
                payload=payload,
                response_model=response_model,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )

        return self._call_anthropic(
            instructions=instructions,
            payload=payload,
            response_model=response_model,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        )

    def _call_openai(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        schema = _openai_strict_schema(response_model.model_json_schema())

        request_kwargs = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            "store": False,
        }

        def _usage_from_response(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage", None)
            return LLMUsage(
                input_tokens=int(
                    getattr(usage_obj, "input_tokens", 0) or 0
                ),
                output_tokens=int(
                    getattr(usage_obj, "output_tokens", 0) or 0
                ),
            )

        def _incomplete_reason(value: Any) -> str:
            details = getattr(value, "incomplete_details", None)
            return str(getattr(details, "reason", "") or "")

        response = self._openai_client.responses.create(
            **request_kwargs,
            max_output_tokens=max_output_tokens,
        )

        total_usage = _usage_from_response(response)
        raw_text = str(
            getattr(response, "output_text", "") or ""
        ).strip()

        status = str(getattr(response, "status", "") or "")
        incomplete_reason = _incomplete_reason(response)

        # One fixed recovery attempt is allowed for an output-budget failure.
        # Important: OpenAI can return a NON-empty but truncated JSON string
        # when status='incomplete' and reason='max_output_tokens'.  The older
        # implementation retried only when output_text was empty, which let
        # truncated JSON fall through to Pydantic and fail validation.
        #
        # This retry does not change the prompt, schema, reasoning effort,
        # retrieval logic, or model.  Only the output ceiling is increased.
        should_retry_for_budget = (
            not raw_text
            or (
                status == "incomplete"
                and incomplete_reason == "max_output_tokens"
            )
        )

        if should_retry_for_budget:
            retry_max_output_tokens = max(
                max_output_tokens * 2,
                3000,
            )

            retry_response = self._openai_client.responses.create(
                **request_kwargs,
                max_output_tokens=retry_max_output_tokens,
            )

            retry_usage = _usage_from_response(retry_response)
            total_usage = LLMUsage(
                input_tokens=(
                    total_usage.input_tokens
                    + retry_usage.input_tokens
                ),
                output_tokens=(
                    total_usage.output_tokens
                    + retry_usage.output_tokens
                ),
            )

            response = retry_response
            raw_text = str(
                getattr(response, "output_text", "") or ""
            ).strip()
            status = str(getattr(response, "status", "") or "")
            incomplete_reason = _incomplete_reason(response)

        if not raw_text:
            incomplete_details = getattr(
                response,
                "incomplete_details",
                None,
            )
            raise RuntimeError(
                "OpenAI returned empty output after the fixed recovery "
                "policy. "
                f"status={status!r} "
                f"incomplete_details={incomplete_details!r}"
            )

        if (
            status == "incomplete"
            and incomplete_reason == "max_output_tokens"
        ):
            incomplete_details = getattr(
                response,
                "incomplete_details",
                None,
            )
            preview = raw_text[:1200]
            raise RuntimeError(
                "OpenAI structured output remained incomplete after the "
                "fixed one-retry output-budget recovery policy. "
                f"status={status!r} "
                f"incomplete_details={incomplete_details!r} "
                f"raw_text_preview={preview!r}"
            )

        try:
            validated = response_model.model_validate_json(raw_text)
        except Exception as exc:
            incomplete_details = getattr(response, "incomplete_details", None)
            preview = raw_text[:1200]
            raise RuntimeError(
                "OpenAI structured-output validation failed. "
                f"status={status!r} "
                f"incomplete_details={incomplete_details!r} "
                f"raw_text_preview={preview!r} "
                f"validation_error={type(exc).__name__}: {exc}"
            ) from exc

        return StructuredLLMResult(
            data=validated.model_dump(mode="json"),
            usage=total_usage,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )

    def _call_anthropic(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        """Call Claude with Anthropic-native structured outputs.

        The Anthropic Python SDK ``messages.parse`` helper converts the supplied
        Pydantic model to the provider's native structured-output schema and
        validates the response.  A single fixed retry is allowed only when the
        provider explicitly reports ``stop_reason='max_tokens'`` (or returns an
        empty text response), using the same larger output ceiling policy as
        OpenAI.
        """
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not initialized.")

        messages_api = self._anthropic_client.messages
        parse_method = getattr(messages_api, "parse", None)
        if parse_method is None:
            raise RuntimeError(
                "The installed Anthropic Python SDK does not expose "
                "client.messages.parse(), which is required for native "
                "structured outputs. Upgrade the 'anthropic' package before "
                "running a model-comparison benchmark."
            )

        user_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def _call_once(token_limit: int) -> Any:
            try:
                return parse_method(
                    model=self.model,
                    max_tokens=token_limit,
                    system=instructions,
                    messages=[
                        {
                            "role": "user",
                            "content": user_content,
                        }
                    ],
                    output_format=response_model,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Anthropic native structured-output call failed before "
                    "the request could be completed. Ensure the installed "
                    "Anthropic SDK supports "
                    "messages.parse(output_format=...). "
                    f"Original error: {exc}"
                ) from exc

        def _usage_from_response(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage", None)
            return LLMUsage(
                input_tokens=int(
                    getattr(usage_obj, "input_tokens", 0) or 0
                ),
                output_tokens=int(
                    getattr(usage_obj, "output_tokens", 0) or 0
                ),
            )

        def _text_from_response(value: Any) -> str:
            text_parts: list[str] = []
            for block in getattr(value, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    block_text = getattr(block, "text", None)
                    if block_text:
                        text_parts.append(str(block_text))
            return "".join(text_parts).strip()

        response = _call_once(max_output_tokens)
        total_usage = _usage_from_response(response)
        raw_text = _text_from_response(response)
        stop_reason = str(getattr(response, "stop_reason", "") or "")

        # Anthropic documents max_tokens as a truncation condition for
        # structured outputs and recommends retrying with a larger token limit.
        # Use exactly one deterministic retry, matching the OpenAI policy.
        if not raw_text or stop_reason == "max_tokens":
            retry_max_output_tokens = max(
                max_output_tokens * 2,
                3000,
            )

            retry_response = _call_once(retry_max_output_tokens)
            retry_usage = _usage_from_response(retry_response)
            total_usage = LLMUsage(
                input_tokens=(
                    total_usage.input_tokens
                    + retry_usage.input_tokens
                ),
                output_tokens=(
                    total_usage.output_tokens
                    + retry_usage.output_tokens
                ),
            )

            response = retry_response
            raw_text = _text_from_response(response)
            stop_reason = str(
                getattr(response, "stop_reason", "") or ""
            )

        if not raw_text:
            raise RuntimeError(
                "Anthropic returned empty structured output after the fixed "
                "recovery policy. "
                f"stop_reason={stop_reason!r}"
            )

        if stop_reason == "max_tokens":
            preview = raw_text[:1200]
            raise RuntimeError(
                "Anthropic structured output remained incomplete after the "
                "fixed one-retry output-budget recovery policy. "
                f"stop_reason={stop_reason!r} "
                f"raw_text_preview={preview!r}"
            )

        parsed_output = getattr(response, "parsed_output", None)

        try:
            if isinstance(parsed_output, response_model):
                validated = parsed_output
            elif parsed_output is not None:
                validated = response_model.model_validate(parsed_output)
            else:
                validated = response_model.model_validate_json(raw_text)
        except Exception as exc:
            preview = raw_text[:1200]
            raise RuntimeError(
                "Anthropic native structured-output validation failed. "
                f"schema_name={schema_name!r} "
                f"stop_reason={stop_reason!r} "
                f"raw_text_preview={preview!r} "
                f"validation_error={type(exc).__name__}: {exc}"
            ) from exc

        return StructuredLLMResult(
            data=validated.model_dump(mode="json"),
            usage=total_usage,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )


def build_pipeline_llm(settings: Any) -> StructuredLLMProvider:
    """
    Build the shared LLM adapter from app Settings.

    All LLM-dependent pipeline stages should call this factory so the entire
    pipeline switches provider/model consistently from environment settings.
    """

    return StructuredLLMProvider(
        provider=settings.pipeline_llm_provider,
        model=settings.pipeline_llm_model,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        openai_timeout_seconds=settings.openai_timeout_seconds,
        anthropic_timeout_seconds=settings.anthropic_timeout_seconds,
        openai_max_retries=settings.openai_max_retries,
        anthropic_max_retries=settings.anthropic_max_retries,
    )