from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

try:  # Imported only when the Google provider is selected.
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency checked at runtime
    genai = None
    genai_types = None


@dataclass(slots=True)
class LLMUsage:
    """Provider-neutral token-usage container."""

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
    """Provider-neutral structured LLM response."""

    data: dict[str, Any]
    usage: LLMUsage
    provider: str
    model: str
    raw_text: str


def _strip_json_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```json"):
        value = value[len("```json") :].strip()
    elif value.startswith("```"):
        value = value[3:].strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    return value


def _extract_json_object(text: str) -> str:
    value = _strip_json_fences(text)
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return value[start : end + 1]


def _strict_schema(value: Any) -> Any:
    """Normalize a Pydantic schema into the strict shared output contract.

    Every object property is required and Pydantic ``default`` annotations are
    removed. OpenAI requires this form natively; Ollama accepts the same JSON
    Schema. Google receives the original Pydantic model but the returned payload
    is still validated against the same Pydantic contract in Python.
    """
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "default":
                continue
            normalized[key] = _strict_schema(item)
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    return value


def _combine_usage(first: LLMUsage, second: LLMUsage) -> LLMUsage:
    return LLMUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
    )


class StructuredLLMProvider:
    """Unified structured-output adapter for every evaluated LLM provider.

    Supported providers:
      - openai   -> GPT models
      - anthropic -> Claude models (kept for compatibility; not required now)
      - google   -> Gemini models through the official google-genai SDK
      - ollama   -> local Qwen/Aya models through Ollama's native REST API

    Planner, route verifier, reranker, generator, and citation retry depend only
    on ``generate_structured``. Provider-specific details remain isolated here.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        openai_timeout_seconds: float = 120.0,
        anthropic_timeout_seconds: float = 120.0,
        google_timeout_seconds: float = 120.0,
        openai_max_retries: int = 3,
        anthropic_max_retries: int = 3,
        ollama_base_url: str = "http://host.docker.internal:11434",
        ollama_timeout_seconds: float = 600.0,
        ollama_num_ctx: int = 8192,
    ) -> None:
        provider = provider.strip().lower()
        allowed = {"openai", "anthropic", "google", "ollama"}
        if provider not in allowed:
            raise ValueError(
                "pipeline_llm_provider must be one of: "
                + ", ".join(sorted(allowed))
            )
        if not model.strip():
            raise ValueError("pipeline_llm_model must not be empty.")

        self.provider = provider
        self.model = model.strip()
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_timeout_seconds = float(ollama_timeout_seconds)
        self.ollama_num_ctx = int(ollama_num_ctx)

        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None
        self._google_client: Any | None = None
        self._ollama_session: requests.Session | None = None

        if provider == "openai":
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

        elif provider == "anthropic":
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

        elif provider == "google":
            if not google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is required when "
                    "PIPELINE_LLM_PROVIDER=google."
                )
            if genai is None or genai_types is None:
                raise RuntimeError(
                    "The 'google-genai' package is required when "
                    "PIPELINE_LLM_PROVIDER=google."
                )
            # google-genai HttpOptions timeout is expressed in milliseconds.
            self._google_client = genai.Client(
                api_key=google_api_key,
                http_options=genai_types.HttpOptions(
                    timeout=int(float(google_timeout_seconds) * 1000),
                ),
            )

        else:  # ollama
            self._ollama_session = requests.Session()

    def generate_structured(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
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
        if self.provider == "anthropic":
            return self._call_anthropic(
                instructions=instructions,
                payload=payload,
                response_model=response_model,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
        if self.provider == "google":
            return self._call_google(
                instructions=instructions,
                payload=payload,
                response_model=response_model,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
        return self._call_ollama(
            instructions=instructions,
            payload=payload,
            response_model=response_model,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

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

        schema = _strict_schema(response_model.model_json_schema())
        request_kwargs = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
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

        def usage_from(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage", None)
            return LLMUsage(
                input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
            )

        def incomplete_reason(value: Any) -> str:
            details = getattr(value, "incomplete_details", None)
            return str(getattr(details, "reason", "") or "")

        response = self._openai_client.responses.create(
            **request_kwargs,
            max_output_tokens=max_output_tokens,
        )
        total_usage = usage_from(response)
        raw_text = str(getattr(response, "output_text", "") or "").strip()
        status = str(getattr(response, "status", "") or "")
        reason = incomplete_reason(response)

        # Exactly one provider-neutral output-budget recovery attempt.
        if not raw_text or (status == "incomplete" and reason == "max_output_tokens"):
            retry_limit = max(max_output_tokens * 2, 3000)
            retry_response = self._openai_client.responses.create(
                **request_kwargs,
                max_output_tokens=retry_limit,
            )
            total_usage = _combine_usage(total_usage, usage_from(retry_response))
            response = retry_response
            raw_text = str(getattr(response, "output_text", "") or "").strip()
            status = str(getattr(response, "status", "") or "")
            reason = incomplete_reason(response)

        if not raw_text:
            raise RuntimeError(
                "OpenAI returned empty structured output after the fixed "
                f"recovery policy. status={status!r} reason={reason!r}"
            )
        if status == "incomplete" and reason == "max_output_tokens":
            raise RuntimeError(
                "OpenAI structured output remained incomplete after the fixed "
                "one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
            )

        try:
            validated = response_model.model_validate_json(raw_text)
        except Exception as exc:
            raise RuntimeError(
                "OpenAI structured-output validation failed. "
                f"schema_name={schema_name!r} raw_text_preview={raw_text[:1200]!r} "
                f"validation_error={type(exc).__name__}: {exc}"
            ) from exc

        return StructuredLLMResult(
            data=validated.model_dump(mode="json"),
            usage=total_usage,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------
    # Anthropic (kept for compatibility)
    # ------------------------------------------------------------------

    def _call_anthropic(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not initialized.")

        parse_method = getattr(self._anthropic_client.messages, "parse", None)
        if parse_method is None:
            raise RuntimeError(
                "The installed Anthropic SDK does not expose messages.parse(); "
                "upgrade the anthropic package before using this provider."
            )

        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        def call_once(token_limit: int) -> Any:
            return parse_method(
                model=self.model,
                max_tokens=token_limit,
                system=instructions,
                messages=[{"role": "user", "content": user_content}],
                output_format=response_model,
            )

        def usage_from(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage", None)
            return LLMUsage(
                input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
            )

        def text_from(value: Any) -> str:
            parts: list[str] = []
            for block in getattr(value, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(str(text))
            return "".join(parts).strip()

        response = call_once(max_output_tokens)
        total_usage = usage_from(response)
        raw_text = text_from(response)
        stop_reason = str(getattr(response, "stop_reason", "") or "")

        if not raw_text or stop_reason == "max_tokens":
            retry_limit = max(max_output_tokens * 2, 3000)
            retry_response = call_once(retry_limit)
            total_usage = _combine_usage(total_usage, usage_from(retry_response))
            response = retry_response
            raw_text = text_from(response)
            stop_reason = str(getattr(response, "stop_reason", "") or "")

        if not raw_text:
            raise RuntimeError(
                "Anthropic returned empty structured output after the fixed "
                f"recovery policy. stop_reason={stop_reason!r}"
            )
        if stop_reason == "max_tokens":
            raise RuntimeError(
                "Anthropic structured output remained incomplete after the fixed "
                "one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
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
            raise RuntimeError(
                "Anthropic structured-output validation failed. "
                f"schema_name={schema_name!r} raw_text_preview={raw_text[:1200]!r} "
                f"validation_error={type(exc).__name__}: {exc}"
            ) from exc

        return StructuredLLMResult(
            data=validated.model_dump(mode="json"),
            usage=total_usage,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------

    def _call_google(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._google_client is None or genai_types is None:
            raise RuntimeError("Google GenAI client is not initialized.")

        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        def call_once(token_limit: int) -> Any:
            return self._google_client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=instructions,
                    response_mime_type="application/json",
                    response_schema=response_model,
                    max_output_tokens=token_limit,
                ),
            )

        def usage_from(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage_metadata", None)
            return LLMUsage(
                input_tokens=int(
                    getattr(usage_obj, "prompt_token_count", 0) or 0
                ),
                output_tokens=int(
                    getattr(usage_obj, "candidates_token_count", 0) or 0
                ),
            )

        def text_from(value: Any) -> str:
            try:
                return str(getattr(value, "text", "") or "").strip()
            except Exception:
                return ""

        def finish_reason(value: Any) -> str:
            candidates = getattr(value, "candidates", None) or []
            if not candidates:
                return ""
            reason = getattr(candidates[0], "finish_reason", None)
            name = getattr(reason, "name", None)
            if name:
                return str(name).upper()
            text = str(reason or "").upper()
            return text.rsplit(".", 1)[-1]

        response = call_once(max_output_tokens)
        total_usage = usage_from(response)
        raw_text = text_from(response)
        reason = finish_reason(response)

        # Exactly one retry only for an explicit token ceiling or an otherwise
        # empty normal completion. Safety/block reasons are never retried here.
        if reason == "MAX_TOKENS" or (not raw_text and reason in {"", "STOP"}):
            retry_limit = max(max_output_tokens * 2, 3000)
            retry_response = call_once(retry_limit)
            total_usage = _combine_usage(total_usage, usage_from(retry_response))
            response = retry_response
            raw_text = text_from(response)
            reason = finish_reason(response)

        if not raw_text:
            raise RuntimeError(
                "Gemini returned no structured text. "
                f"finish_reason={reason!r} schema_name={schema_name!r}"
            )
        if reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini structured output remained incomplete after the fixed "
                "one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
            )
        if reason not in {"", "STOP"}:
            raise RuntimeError(
                "Gemini stopped for a non-normal reason. "
                f"finish_reason={reason!r} raw_text_preview={raw_text[:1200]!r}"
            )

        try:
            validated = response_model.model_validate_json(raw_text)
        except Exception as exc:
            raise RuntimeError(
                "Gemini structured-output validation failed. "
                f"schema_name={schema_name!r} raw_text_preview={raw_text[:1200]!r} "
                f"validation_error={type(exc).__name__}: {exc}"
            ) from exc

        return StructuredLLMResult(
            data=validated.model_dump(mode="json"),
            usage=total_usage,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------
    # Ollama (Qwen / Aya)
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._ollama_session is None:
            raise RuntimeError("Ollama session is not initialized.")

        schema = _strict_schema(response_model.model_json_schema())
        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        url = f"{self.ollama_base_url}/api/chat"

        def call_once(token_limit: int) -> dict[str, Any]:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "format": schema,
                # Disable separate thinking output so the fixed output budget is
                # reserved for the structured JSON contract on thinking models.
                "think": False,
                "options": {
                    "num_ctx": self.ollama_num_ctx,
                    "num_predict": token_limit,
                },
            }
            response = self._ollama_session.post(
                url,
                json=body,
                timeout=self.ollama_timeout_seconds,
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("Ollama returned a non-object JSON response.")
            return value

        def usage_from(value: dict[str, Any]) -> LLMUsage:
            return LLMUsage(
                input_tokens=int(value.get("prompt_eval_count") or 0),
                output_tokens=int(value.get("eval_count") or 0),
            )

        def text_from(value: dict[str, Any]) -> str:
            message = value.get("message")
            if isinstance(message, dict):
                return str(message.get("content") or "").strip()
            return ""

        response = call_once(max_output_tokens)
        total_usage = usage_from(response)
        raw_text = text_from(response)
        done_reason = str(response.get("done_reason") or "").lower()

        # Ollama reports a length-limited generation through done_reason. Also
        # retry one empty normal completion. No other semantic retry is allowed.
        budget_limited = done_reason in {"length", "max_tokens", "max_output_tokens"}
        if budget_limited or (not raw_text and done_reason in {"", "stop"}):
            retry_limit = max(max_output_tokens * 2, 3000)
            retry_response = call_once(retry_limit)
            total_usage = _combine_usage(total_usage, usage_from(retry_response))
            response = retry_response
            raw_text = text_from(response)
            done_reason = str(response.get("done_reason") or "").lower()
            budget_limited = done_reason in {
                "length", "max_tokens", "max_output_tokens"
            }

        if not raw_text:
            raise RuntimeError(
                "Ollama returned empty structured output after the fixed "
                f"recovery policy. done_reason={done_reason!r}"
            )
        if budget_limited:
            raise RuntimeError(
                "Ollama structured output remained incomplete after the fixed "
                "one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
            )

        try:
            json_text = _extract_json_object(raw_text)
            validated = response_model.model_validate_json(json_text)
        except Exception as exc:
            raise RuntimeError(
                "Ollama structured-output validation failed. "
                f"schema_name={schema_name!r} done_reason={done_reason!r} "
                f"raw_text_preview={raw_text[:1200]!r} "
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
    """Build the one shared adapter used by all LLM-dependent stages."""

    return StructuredLLMProvider(
        provider=settings.pipeline_llm_provider,
        model=settings.pipeline_llm_model,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=getattr(settings, "anthropic_api_key", ""),
        google_api_key=getattr(settings, "google_api_key", ""),
        openai_timeout_seconds=settings.openai_timeout_seconds,
        anthropic_timeout_seconds=getattr(
            settings, "anthropic_timeout_seconds", 120.0
        ),
        google_timeout_seconds=getattr(settings, "google_timeout_seconds", 120.0),
        openai_max_retries=settings.openai_max_retries,
        anthropic_max_retries=getattr(settings, "anthropic_max_retries", 3),
        ollama_base_url=getattr(
            settings,
            "ollama_base_url",
            "http://host.docker.internal:11434",
        ),
        ollama_timeout_seconds=getattr(settings, "ollama_timeout_seconds", 600.0),
        ollama_num_ctx=getattr(settings, "ollama_num_ctx", 8192),
    )
