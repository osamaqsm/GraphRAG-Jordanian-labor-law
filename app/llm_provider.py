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
      - opencode -> hosted Qwen models through OpenCode Go /v1/messages
      - cohere   -> hosted Aya models through Cohere Chat API V2
      - ollama   -> local models through Ollama's native REST API

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
        opencode_api_key: str | None = None,
        cohere_api_key: str | None = None,
        openai_timeout_seconds: float = 120.0,
        anthropic_timeout_seconds: float = 120.0,
        google_timeout_seconds: float = 120.0,
        opencode_base_url: str = "https://opencode.ai/zen/go",
        opencode_timeout_seconds: float = 120.0,
        cohere_base_url: str = "https://api.cohere.com",
        cohere_timeout_seconds: float = 120.0,
        openai_max_retries: int = 3,
        anthropic_max_retries: int = 3,
        ollama_base_url: str = "http://host.docker.internal:11434",
        ollama_timeout_seconds: float = 600.0,
        ollama_num_ctx: int = 8192,
    ) -> None:
        provider = provider.strip().lower()
        allowed = {
            "openai",
            "anthropic",
            "google",
            "opencode",
            "cohere",
            "ollama",
        }
        if provider not in allowed:
            raise ValueError(
                "pipeline_llm_provider must be one of: "
                + ", ".join(sorted(allowed))
            )
        if not model.strip():
            raise ValueError("pipeline_llm_model must not be empty.")

        self.provider = provider
        self.model = model.strip()
        self.opencode_api_key = (opencode_api_key or "").strip()
        self.opencode_base_url = opencode_base_url.rstrip("/")
        self.opencode_timeout_seconds = float(opencode_timeout_seconds)

        self.cohere_api_key = (cohere_api_key or "").strip()
        self.cohere_base_url = cohere_base_url.rstrip("/")
        self.cohere_timeout_seconds = float(cohere_timeout_seconds)

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_timeout_seconds = float(ollama_timeout_seconds)
        self.ollama_num_ctx = int(ollama_num_ctx)

        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None
        self._google_client: Any | None = None
        self._opencode_session: requests.Session | None = None
        self._cohere_session: requests.Session | None = None
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

        elif provider == "opencode":
            if not self.opencode_api_key:
                raise ValueError(
                    "OPENCODE_API_KEY is required when "
                    "PIPELINE_LLM_PROVIDER=opencode."
                )
            self._opencode_session = requests.Session()

        elif provider == "cohere":
            if not self.cohere_api_key:
                raise ValueError(
                    "COHERE_API_KEY is required when "
                    "PIPELINE_LLM_PROVIDER=cohere."
                )
            self._cohere_session = requests.Session()

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
        if self.provider == "opencode":
            return self._call_opencode(
                instructions=instructions,
                payload=payload,
                response_model=response_model,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
        if self.provider == "cohere":
            return self._call_cohere(
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
                    # Pydantic emits standard JSON Schema (including
                    # additionalProperties=False for extra="forbid").
                    # Gemini's response_schema field is an OpenAPI-style
                    # subset and rejects that keyword. response_json_schema
                    # is the official JSON-Schema path and accepts the
                    # Pydantic schema used by the shared pipeline contract.
                    response_json_schema=response_model.model_json_schema(),
                    max_output_tokens=token_limit,
                ),
            )

        def usage_from(value: Any) -> LLMUsage:
            usage_obj = getattr(value, "usage_metadata", None)
            return LLMUsage(
                input_tokens=int(
                    getattr(usage_obj, "prompt_token_count", 0) or 0
                ),
                # Gemini reports visible candidate tokens and internal
                # thinking tokens separately.  Count both as output usage so
                # the provider-neutral total reflects billed generation work.
                output_tokens=(
                    int(getattr(usage_obj, "candidates_token_count", 0) or 0)
                    + int(getattr(usage_obj, "thoughts_token_count", 0) or 0)
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
    # OpenCode Go (hosted Qwen through Anthropic-compatible Messages API)
    # ------------------------------------------------------------------

    def _call_opencode(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._opencode_session is None:
            raise RuntimeError("OpenCode session is not initialized.")

        schema = _strict_schema(response_model.model_json_schema())
        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

        # OpenCode documents qwen3.7-plus on its Anthropic-compatible /v1/messages
        # endpoint. Unlike OpenAI/Gemini, the gateway does not document a native
        # JSON-Schema response parameter for this model, so the exact same schema
        # contract is supplied in the system instruction and then enforced again
        # with Pydantic in Python. No semantic repair retry is allowed.
        structured_instructions = (
            instructions
            + "\n\nSTRUCTURED OUTPUT CONTRACT\n"
            + f"Return exactly one JSON object matching schema {schema_name!r}. "
              "Return JSON only: no Markdown fences, commentary, or extra keys.\n"
            + "JSON Schema:\n"
            + schema_text
        )

        url = f"{self.opencode_base_url}/v1/messages"
        headers = {
            # OpenCode Go exposes qwen3.7-plus through its Anthropic-compatible
            # /v1/messages endpoint. Anthropic Messages authentication uses
            # x-api-key rather than the OpenAI-compatible Bearer header.
            "x-api-key": self.opencode_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        def call_once(token_limit: int) -> dict[str, Any]:
            body = {
                "model": self.model,
                "max_tokens": token_limit,
                "system": structured_instructions,
                "messages": [
                    {"role": "user", "content": user_content},
                ],
            }
            response = self._opencode_session.post(
                url,
                headers=headers,
                json=body,
                timeout=self.opencode_timeout_seconds,
            )
            if response.status_code >= 400:
                preview = response.text[:1600]
                raise RuntimeError(
                    "OpenCode request failed. "
                    f"status={response.status_code} body_preview={preview!r}"
                )
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("OpenCode returned a non-object JSON response.")
            return value

        def usage_from(value: dict[str, Any]) -> LLMUsage:
            usage = value.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            return LLMUsage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )

        def text_from(value: dict[str, Any]) -> str:
            parts: list[str] = []
            content = value.get("content")
            if not isinstance(content, list):
                return ""
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
            return "".join(parts).strip()

        response = call_once(max_output_tokens)
        total_usage = usage_from(response)
        raw_text = text_from(response)
        stop_reason = str(response.get("stop_reason") or "").lower()

        # Same fixed reliability policy used by the other providers: exactly
        # one retry for an output-token ceiling or an empty normal completion.
        budget_limited = stop_reason in {
            "max_tokens", "length", "max_output_tokens"
        }
        if budget_limited or (not raw_text and stop_reason in {"", "end_turn", "stop"}):
            retry_limit = max(max_output_tokens * 2, 3000)
            retry_response = call_once(retry_limit)
            total_usage = _combine_usage(total_usage, usage_from(retry_response))
            response = retry_response
            raw_text = text_from(response)
            stop_reason = str(response.get("stop_reason") or "").lower()
            budget_limited = stop_reason in {
                "max_tokens", "length", "max_output_tokens"
            }

        if not raw_text:
            raise RuntimeError(
                "OpenCode returned empty structured output after the fixed "
                f"recovery policy. stop_reason={stop_reason!r}"
            )
        if budget_limited:
            raise RuntimeError(
                "OpenCode structured output remained incomplete after the fixed "
                "one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
            )

        try:
            json_text = _extract_json_object(raw_text)
            validated = response_model.model_validate_json(json_text)
        except Exception as exc:
            raise RuntimeError(
                "OpenCode structured-output validation failed. "
                f"schema_name={schema_name!r} stop_reason={stop_reason!r} "
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


    # ------------------------------------------------------------------
    # Cohere (hosted Aya through Chat API V2)
    # ------------------------------------------------------------------

    def _call_cohere(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        schema_name: str,
        max_output_tokens: int,
    ) -> StructuredLLMResult:
        if self._cohere_session is None:
            raise RuntimeError("Cohere session is not initialized.")

        schema = _strict_schema(response_model.model_json_schema())
        user_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # Aya Expanse is not documented by Cohere as supporting guaranteed
        # native JSON-Schema structured outputs. Keep the same logical output
        # contract by placing the schema in the system instruction, then
        # enforce it locally with the same Pydantic response model.
        #
        # There is no semantic repair retry. The only retry allowed below is
        # the frozen provider-neutral recovery for an output-token ceiling or
        # an empty otherwise-normal completion.
        structured_instructions = (
            instructions
            + "\n\nSTRUCTURED OUTPUT CONTRACT\n"
            + f"Return exactly one JSON object matching schema {schema_name!r}. "
              "Return JSON only: no Markdown fences, commentary, or extra keys.\n"
            + "JSON Schema:\n"
            + schema_text
        )

        url = f"{self.cohere_base_url}/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json",
        }

        def call_once(token_limit: int) -> dict[str, Any]:
            # Aya Expanse 32B exposes a 4K maximum output window.
            effective_limit = min(int(token_limit), 4000)

            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": structured_instructions,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "max_tokens": effective_limit,
            }

            response = self._cohere_session.post(
                url,
                headers=headers,
                json=body,
                timeout=self.cohere_timeout_seconds,
            )

            if response.status_code >= 400:
                preview = response.text[:1600]
                raise RuntimeError(
                    "Cohere request failed. "
                    f"status={response.status_code} "
                    f"body_preview={preview!r}"
                )

            value = response.json()

            if not isinstance(value, dict):
                raise RuntimeError(
                    "Cohere returned a non-object JSON response."
                )

            return value

        def usage_from(value: dict[str, Any]) -> LLMUsage:
            usage = value.get("usage")
            if not isinstance(usage, dict):
                usage = {}

            tokens = usage.get("tokens")
            if not isinstance(tokens, dict):
                tokens = usage.get("billed_units")
            if not isinstance(tokens, dict):
                tokens = {}

            return LLMUsage(
                input_tokens=int(tokens.get("input_tokens") or 0),
                output_tokens=int(tokens.get("output_tokens") or 0),
            )

        def text_from(value: dict[str, Any]) -> str:
            message = value.get("message")
            if not isinstance(message, dict):
                return ""

            content = message.get("content")
            if not isinstance(content, list):
                return ""

            parts: list[str] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))

            return "".join(parts).strip()

        response = call_once(max_output_tokens)
        total_usage = usage_from(response)
        raw_text = text_from(response)
        finish_reason = str(
            response.get("finish_reason") or ""
        ).lower()

        if finish_reason in {"error", "timeout"}:
            raise RuntimeError(
                "Cohere generation failed. "
                f"finish_reason={finish_reason!r}"
            )

        budget_limited = finish_reason in {
            "max_tokens",
            "length",
            "max_output_tokens",
        }

        if budget_limited or (
            not raw_text
            and finish_reason in {"", "complete", "stop_sequence"}
        ):
            retry_limit = min(
                max(max_output_tokens * 2, 3000),
                4000,
            )

            retry_response = call_once(retry_limit)

            total_usage = _combine_usage(
                total_usage,
                usage_from(retry_response),
            )

            response = retry_response
            raw_text = text_from(response)
            finish_reason = str(
                response.get("finish_reason") or ""
            ).lower()

            if finish_reason in {"error", "timeout"}:
                raise RuntimeError(
                    "Cohere generation failed during the fixed "
                    "output-budget recovery attempt. "
                    f"finish_reason={finish_reason!r}"
                )

            budget_limited = finish_reason in {
                "max_tokens",
                "length",
                "max_output_tokens",
            }

        if not raw_text:
            raise RuntimeError(
                "Cohere returned empty structured output after the "
                "fixed recovery policy. "
                f"finish_reason={finish_reason!r}"
            )

        if budget_limited:
            raise RuntimeError(
                "Cohere structured output remained incomplete after "
                "the fixed one-retry output-budget policy. "
                f"raw_text_preview={raw_text[:1200]!r}"
            )

        try:
            json_text = _extract_json_object(raw_text)

            # Aya may emit literal ASCII control characters such as
            # newlines inside JSON string values. Python's lenient JSON
            # decoder accepts those characters without changing their
            # semantic content. Pydantic still performs the same strict
            # schema/type validation on the resulting object.
            parsed_json = json.loads(
                json_text,
                strict=False,
            )
            validated = response_model.model_validate(
                parsed_json
            )
        except Exception as exc:
            raise RuntimeError(
                "Cohere structured-output validation failed. "
                f"schema_name={schema_name!r} "
                f"finish_reason={finish_reason!r} "
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
        opencode_api_key=getattr(settings, "opencode_api_key", ""),
        cohere_api_key=getattr(settings, "cohere_api_key", ""),
        openai_timeout_seconds=settings.openai_timeout_seconds,
        anthropic_timeout_seconds=getattr(
            settings, "anthropic_timeout_seconds", 120.0
        ),
        google_timeout_seconds=getattr(settings, "google_timeout_seconds", 120.0),
        opencode_base_url=getattr(
            settings, "opencode_base_url", "https://opencode.ai/zen/go"
        ),
        opencode_timeout_seconds=getattr(
            settings, "opencode_timeout_seconds", 120.0
        ),
        cohere_base_url=getattr(
            settings,
            "cohere_base_url",
            "https://api.cohere.com",
        ),
        cohere_timeout_seconds=getattr(
            settings,
            "cohere_timeout_seconds",
            120.0,
        ),
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