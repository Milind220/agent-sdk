import asyncio
import os
import random
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import grpc
from xai_sdk import AsyncClient
from xai_sdk import chat as xai_chat
from xai_sdk.chat import Response
from xai_sdk.proto import chat_pb2

import logging

from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from bu_agent_sdk.llm.grok.serializer import GrokMessageSerializer
from bu_agent_sdk.llm.messages import BaseMessage, Function, ToolCall
from bu_agent_sdk.llm.streaming import (
    CompletionDeltaEvent,
    ModelCapabilities,
    ModelDeltaEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
)
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage

logger = logging.getLogger("bu_agent_sdk.llm.grok")


@dataclass
class ChatGrok(BaseChatModel):
    """A wrapper around the official xAI Python SDK for Grok chat models."""

    model: str

    temperature: float | None = 0.2
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = 4096
    parallel_tool_calls: bool | None = True
    reasoning_effort: Literal["low", "high"] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    user: str | None = None
    response_format: str | dict[str, Any] | type | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    search_parameters: Any | None = None
    include: list[str] | None = None
    store_messages: bool | None = None
    previous_response_id: str | None = None
    use_encrypted_content: bool | None = None
    max_turns: int | None = None
    conversation_id: str | None = None

    max_retries: int = 5
    retryable_status_codes: list[int] = field(
        default_factory=lambda: [408, 409, 429, 500, 502, 503, 504]
    )
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    api_key: str | None = None
    management_api_key: str | None = None
    api_host: str = "api.x.ai"
    management_api_host: str = "management-api.x.ai"
    timeout: float | None = None
    metadata: tuple[tuple[str, str], ...] | None = None
    channel_options: list[tuple[str, Any]] | None = None
    use_insecure_channel: bool = False

    _client: AsyncClient | None = None

    @property
    def provider(self) -> str:
        return "grok"

    @property
    def model_logger(self) -> logging.Logger:
        return logging.getLogger(f"bu_agent_sdk.llm.grok.{self.model}")

    @property
    def name(self) -> str:
        return str(self.model)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_token_stream=True,
            supports_reasoning_stream=True,
            supports_server_state=True,
        )

    def _get_client_params(self) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("XAI_API_KEY")

        params: dict[str, Any] = {
            "api_key": api_key,
            "api_host": self.api_host,
            "management_api_key": self.management_api_key,
            "management_api_host": self.management_api_host,
            "use_insecure_channel": self.use_insecure_channel,
        }

        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.metadata is not None:
            params["metadata"] = self.metadata
        if self.channel_options is not None:
            params["channel_options"] = self.channel_options

        return params

    def get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(**self._get_client_params())
        return self._client

    def _serialize_tools(self, tools: list[ToolDefinition]) -> list[chat_pb2.Tool]:
        return [
            xai_chat.tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools
        ]

    def _get_tool_choice(
        self, tool_choice: ToolChoice | None, tools: list[ToolDefinition] | None
    ) -> str | chat_pb2.ToolChoice | None:
        if tool_choice is None or tools is None:
            return None

        if tool_choice in {"auto", "required", "none"}:
            return tool_choice

        return xai_chat.required_tool(tool_choice)

    def _extract_tool_calls(self, response: Response) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []

        for call in response.tool_calls:
            arguments = call.function.arguments or "{}"
            tool_call_id = call.id or f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(
                ToolCall(
                    id=tool_call_id,
                    function=Function(
                        name=call.function.name,
                        arguments=arguments,
                    ),
                    type="function",
                )
            )

        return tool_calls

    def _extract_stop_reason(self, response: Response) -> str | None:
        finish_reason = response.finish_reason
        if not finish_reason:
            return None

        return finish_reason.removeprefix("FINISH_REASON_").lower()

    def _get_usage(self, response: Response) -> ChatInvokeUsage | None:
        usage = response.usage
        if usage is None:
            return None

        cached_prompt_tokens = usage.cached_prompt_text_tokens or None
        prompt_image_tokens = usage.prompt_image_tokens or None
        completion_tokens = usage.completion_tokens + (usage.reasoning_tokens or 0)

        return ChatInvokeUsage(
            prompt_tokens=usage.prompt_tokens,
            prompt_cached_tokens=cached_prompt_tokens,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=prompt_image_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _build_create_params(
        self,
        serialized_messages: list[chat_pb2.Message],
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        create_params: dict[str, Any] = {
            "model": self.model,
            "messages": serialized_messages,
        }

        if self.max_tokens is not None:
            create_params["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            create_params["temperature"] = self.temperature
        if self.top_p is not None:
            create_params["top_p"] = self.top_p
        if self.seed is not None:
            create_params["seed"] = self.seed
        if self.parallel_tool_calls is not None:
            create_params["parallel_tool_calls"] = self.parallel_tool_calls
        if self.reasoning_effort is not None:
            create_params["reasoning_effort"] = self.reasoning_effort
        if self.frequency_penalty is not None:
            create_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            create_params["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            create_params["stop"] = self.stop
        if self.user is not None:
            create_params["user"] = self.user
        if self.response_format is not None:
            create_params["response_format"] = self.response_format
        if self.logprobs is not None:
            create_params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            create_params["top_logprobs"] = self.top_logprobs
        if self.search_parameters is not None:
            create_params["search_parameters"] = self.search_parameters
        if self.include is not None:
            create_params["include"] = self.include
        if self.store_messages is not None:
            create_params["store_messages"] = self.store_messages
        if self.previous_response_id is not None:
            create_params["previous_response_id"] = self.previous_response_id
        if self.use_encrypted_content is not None:
            create_params["use_encrypted_content"] = self.use_encrypted_content
        if self.max_turns is not None:
            create_params["max_turns"] = self.max_turns
        if self.conversation_id is not None:
            create_params["conversation_id"] = self.conversation_id

        if tools:
            create_params["tools"] = self._serialize_tools(tools)
            selected_tool_choice = self._get_tool_choice(tool_choice, tools)
            if selected_tool_choice is not None:
                create_params["tool_choice"] = selected_tool_choice

        if extra_kwargs:
            create_params.update(extra_kwargs)

        return create_params

    def _infer_status_code(self, error: Exception) -> int | None:
        if hasattr(error, "status_code"):
            status_code = getattr(error, "status_code", None)
            if isinstance(status_code, int):
                return status_code

        if isinstance(error, grpc.RpcError):
            grpc_code = error.code()
            grpc_to_http = {
                grpc.StatusCode.INVALID_ARGUMENT: 400,
                grpc.StatusCode.UNAUTHENTICATED: 401,
                grpc.StatusCode.PERMISSION_DENIED: 403,
                grpc.StatusCode.NOT_FOUND: 404,
                grpc.StatusCode.DEADLINE_EXCEEDED: 408,
                grpc.StatusCode.ABORTED: 409,
                grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
                grpc.StatusCode.CANCELLED: 499,
                grpc.StatusCode.INTERNAL: 500,
                grpc.StatusCode.UNIMPLEMENTED: 501,
                grpc.StatusCode.UNAVAILABLE: 503,
                grpc.StatusCode.UNKNOWN: 502,
            }
            return grpc_to_http.get(grpc_code, 502)

        lowered = str(error).lower()
        if any(token in lowered for token in ["api key", "unauthorized", "401"]):
            return 401
        if any(token in lowered for token in ["forbidden", "permission denied", "403"]):
            return 403
        if any(
            token in lowered
            for token in [
                "rate limit",
                "resource exhausted",
                "quota exceeded",
                "too many requests",
                "429",
            ]
        ):
            return 429
        if any(token in lowered for token in ["timeout", "deadline exceeded", "408"]):
            return 408
        if any(token in lowered for token in ["service unavailable", "unavailable"]):
            return 503

        match = re.search(r"\b([45]\d{2})\b", lowered)
        if match:
            return int(match.group(1))

        return None

    def _error_message(self, error: Exception) -> str:
        if isinstance(error, grpc.RpcError):
            details = error.details()
            if details:
                return details
        return str(error)

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        serialized_messages = GrokMessageSerializer.serialize_messages(messages)
        create_params = self._build_create_params(
            serialized_messages=serialized_messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_kwargs=kwargs,
        )

        assert self.max_retries >= 1, "max_retries must be at least 1"

        for attempt in range(self.max_retries):
            start_time = time.time()
            self.model_logger.debug(f"🚀 Starting xAI call to {self.model}")

            try:
                chat = self.get_client().chat.create(**create_params)
                response = await chat.sample()
                elapsed = time.time() - start_time
                self.model_logger.debug(f"✅ xAI response in {elapsed:.2f}s")

                usage = self._get_usage(response)
                if usage and os.getenv("bu_agent_sdk_LLM_DEBUG"):
                    cached = usage.prompt_cached_tokens or 0
                    input_tokens = usage.prompt_tokens - cached
                    logger.info(
                        f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                    )

                return ChatInvokeCompletion(
                    content=response.content or None,
                    tool_calls=self._extract_tool_calls(response),
                    thinking=response.reasoning_content or None,
                    redacted_thinking=response.encrypted_content or None,
                    usage=usage,
                    stop_reason=self._extract_stop_reason(response),
                )
            except Exception as e:
                elapsed = time.time() - start_time
                status_code = self._infer_status_code(e) or 502
                error_message = self._error_message(e)
                self.model_logger.error(
                    f"💥 xAI call failed after {elapsed:.2f}s: status={status_code} error={error_message}"
                )

                if status_code == 429:
                    model_error: ModelProviderError = ModelRateLimitError(
                        message=error_message,
                        model=self.name,
                    )
                else:
                    model_error = ModelProviderError(
                        message=error_message,
                        status_code=status_code,
                        model=self.name,
                    )

                if (
                    model_error.status_code in self.retryable_status_codes
                    and attempt < self.max_retries - 1
                ):
                    delay = min(
                        self.retry_base_delay * (2**attempt),
                        self.retry_max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.model_logger.warning(
                        f"⚠️ xAI retry in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue

                raise model_error from e

        raise RuntimeError("Retry loop completed without return or exception")

    async def astream_invoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelDeltaEvent]:
        serialized_messages = GrokMessageSerializer.serialize_messages(messages)
        create_params = self._build_create_params(
            serialized_messages=serialized_messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_kwargs=kwargs,
        )

        assert self.max_retries >= 1, "max_retries must be at least 1"

        for attempt in range(self.max_retries):
            start_time = time.time()
            self.model_logger.debug(f"🚀 Starting xAI stream call to {self.model}")

            try:
                chat = self.get_client().chat.create(**create_params)
                final_response: Response | None = None

                async for response, chunk in chat.stream():
                    final_response = response
                    if chunk.content:
                        yield TextDeltaEvent(content=chunk.content)
                    if chunk.reasoning_content:
                        yield ThinkingDeltaEvent(content=chunk.reasoning_content)

                if final_response is None:
                    final_response = await chat.sample()

                elapsed = time.time() - start_time
                self.model_logger.debug(f"✅ xAI stream response in {elapsed:.2f}s")

                usage = self._get_usage(final_response)
                if usage and os.getenv("bu_agent_sdk_LLM_DEBUG"):
                    cached = usage.prompt_cached_tokens or 0
                    input_tokens = usage.prompt_tokens - cached
                    logger.info(
                        f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                    )

                yield CompletionDeltaEvent(
                    completion=ChatInvokeCompletion(
                        content=final_response.content or None,
                        tool_calls=self._extract_tool_calls(final_response),
                        thinking=final_response.reasoning_content or None,
                        redacted_thinking=final_response.encrypted_content or None,
                        usage=usage,
                        stop_reason=self._extract_stop_reason(final_response),
                    )
                )
                return
            except Exception as e:
                elapsed = time.time() - start_time
                status_code = self._infer_status_code(e) or 502
                error_message = self._error_message(e)
                self.model_logger.error(
                    f"💥 xAI stream call failed after {elapsed:.2f}s: status={status_code} error={error_message}"
                )

                if status_code == 429:
                    model_error: ModelProviderError = ModelRateLimitError(
                        message=error_message,
                        model=self.name,
                    )
                else:
                    model_error = ModelProviderError(
                        message=error_message,
                        status_code=status_code,
                        model=self.name,
                    )

                if (
                    model_error.status_code in self.retryable_status_codes
                    and attempt < self.max_retries - 1
                ):
                    delay = min(
                        self.retry_base_delay * (2**attempt),
                        self.retry_max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.model_logger.warning(
                        f"⚠️ xAI stream retry in {total_delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue

                raise model_error from e

        raise RuntimeError("Retry loop completed without return or exception")
