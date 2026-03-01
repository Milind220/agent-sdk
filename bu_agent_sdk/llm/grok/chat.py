import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from xai_sdk import AsyncClient
from xai_sdk import chat as xai_chat
from xai_sdk.chat import Response
from xai_sdk.proto import chat_pb2

from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from bu_agent_sdk.llm.grok.serializer import GrokMessageSerializer
from bu_agent_sdk.llm.messages import BaseMessage, Function, ToolCall
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage


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

    api_key: str | None = None
    api_host: str = "api.x.ai"
    timeout: float | None = None
    metadata: tuple[tuple[str, str], ...] | None = None
    channel_options: list[tuple[str, Any]] | None = None
    use_insecure_channel: bool = False

    _client: AsyncClient | None = None

    @property
    def provider(self) -> str:
        return "grok"

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_client_params(self) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("XAI_API_KEY")

        params: dict[str, Any] = {
            "api_key": api_key,
            "api_host": self.api_host,
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

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        serialized_messages = GrokMessageSerializer.serialize_messages(messages)

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

        if tools:
            create_params["tools"] = self._serialize_tools(tools)
            selected_tool_choice = self._get_tool_choice(tool_choice, tools)
            if selected_tool_choice is not None:
                create_params["tool_choice"] = selected_tool_choice

        if kwargs:
            create_params.update(kwargs)

        try:
            chat = self.get_client().chat.create(**create_params)
            response = await chat.sample()

            content = response.content or None
            tool_calls = self._extract_tool_calls(response)
            thinking = response.reasoning_content or None
            usage = self._get_usage(response)

            return ChatInvokeCompletion(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
                usage=usage,
                stop_reason=self._extract_stop_reason(response),
            )
        except Exception as e:
            error_message = str(e)
            lowered = error_message.lower()

            if "rate limit" in lowered or "resource exhausted" in lowered or "429" in lowered:
                raise ModelRateLimitError(message=error_message, model=self.name) from e

            status_code: int | None = None
            for code in (400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504):
                if f"{code}" in lowered:
                    status_code = code
                    break

            raise ModelProviderError(
                message=error_message,
                status_code=status_code or 502,
                model=self.name,
            ) from e
