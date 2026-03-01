"""Streaming and capability primitives for chat models."""

from dataclasses import dataclass

from bu_agent_sdk.llm.views import ChatInvokeCompletion


@dataclass(frozen=True)
class ModelCapabilities:
    """Optional capability flags for a model adapter."""

    supports_token_stream: bool = False
    supports_reasoning_stream: bool = False
    supports_tool_argument_stream: bool = False
    supports_server_state: bool = False
    supports_image_input: bool = False


@dataclass(frozen=True)
class TextDeltaEvent:
    """Incremental text token/chunk."""

    content: str


@dataclass(frozen=True)
class ThinkingDeltaEvent:
    """Incremental reasoning/thinking chunk."""

    content: str


@dataclass(frozen=True)
class ToolCallArgumentsDeltaEvent:
    """Incremental tool arguments chunk."""

    tool_call_id: str
    arguments_delta: str


@dataclass(frozen=True)
class CompletionDeltaEvent:
    """Final normalized completion for a streamed invocation."""

    completion: ChatInvokeCompletion


ModelDeltaEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolCallArgumentsDeltaEvent
    | CompletionDeltaEvent
)

