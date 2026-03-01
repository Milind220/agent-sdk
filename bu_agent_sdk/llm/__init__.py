"""
LLM abstraction layer with type-safe tool calling support.

This module provides a unified interface for chat models across supported providers.
"""

from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

from bu_agent_sdk.llm.base import (
    BaseChatModel,
    SupportsModelCapabilities,
    SupportsStreamingInvoke,
    ToolChoice,
    ToolDefinition,
)
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    Function,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from bu_agent_sdk.llm.messages import ContentPartImageParam as ContentImage
from bu_agent_sdk.llm.messages import (
    ContentPartRedactedThinkingParam as ContentRedactedThinking,
)
from bu_agent_sdk.llm.messages import ContentPartRefusalParam as ContentRefusal
from bu_agent_sdk.llm.messages import ContentPartTextParam as ContentText
from bu_agent_sdk.llm.messages import ContentPartThinkingParam as ContentThinking
from bu_agent_sdk.llm.streaming import (
    CompletionDeltaEvent,
    ModelCapabilities,
    ModelDeltaEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallArgumentsDeltaEvent,
)
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage

if TYPE_CHECKING:
    from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
    from bu_agent_sdk.llm.google.chat import ChatGoogle
    from bu_agent_sdk.llm.grok.chat import ChatGrok
    from bu_agent_sdk.llm.openai.chat import ChatOpenAI
    from bu_agent_sdk.llm.openai.like import ChatOpenAILike

    openai_gpt_4o: ChatOpenAI
    openai_gpt_4o_mini: ChatOpenAI
    openai_gpt_4_1_mini: ChatOpenAI
    openai_gpt_5: ChatOpenAI
    openai_gpt_5_mini: ChatOpenAI
    openai_gpt_5_nano: ChatOpenAI

    anthropic_claude_sonnet_4: ChatAnthropic

    google_gemini_2_0_flash: ChatGoogle
    google_gemini_2_5_pro: ChatGoogle
    google_gemini_2_5_flash: ChatGoogle
    google_gemini_2_5_flash_lite: ChatGoogle

    grok_4: ChatGrok
    grok_4_latest: ChatGrok
    grok_3_latest: ChatGrok


_LAZY_IMPORTS = {
    "ChatAnthropic": ("bu_agent_sdk.llm.anthropic.chat", "ChatAnthropic"),
    "ChatGoogle": ("bu_agent_sdk.llm.google.chat", "ChatGoogle"),
    "ChatGrok": ("bu_agent_sdk.llm.grok.chat", "ChatGrok"),
    "ChatOpenAI": ("bu_agent_sdk.llm.openai.chat", "ChatOpenAI"),
    "ChatOpenAILike": ("bu_agent_sdk.llm.openai.like", "ChatOpenAILike"),
}

_model_cache: dict[str, "BaseChatModel"] = {}


def __getattr__(name: str):
    """Lazy import mechanism for chat model classes and model aliases."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module

            module = import_module(module_path)
            return getattr(module, attr_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}") from e

    if name in _model_cache:
        return _model_cache[name]

    try:
        from bu_agent_sdk.llm.models import __getattr__ as models_getattr

        attr = models_getattr(name)
        _model_cache[name] = attr
        return attr
    except (AttributeError, ImportError):
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "DeveloperMessage",
    "ToolCall",
    "Function",
    "ToolDefinition",
    "ToolChoice",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    "ContentText",
    "ContentRefusal",
    "ContentImage",
    "ContentThinking",
    "ContentRedactedThinking",
    "BaseChatModel",
    "SupportsStreamingInvoke",
    "SupportsModelCapabilities",
    "ModelCapabilities",
    "ModelDeltaEvent",
    "TextDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolCallArgumentsDeltaEvent",
    "CompletionDeltaEvent",
    "ChatOpenAI",
    "ChatOpenAILike",
    "ChatAnthropic",
    "ChatGoogle",
    "ChatGrok",
]
