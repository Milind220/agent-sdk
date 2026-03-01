"""
Convenient access to preconfigured LLM model aliases.

Usage:
    from bu_agent_sdk import llm

    model = llm.openai_gpt_5
    model = llm.anthropic_claude_sonnet_4
    model = llm.google_gemini_2_5_pro
    model = llm.grok_4_latest
"""

import os
from typing import TYPE_CHECKING

from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.llm.google.chat import ChatGoogle
from bu_agent_sdk.llm.grok.chat import ChatGrok
from bu_agent_sdk.llm.openai.chat import ChatOpenAI

if TYPE_CHECKING:
    from bu_agent_sdk.llm.base import BaseChatModel


_MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # OpenAI
    "openai_gpt_4o": ("openai", "gpt-4o"),
    "openai_gpt_4o_mini": ("openai", "gpt-4o-mini"),
    "openai_gpt_4_1_mini": ("openai", "gpt-4.1-mini"),
    "openai_gpt_5": ("openai", "gpt-5"),
    "openai_gpt_5_mini": ("openai", "gpt-5-mini"),
    "openai_gpt_5_nano": ("openai", "gpt-5-nano"),
    "openai_o1": ("openai", "o1"),
    "openai_o1_mini": ("openai", "o1-mini"),
    "openai_o1_pro": ("openai", "o1-pro"),
    "openai_o3": ("openai", "o3"),
    "openai_o3_mini": ("openai", "o3-mini"),
    "openai_o3_pro": ("openai", "o3-pro"),
    "openai_o4_mini": ("openai", "o4-mini"),
    # Anthropic
    "anthropic_claude_sonnet_4": ("anthropic", "claude-sonnet-4-20250514"),
    # Google
    "google_gemini_2_0_flash": ("google", "gemini-2.0-flash"),
    "google_gemini_2_5_pro": ("google", "gemini-2.5-pro"),
    "google_gemini_2_5_flash": ("google", "gemini-2.5-flash"),
    "google_gemini_2_5_flash_lite": ("google", "gemini-2.5-flash-lite"),
    # xAI Grok
    "grok_4": ("grok", "grok-4"),
    "grok_4_latest": ("grok", "grok-4-latest"),
    "grok_3_latest": ("grok", "grok-3-latest"),
}


def _normalize_model_name(provider: str, model_part: str) -> str:
    if provider == "openai":
        model = model_part.replace("_", "-")
        model = model.replace("gpt-4-1", "gpt-4.1")
        return model

    if provider == "google":
        model = model_part.replace("gemini_2_0", "gemini-2.0")
        model = model.replace("gemini_2_5", "gemini-2.5")
        model = model.replace("_", "-")
        return model

    return model_part.replace("_", "-")


def _build_model(provider: str, model: str):
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return ChatGoogle(model=model, api_key=api_key)

    if provider == "grok":
        return ChatGrok(model=model, api_key=os.getenv("XAI_API_KEY"))

    raise ValueError(
        f"Unknown provider: '{provider}'. Available providers: openai, anthropic, google, grok"
    )


def get_llm_by_name(model_name: str):
    """Factory function to create LLM instances from model alias strings."""
    if not model_name:
        raise ValueError("Model name cannot be empty")

    if model_name in _MODEL_ALIASES:
        provider, resolved_model = _MODEL_ALIASES[model_name]
        return _build_model(provider, resolved_model)

    if "_" not in model_name:
        raise ValueError(
            f"Invalid model name format: '{model_name}'. Expected 'provider_model_name'."
        )

    provider, model_part = model_name.split("_", 1)
    if provider not in {"openai", "anthropic", "google", "grok"}:
        raise ValueError(
            f"Unknown provider: '{provider}'. Available providers: openai, anthropic, google, grok"
        )

    resolved_model = _normalize_model_name(provider, model_part)
    return _build_model(provider, resolved_model)


def __getattr__(name: str) -> "BaseChatModel":
    if name == "ChatOpenAI":
        return ChatOpenAI  # type: ignore
    if name == "ChatAnthropic":
        return ChatAnthropic  # type: ignore
    if name == "ChatGoogle":
        return ChatGoogle  # type: ignore
    if name == "ChatGrok":
        return ChatGrok  # type: ignore

    try:
        return get_llm_by_name(name)
    except ValueError as e:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from e


__all__ = [
    "ChatOpenAI",
    "ChatAnthropic",
    "ChatGoogle",
    "ChatGrok",
    "get_llm_by_name",
    *_MODEL_ALIASES.keys(),
]
