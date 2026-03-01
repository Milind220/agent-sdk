"""
Live smoke checks for chat models and Agent integration.

Usage:
    python -m bu_agent_sdk.examples.live_chat_model_smoke
    python -m bu_agent_sdk.examples.live_chat_model_smoke --provider grok
    python -m bu_agent_sdk.examples.live_chat_model_smoke --provider openai --model gpt-5

Environment variables:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY or GEMINI_API_KEY
    XAI_API_KEY
"""

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import FinalResponseEvent, TextEvent, ThinkingEvent
from bu_agent_sdk.llm import (
    ChatAnthropic,
    ChatGoogle,
    ChatGrok,
    ChatOpenAI,
    UserMessage,
)
from bu_agent_sdk.llm.base import BaseChatModel, SupportsStreamingInvoke
from bu_agent_sdk.llm.streaming import CompletionDeltaEvent, TextDeltaEvent, ThinkingDeltaEvent


@dataclass
class ProviderConfig:
    name: str
    env_key: str
    default_model: str
    factory: Callable[[str], BaseChatModel]


def _provider_configs() -> dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(
            name="openai",
            env_key="OPENAI_API_KEY",
            default_model="gpt-5",
            factory=lambda model: ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY")),
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            env_key="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-20250514",
            factory=lambda model: ChatAnthropic(
                model=model, api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
        ),
        "google": ProviderConfig(
            name="google",
            env_key="GOOGLE_API_KEY",
            default_model="gemini-2.5-flash",
            factory=lambda model: ChatGoogle(
                model=model,
                api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            ),
        ),
        "grok": ProviderConfig(
            name="grok",
            env_key="XAI_API_KEY",
            default_model="grok-4-latest",
            factory=lambda model: ChatGrok(model=model, api_key=os.getenv("XAI_API_KEY")),
        ),
    }


async def _smoke_ainvoke(model: BaseChatModel) -> None:
    response = await model.ainvoke(
        [UserMessage(content="Reply with exactly: pong")],
    )
    if not response.content:
        raise RuntimeError("ainvoke returned empty content")
    print(f"  ainvoke: ok ({response.content[:80]!r})")


async def _smoke_astream(model: BaseChatModel) -> None:
    if not isinstance(model, SupportsStreamingInvoke):
        print("  astream_invoke: skipped (adapter does not implement streaming)")
        return

    saw_completion = False
    text_chunks = 0
    thinking_chunks = 0

    async for delta in model.astream_invoke(
        [UserMessage(content="Reply with exactly: pong")],
    ):
        if isinstance(delta, TextDeltaEvent) and delta.content:
            text_chunks += 1
        elif isinstance(delta, ThinkingDeltaEvent) and delta.content:
            thinking_chunks += 1
        elif isinstance(delta, CompletionDeltaEvent):
            saw_completion = True

    if not saw_completion:
        raise RuntimeError("astream_invoke ended without CompletionDeltaEvent")
    print(
        f"  astream_invoke: ok ({text_chunks} text chunks, {thinking_chunks} thinking chunks)"
    )


async def _smoke_agent_stream(model: BaseChatModel) -> None:
    agent = Agent(llm=model, tools=[], max_iterations=2)

    text_events = 0
    thinking_events = 0
    saw_final = False
    async for event in agent.query_stream("Reply with exactly: pong"):
        if isinstance(event, TextEvent):
            text_events += 1
        elif isinstance(event, ThinkingEvent):
            thinking_events += 1
        elif isinstance(event, FinalResponseEvent):
            saw_final = True

    if not saw_final:
        raise RuntimeError("Agent query_stream ended without FinalResponseEvent")
    print(
        f"  agent.query_stream: ok ({text_events} text events, {thinking_events} thinking events)"
    )


def _get_key(config: ProviderConfig) -> str | None:
    if config.name == "google":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return os.getenv(config.env_key)


async def _run_provider(config: ProviderConfig, model_name: str | None) -> bool:
    key = _get_key(config)
    if not key:
        print(f"- {config.name}: skipped (missing key)")
        return False

    selected_model = model_name or config.default_model
    print(f"- {config.name}: model={selected_model}")
    model = config.factory(selected_model)

    await _smoke_ainvoke(model)
    await _smoke_astream(model)
    await _smoke_agent_stream(model)
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description="Live chat model smoke checks")
    parser.add_argument(
        "--provider",
        choices=["all", "openai", "anthropic", "google", "grok"],
        default="all",
        help="Provider to test",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model for single-provider runs",
    )
    args = parser.parse_args()

    configs = _provider_configs()
    targets = (
        list(configs.values())
        if args.provider == "all"
        else [configs[args.provider]]
    )

    if args.model and len(targets) != 1:
        raise SystemExit("--model override only valid with single --provider")

    ran_any = False
    failures = 0

    for config in targets:
        try:
            ran = await _run_provider(config, args.model)
            ran_any = ran_any or ran
        except Exception as e:
            failures += 1
            print(f"  ERROR: {type(e).__name__}: {e}")

    if not ran_any:
        print("No providers executed (no API keys found).")
        return 2
    if failures > 0:
        print(f"Done with {failures} provider failure(s).")
        return 1

    print("All executed provider checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

