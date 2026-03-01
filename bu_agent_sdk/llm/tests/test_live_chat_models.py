"""
Optional live API tests.

Run only when explicitly enabled:
    RUN_LIVE_TESTS=1 pytest bu_agent_sdk/llm/tests/test_live_chat_models.py
"""

import os

import pytest

from bu_agent_sdk.llm import ChatAnthropic, ChatGoogle, ChatGrok, ChatOpenAI, UserMessage
from bu_agent_sdk.llm.base import SupportsStreamingInvoke
from bu_agent_sdk.llm.streaming import CompletionDeltaEvent


if os.getenv("RUN_LIVE_TESTS") != "1":
    pytestmark = pytest.mark.skip(reason="Set RUN_LIVE_TESTS=1 to run live tests")


@pytest.mark.anyio
async def test_live_openai_ainvoke() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5"), api_key=api_key)
    response = await llm.ainvoke([UserMessage(content="Reply with exactly: pong")])
    assert response.content


@pytest.mark.anyio
async def test_live_anthropic_ainvoke() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        api_key=api_key,
    )
    response = await llm.ainvoke([UserMessage(content="Reply with exactly: pong")])
    assert response.content


@pytest.mark.anyio
async def test_live_google_ainvoke() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    llm = ChatGoogle(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        api_key=api_key,
    )
    response = await llm.ainvoke([UserMessage(content="Reply with exactly: pong")])
    assert response.content


@pytest.mark.anyio
async def test_live_grok_streaming_completion_event() -> None:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY not set")

    llm = ChatGrok(
        model=os.getenv("XAI_MODEL", "grok-4-latest"),
        api_key=api_key,
    )
    assert isinstance(llm, SupportsStreamingInvoke)

    saw_completion = False
    async for delta in llm.astream_invoke(
        [UserMessage(content="Reply with exactly: pong")]
    ):
        if isinstance(delta, CompletionDeltaEvent):
            saw_completion = True

    assert saw_completion

