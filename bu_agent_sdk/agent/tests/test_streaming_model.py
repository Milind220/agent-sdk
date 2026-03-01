from dataclasses import dataclass

import pytest

from bu_agent_sdk.agent import Agent, FinalResponseEvent, TextEvent
from bu_agent_sdk.llm.base import ToolChoice, ToolDefinition
from bu_agent_sdk.llm.messages import BaseMessage
from bu_agent_sdk.llm.streaming import CompletionDeltaEvent, TextDeltaEvent
from bu_agent_sdk.llm.views import ChatInvokeCompletion


@dataclass
class _FakeStreamingModel:
    model: str = "fake-stream-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        return ChatInvokeCompletion(content="fallback")

    async def astream_invoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ):
        yield TextDeltaEvent(content="hel")
        yield TextDeltaEvent(content="lo")
        yield CompletionDeltaEvent(
            completion=ChatInvokeCompletion(content="hello", tool_calls=[])
        )


@pytest.mark.anyio
async def test_query_stream_uses_streaming_model_deltas_without_duplication() -> None:
    agent = Agent(llm=_FakeStreamingModel(), tools=[])

    events = [event async for event in agent.query_stream("hi")]

    text_events = [event for event in events if isinstance(event, TextEvent)]
    final_events = [event for event in events if isinstance(event, FinalResponseEvent)]

    assert [event.content for event in text_events] == ["hel", "lo"]
    assert len(final_events) == 1
    assert final_events[0].content == "hello"
