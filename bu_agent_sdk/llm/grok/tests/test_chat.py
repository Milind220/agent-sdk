from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from bu_agent_sdk.llm.grok.chat import ChatGrok
from bu_agent_sdk.llm.messages import UserMessage


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    cached_prompt_text_tokens: int = 2
    prompt_image_tokens: int = 0
    completion_tokens: int = 4
    reasoning_tokens: int = 1
    total_tokens: int = 15


@dataclass
class FakeResponse:
    content: str = "ok"
    reasoning_content: str = "thinking"
    encrypted_content: str = "encrypted"
    finish_reason: str = "FINISH_REASON_STOP"
    tool_calls: list = None
    usage: FakeUsage = field(default_factory=FakeUsage)

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


class _FakeChatSession:
    def __init__(self, responses: list[object]):
        self._responses = responses

    async def sample(self):
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeChatEndpoint:
    def __init__(self, responses: list[object]):
        self.responses = responses
        self.last_create_kwargs = None

    def create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return _FakeChatSession(self.responses)


class _FakeClient:
    def __init__(self, responses: list[object]):
        self.chat = _FakeChatEndpoint(responses)


@pytest.mark.anyio
async def test_grok_retries_then_succeeds() -> None:
    model = ChatGrok(model="grok-4-latest", max_retries=2, retry_base_delay=0)
    fake_client = _FakeClient([Exception("429 too many requests"), FakeResponse()])
    model.get_client = lambda: fake_client  # type: ignore[method-assign]

    result = await model.ainvoke([UserMessage(content="hello")])

    assert result.content == "ok"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10


@pytest.mark.anyio
async def test_grok_create_params_include_parity_fields() -> None:
    model = ChatGrok(
        model="grok-4-latest",
        max_tokens=123,
        temperature=0.7,
        top_p=0.9,
        seed=7,
        parallel_tool_calls=False,
        reasoning_effort="low",
        frequency_penalty=0.1,
        presence_penalty=0.2,
        stop=["done"],
        user="u-1",
        logprobs=True,
        top_logprobs=3,
        include=["inline_citations"],
        store_messages=True,
        previous_response_id="resp_123",
        use_encrypted_content=True,
        max_turns=5,
        conversation_id="conv_1",
    )
    fake_client = _FakeClient([FakeResponse()])
    model.get_client = lambda: fake_client  # type: ignore[method-assign]

    await model.ainvoke([UserMessage(content="hello")])

    create_kwargs = fake_client.chat.last_create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["max_tokens"] == 123
    assert create_kwargs["temperature"] == 0.7
    assert create_kwargs["top_p"] == 0.9
    assert create_kwargs["seed"] == 7
    assert create_kwargs["parallel_tool_calls"] is False
    assert create_kwargs["reasoning_effort"] == "low"
    assert create_kwargs["frequency_penalty"] == 0.1
    assert create_kwargs["presence_penalty"] == 0.2
    assert create_kwargs["stop"] == ["done"]
    assert create_kwargs["user"] == "u-1"
    assert create_kwargs["logprobs"] is True
    assert create_kwargs["top_logprobs"] == 3
    assert create_kwargs["include"] == ["inline_citations"]
    assert create_kwargs["store_messages"] is True
    assert create_kwargs["previous_response_id"] == "resp_123"
    assert create_kwargs["use_encrypted_content"] is True
    assert create_kwargs["max_turns"] == 5
    assert create_kwargs["conversation_id"] == "conv_1"
