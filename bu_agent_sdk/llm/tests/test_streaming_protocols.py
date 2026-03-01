from dataclasses import dataclass

from pydantic import BaseModel

from bu_agent_sdk.llm.base import BaseChatModel, SupportsStreamingInvoke
from bu_agent_sdk.llm.messages import BaseMessage
from bu_agent_sdk.llm.streaming import CompletionDeltaEvent
from bu_agent_sdk.llm.views import ChatInvokeCompletion


@dataclass
class _StreamingOnlyAdapter:
    async def astream_invoke(
        self,
        messages,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        yield CompletionDeltaEvent(completion=ChatInvokeCompletion(content="ok"))


def test_streaming_protocol_isinstance_for_streaming_only_adapter() -> None:
    adapter = _StreamingOnlyAdapter()
    assert isinstance(adapter, SupportsStreamingInvoke)


@dataclass
class _BaseChatAdapter:
    model: str = "fake-model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        return ChatInvokeCompletion(content="ok")


class _ModelWithLLM(BaseModel):
    llm: BaseChatModel


def test_base_chat_model_protocol_still_works_in_pydantic_model() -> None:
    wrapped = _ModelWithLLM(llm=_BaseChatAdapter())
    assert wrapped.llm is not None

