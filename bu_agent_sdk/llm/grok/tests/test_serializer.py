from bu_agent_sdk.llm.grok.serializer import GrokMessageSerializer
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    ImageURL,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def test_serialize_user_message_text_and_image() -> None:
    message = UserMessage(
        content=[
            ContentPartTextParam(text="look"),
            ContentPartImageParam(
                image_url=ImageURL(url="https://example.com/image.png", detail="low")
            ),
        ]
    )

    serialized = GrokMessageSerializer.serialize_messages([message])
    assert len(serialized) == 1
    assert serialized[0].role == 1  # ROLE_USER
    assert len(serialized[0].content) == 2
    assert serialized[0].content[0].text == "look"


def test_serialize_assistant_tool_calls() -> None:
    message = AssistantMessage(
        content="Calling tool",
        tool_calls=[
            ToolCall(
                id="call_1",
                function=Function(name="get_weather", arguments='{"city":"Tokyo"}'),
            )
        ],
    )

    serialized = GrokMessageSerializer.serialize_messages([message])
    assert len(serialized) == 1
    assert serialized[0].role == 2  # ROLE_ASSISTANT
    assert len(serialized[0].tool_calls) == 1
    assert serialized[0].tool_calls[0].function.name == "get_weather"


def test_serialize_tool_message_error() -> None:
    message = ToolMessage(
        tool_call_id="call_1",
        tool_name="get_weather",
        content="failed",
        is_error=True,
    )

    serialized = GrokMessageSerializer.serialize_messages([message])
    assert len(serialized) == 1
    assert serialized[0].role == 5  # ROLE_TOOL
    assert serialized[0].tool_call_id == "call_1"
    assert "[ERROR]" in serialized[0].content[0].text
