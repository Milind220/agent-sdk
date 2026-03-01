import json

from xai_sdk import chat as xai_chat
from xai_sdk.proto import chat_pb2

from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


class GrokMessageSerializer:
    """Serializer for converting internal message types to xAI chat protobuf messages."""

    @staticmethod
    def _text_content(content: str | list) -> str:
        if isinstance(content, str):
            return content

        parts: list[str] = []
        for part in content:
            if part.type == "text":
                parts.append(part.text)
            elif part.type == "refusal":
                parts.append(f"[Refusal] {part.refusal}")
            elif part.type == "thinking":
                continue
            elif part.type == "redacted_thinking":
                continue

        return "\n".join(parts)

    @staticmethod
    def _serialize_user_message(message: UserMessage) -> chat_pb2.Message:
        if isinstance(message.content, str):
            return xai_chat.user(message.content)

        contents: list[chat_pb2.Content] = []
        for part in message.content:
            if part.type == "text" and part.text:
                contents.append(xai_chat.text(part.text))
            elif part.type == "image_url":
                contents.append(
                    xai_chat.image(
                        part.image_url.url,
                        detail=part.image_url.detail,
                    )
                )
            elif part.type == "document":
                contents.append(xai_chat.text("[PDF content omitted]"))

        return chat_pb2.Message(role=chat_pb2.ROLE_USER, content=contents)

    @staticmethod
    def _serialize_assistant_message(message: AssistantMessage) -> chat_pb2.Message:
        content = GrokMessageSerializer._text_content(message.content or "")
        content_parts = [xai_chat.text(content)] if content else []

        serialized = chat_pb2.Message(
            role=chat_pb2.ROLE_ASSISTANT,
            content=content_parts,
        )

        if message.tool_calls:
            for tool_call in message.tool_calls:
                serialized.tool_calls.append(
                    chat_pb2.ToolCall(
                        id=tool_call.id,
                        type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                        function=chat_pb2.FunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                )

        return serialized

    @staticmethod
    def _serialize_system_or_developer_message(
        message: SystemMessage | DeveloperMessage,
    ) -> chat_pb2.Message:
        content = GrokMessageSerializer._text_content(message.content)
        role = (
            chat_pb2.ROLE_SYSTEM
            if isinstance(message, SystemMessage)
            else chat_pb2.ROLE_DEVELOPER
        )
        parts = [xai_chat.text(content)] if content else []
        return chat_pb2.Message(role=role, content=parts)

    @staticmethod
    def _serialize_tool_message(message: ToolMessage) -> chat_pb2.Message:
        if message.destroyed:
            payload = "<removed to save context>"
        elif message.is_error:
            payload = f"[ERROR] {message.text}"
        elif isinstance(message.content, str):
            payload = message.content
        else:
            try:
                payload = json.dumps(message.content)
            except TypeError:
                payload = message.text

        return xai_chat.tool_result(payload, tool_call_id=message.tool_call_id)

    @staticmethod
    def serialize_messages(messages: list[BaseMessage]) -> list[chat_pb2.Message]:
        serialized: list[chat_pb2.Message] = []

        for message in messages:
            if isinstance(message, UserMessage):
                serialized.append(GrokMessageSerializer._serialize_user_message(message))
            elif isinstance(message, AssistantMessage):
                serialized.append(
                    GrokMessageSerializer._serialize_assistant_message(message)
                )
            elif isinstance(message, (SystemMessage, DeveloperMessage)):
                serialized.append(
                    GrokMessageSerializer._serialize_system_or_developer_message(message)
                )
            elif isinstance(message, ToolMessage):
                serialized.append(GrokMessageSerializer._serialize_tool_message(message))

        return serialized
