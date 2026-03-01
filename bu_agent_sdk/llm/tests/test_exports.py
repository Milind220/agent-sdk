import bu_agent_sdk.llm as llm


def test_chat_exports_are_importable() -> None:
    for name in [
        "ChatOpenAI",
        "ChatOpenAILike",
        "ChatAnthropic",
        "ChatGoogle",
        "ChatGrok",
    ]:
        assert getattr(llm, name) is not None


def test_no_stale_chat_exports() -> None:
    stale_names = {
        "ChatBrowserUse",
        "ChatDeepSeek",
        "ChatAnthropicBedrock",
        "ChatAWSBedrock",
        "ChatMistral",
        "ChatAzureOpenAI",
        "ChatOCIRaw",
        "ChatOllama",
        "ChatOpenRouter",
        "ChatVercel",
        "ChatCerebras",
    }

    for name in stale_names:
        assert name not in llm.__all__
