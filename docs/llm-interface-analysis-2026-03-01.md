# LLM Interface Analysis (2026-03-01)

Scope: `bu_agent_sdk` chat abstraction vs current provider SDK capabilities (OpenAI, Anthropic, xAI, Google).

Date basis: 2026-03-01.

---

## Live versions checked (GitHub releases)

- `anthropics/anthropic-sdk-python`: `v0.84.0` (2026-02-25)
- `xai-org/xai-sdk-python`: `v1.7.0` (2026-02-18)
- `openai/openai-python`: `v2.24.0` (2026-02-24)
- `googleapis/python-genai`: `v1.65.0` (2026-02-26)

Repo pins match these minimums in `pyproject.toml`.

---

## Current SDK baseline

### Interface shape today

- Single protocol: `BaseChatModel.ainvoke(...) -> ChatInvokeCompletion`.
- No streaming method in model interface.
- No first-class state handle (`response_id`, `conversation_id`) in interface return.

Refs:
- `bu_agent_sdk/llm/base.py`
- `bu_agent_sdk/llm/views.py`

### Provider adapters today

- OpenAI adapter: uses `chat.completions.create(...)` (not `responses`).
- Anthropic adapter: uses `messages.create(...)` non-streaming.
- Google adapter: uses `models.generate_content(...)` non-streaming.
- Grok adapter: uses `chat.create(...).sample()` non-streaming.

Refs:
- `bu_agent_sdk/llm/openai/chat.py`
- `bu_agent_sdk/llm/anthropic/chat.py`
- `bu_agent_sdk/llm/google/chat.py`
- `bu_agent_sdk/llm/grok/chat.py`

### Agent streaming today

- `agent.query_stream()` streams *agent loop events* (tool calls/results, step events).
- Not token-level model streaming.
- `TextEvent` emitted after model call completion, not per token delta.

Refs:
- `bu_agent_sdk/agent/service.py`
- `bu_agent_sdk/agent/events.py`

---

## Common features current interface misses

- [x] Token-level text streaming deltas. *(implemented for xAI + agent integration on this branch)*
- [x] Reasoning/thinking streaming deltas. *(implemented for xAI + agent integration on this branch)*
- [ ] Tool-argument partial JSON streaming deltas.
- [x] Stream lifecycle event normalization (minimum shape via `CompletionDeltaEvent`).
- [x] Optional server-managed conversation state abstraction. *(capability flags + optional interfaces added; full session APIs pending)*
- [ ] Response lifecycle APIs (store/retrieve/delete/cancel) abstraction.

### Implementation progress (branch: `codex/chat-model-upgrades`)

- [x] Added optional capability + streaming interfaces (`SupportsModelCapabilities`, `SupportsStreamingInvoke`).
- [x] Added normalized stream event primitives (`TextDeltaEvent`, `ThinkingDeltaEvent`, `CompletionDeltaEvent`).
- [x] Implemented `ChatGrok.astream_invoke(...)` using xAI SDK stream path.
- [x] Integrated `Agent.query_stream(...)` with model streaming when adapter supports it.
- [ ] OpenAI adapter move to Responses API.
- [ ] Anthropic streaming adapter integration.
- [ ] Google streaming adapter integration.
- [ ] Tool-arg delta normalization across providers.

---

## Provider capability snapshot

## OpenAI (`openai-python`)

- Chat Completions still present.
- Responses API richer: `responses.create/stream/retrieve/delete/cancel/input_items`.
- Streaming events include text deltas, reasoning deltas, function-arg deltas, completion events.
- Stateful flow via `previous_response_id`, `conversation`, `store`.

Implication:
- Current adapter leaves capability on table by staying on `chat.completions`.

## Anthropic (`anthropic-sdk-python`)

- Streaming surfaces: `messages.create(stream=True)` and managed `messages.stream(...)`.
- Stream supports text delta, tool input JSON delta, thinking delta, signature delta, final message helpers.
- Extended thinking + redacted thinking already modeled in your SDK; only non-stream path implemented now.
- Beta adds `container`, `context_management`, `mcp_servers`, `tool_runner` (provider-specific).

Implication:
- Token/reasoning stream feasible now without major model redesign.

## xAI (`xai-sdk-python`)

- SDK chat session methods: `sample()` and `stream()`.
- `sample()` is valid official non-stream call in this SDK.
- `stream()` yields incremental chunks (content + reasoning content).
- Supports stateful-ish fields in create params: `store_messages`, `previous_response_id`, `conversation_id`, `max_turns`, `use_encrypted_content`.
- Server-side tools exposed separately (`xai_sdk.tools`: web/x/code/collections/mcp).

Implication:
- Adapter not "entirely wrong" for using `sample()`; it is just non-stream path only.
- Big gap is missing `stream()` integration + capability gating by model features.

## Google (`python-genai`)

- Stateless: `models.generate_content`, `models.generate_content_stream`.
- Stateful chat sessions: `client.chats.create()` + `send_message`, `send_message_stream`, `get_history`.
- Thinking config + thought parts/signatures supported.
- AFC/MCP/live APIs add significant provider-specific behavior.

Implication:
- You currently use only stateless non-stream path.

---

## About your xAI error (image input unsupported)

Observed error:

```text
Invalid request content: Image inputs are not supported by this model.
```

Most likely root cause:
- Request included image part (`image_url` content block), but selected model lacks image input capability.
- This is model capability mismatch, not necessarily adapter method-name bug.

xAI docs explicitly split models by input capabilities; only image-input-capable models accept image content.

---

## Recommendation (simple for users + simple for maintainers)

Short answer: **do not fork into many unrelated model classes yet**. Keep one base chat model contract, add optional capability interfaces.

### Suggested shape

1. Keep `BaseChatModel` as minimal stable baseline
   - `ainvoke(...)` remains.

2. Add optional streaming protocol
   - Example: `astream_invoke(...) -> AsyncIterator[ModelDeltaEvent]`.
   - Agent `query_stream()` consumes this when available; fallback to current iteration-level events when unavailable.

3. Add optional stateful protocol
   - Example: `create_session(...)`, `invoke_in_session(...)` or pass/return `state_handle`.
   - Keep optional; avoid forcing all providers.

4. Add explicit capability flags
   - `supports_token_stream`, `supports_reasoning_stream`, `supports_server_state`, `supports_image_input`, etc.
   - Use for runtime guardrails + better errors.

5. Keep provider-specific extras behind `provider_options` / extension fields
   - Don’t pollute base interface with every provider novelty.

### Why this over “many separate model types now”

- Avoids hard split too early.
- Preserves current API ergonomics.
- Lets you ship token streaming first (your priority) with minimum API churn.
- Supports incremental adoption by provider.

---

## Concrete phased plan

### Phase 1 (high value, low churn)

- Add streaming interface + normalized delta event types.
- Implement streaming in:
  - OpenAI adapter (prefer Responses stream path).
  - Anthropic adapter (`messages.stream`).
  - Google adapter (`generate_content_stream`).
  - Grok adapter (`chat.stream`).
- Wire `Agent.query_stream()` to consume model deltas.

### Phase 2 (stateful where worth it)

- OpenAI: add optional `previous_response_id`/`conversation` flow in adapter.
- xAI: expose/validate `store_messages`, `previous_response_id`, `conversation_id` flow.
- Google: optional `ChatSession` wrapper backed by `client.chats.create()`.
- Anthropic: keep client-managed history by default; treat beta state features as opt-in experimental path.

### Phase 3 (provider-specialized)

- Add provider extension namespaces for advanced features (AFC, MCP server tools, lifecycle ops).

---

## Specific call on OpenAI/xAI “Responses vs old chat”

- OpenAI: yes, migrate primary adapter path toward Responses API.
- xAI: current Python SDK’s official call surface is `chat.create(...).sample()/stream()`. Keep using official SDK surface; add stream + state support there.
- If you later need strict OpenAI-like Responses parity for xAI, add separate adapter (opt-in), not forced replacement.

---

## Sources

## Live release sources

- https://github.com/anthropics/anthropic-sdk-python/releases/tag/v0.84.0
- https://github.com/xai-org/xai-sdk-python/releases/tag/v1.7.0
- https://github.com/openai/openai-python/releases/tag/v2.24.0
- https://github.com/googleapis/python-genai/releases/tag/v1.65.0

## Provider docs / capability docs

- xAI models/capabilities: https://docs.x.ai/developers/models
- OpenAI SDK docs (DeepWiki index): https://deepwiki.com/openai/openai-python#4.2
- Anthropic SDK streaming/docs (DeepWiki index): https://deepwiki.com/anthropics/anthropic-sdk-python#6
- xAI SDK chat/docs (DeepWiki index): https://deepwiki.com/xai-org/xai-sdk-python#3
- Google GenAI SDK docs (DeepWiki index): https://deepwiki.com/googleapis/python-genai#3.4

## Local code refs

- `bu_agent_sdk/llm/base.py`
- `bu_agent_sdk/llm/views.py`
- `bu_agent_sdk/llm/openai/chat.py`
- `bu_agent_sdk/llm/anthropic/chat.py`
- `bu_agent_sdk/llm/google/chat.py`
- `bu_agent_sdk/llm/grok/chat.py`
- `bu_agent_sdk/agent/service.py`
- `bu_agent_sdk/agent/events.py`
- `pyproject.toml`
