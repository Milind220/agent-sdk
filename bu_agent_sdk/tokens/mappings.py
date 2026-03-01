# Mapping from model_name to LiteLLM model name
# Only needed for models with non-standard naming conventions

MODEL_TO_LITELLM: dict[str, str] = {
	'gemini-flash-latest': 'gemini/gemini-flash-latest',
	'grok-4': 'xai/grok-4',
	'grok-4-latest': 'xai/grok-4-latest',
	'grok-3-latest': 'xai/grok-3-latest',
}
