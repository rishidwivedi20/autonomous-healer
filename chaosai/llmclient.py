"""Which LLM client the planner and analyzer use. Both only call client.beta.messages.create(...), so every adapter
(Gemini, Mistral, OpenRouter) and the pool of them is a drop-in for the Anthropic SDK client."""
from __future__ import annotations


def make_client(provider: str, members=None):
    if provider == "pool":
        from .pool import PoolClient
        return PoolClient(members or [])
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient()
    if provider == "mistral":
        from .mistral import MistralClient
        return MistralClient()
    if provider == "openrouter":
        from .openrouter import OpenRouterClient
        return OpenRouterClient()
    import anthropic
    return anthropic.Anthropic()


def model_label(client, default: str) -> str:
    """The model(s) that actually answered: a pool knows which of its members did; a single client is just the configured model."""
    used = getattr(client, "used_models", None)
    return ", ".join(used) if used else default
