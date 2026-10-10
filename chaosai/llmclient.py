"""Which LLM client the planner and analyzer use. Both only call client.beta.messages.create(...), so the
Gemini adapter (chaosai/gemini.py) is a drop-in for the Anthropic SDK client."""
from __future__ import annotations


def make_client(provider: str):
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient()
    import anthropic
    return anthropic.Anthropic()
