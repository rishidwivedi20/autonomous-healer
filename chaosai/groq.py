"""Groq adapter (GroqCloud keys, including the free plan): a ChatCompletionsClient pointed at Groq, which serves open
models (for example `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`) very quickly. See chatcompletions.py for the
translation and the safety properties.

The free plan caps requests and tokens per minute and per day, per model. A 429 that asks for a long wait (a spent daily
allowance) fails at once with how long to wait, so the pool can move on instead of retrying.
"""
from __future__ import annotations

import os
from typing import Optional

from .chatcompletions import ChatCompletionsClient, ChatCompletionsError

GroqError = ChatCompletionsError
BASE_URL = "https://api.groq.com/openai/v1"


def api_key() -> Optional[str]:
    return os.environ.get("GROQ_API_KEY") or None


class GroqClient(ChatCompletionsClient):
    PROVIDER = "Groq"
    BASE_URL = BASE_URL
    KEY_ENV = "GROQ_API_KEY"
    MAX_WAIT_S = 70.0        # the free plan allows ~8000 tokens per minute per model, so a per-minute 429 is worth waiting out

    def list_models(self) -> list[str]:
        """Model ids this key can call (used by `chaos llm-check --models`); Groq does not say which support tools."""
        r = self._http.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {self._key}"})
        if r.status_code != 200:
            raise GroqError(f"HTTP {r.status_code}: {self._error_text(r)}", r.status_code)
        return sorted(m["id"] for m in r.json().get("data", []) if m.get("active", True))
