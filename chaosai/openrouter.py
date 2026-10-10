"""OpenRouter adapter: a ChatCompletionsClient pointed at https://openrouter.ai/api/v1, which fronts many models
(for example `openai/gpt-oss-120b`, or its `:free` variant) behind one OpenAI-style API and one key.

OpenRouter quirks handled here:
  * `provider.require_parameters` is set, so a request that uses tools or a JSON schema is only routed to an upstream
    provider that supports them (otherwise a model can silently ignore them);
  * reasoning models return a `reasoning` field that should be sent back on the next turn, so those fields are replayed;
  * the free variants have a per-minute limit and a daily limit; the daily one is recognised and fails at once with when
    it resets, instead of being retried;
  * `X-Title` and `HTTP-Referer` identify this app to OpenRouter (optional headers, no secrets).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .chatcompletions import ChatCompletionsClient, ChatCompletionsError

OpenRouterError = ChatCompletionsError
BASE_URL = "https://openrouter.ai/api/v1"
REFERER = "https://github.com/rishidwivedi20/autonomous-healer"


def api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or None


class OpenRouterClient(ChatCompletionsClient):
    PROVIDER = "OpenRouter"
    BASE_URL = BASE_URL
    KEY_ENV = "OPENROUTER_API_KEY"
    REPLAY_FIELDS = ("reasoning", "reasoning_details")

    def extra_headers(self) -> dict:
        return {"X-Title": "chaos-ai", "HTTP-Referer": REFERER}

    def extra_body(self) -> dict:
        return {"provider": {"require_parameters": True}}

    def final_429(self, r):
        text = r.text.lower()
        if "per-day" not in text and "per day" not in text:
            return None
        reset = r.headers.get("x-ratelimit-reset")                         # epoch milliseconds
        secs = max(0.0, float(reset) / 1000.0 - time.time()) if reset and reset.replace(".", "", 1).isdigit() else None
        return OpenRouterError("daily free-model request limit reached"
                               + (f"; resets in about {secs / 3600:.1f} h" if secs is not None else "")
                               + " (adding credits on openrouter.ai lifts it)", 429, secs if secs is not None else 6 * 3600)

    def list_models(self) -> list[str]:
        """Model ids that accept tools (the planner and analyzer need them); the list is long, so llm-check can filter it."""
        r = self._http.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {self._key}"})
        if r.status_code != 200:
            raise OpenRouterError(f"HTTP {r.status_code}: {self._error_text(r)}", r.status_code)
        return sorted(m["id"] for m in r.json().get("data", []) if "tools" in (m.get("supported_parameters") or []))
