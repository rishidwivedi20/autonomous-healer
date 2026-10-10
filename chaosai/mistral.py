"""Mistral adapter (La Plateforme keys, including the free Experiment tier): a ChatCompletionsClient pointed at
Mistral, with Mistral's two quirks. See chatcompletions.py for the translation and the safety properties.

  * tool-call ids must be exactly 9 alphanumeric characters (so ids from another provider are mapped when the pool
    fails over into Mistral mid-run);
  * a key whose plan allows 0 requests per minute answers every chat call with HTTP 429 and
    `x-ratelimit-limit-req-minute: 0`: waiting cannot help, so that fails at once with the instruction to fix it.
"""
from __future__ import annotations

import os
from typing import Optional

from .chatcompletions import ChatCompletionsClient, ChatCompletionsError

MistralError = ChatCompletionsError
BASE_URL = "https://api.mistral.ai/v1"
MODEL_PREFIXES = ("mistral", "ministral", "magistral", "codestral", "devstral", "pixtral", "open-mistral", "open-mixtral")


def api_key() -> Optional[str]:
    return os.environ.get("MISTRAL_API_KEY") or None


class MistralClient(ChatCompletionsClient):
    PROVIDER = "Mistral"
    BASE_URL = BASE_URL
    KEY_ENV = "MISTRAL_API_KEY"
    NORMALIZE_IDS = True

    def final_429(self, r):
        if r.headers.get("x-ratelimit-limit-req-minute") == "0":
            return MistralError("this key's plan allows 0 requests per minute, so no call can succeed: activate the free Experiment "
                                "plan (or add billing) in the Mistral console (admin.mistral.ai, Billing), then retry", 429, 6 * 3600)
        return None

    def list_models(self) -> list[str]:
        """Chat models this key can use that support function calling (used by `chaos llm-check`)."""
        r = self._http.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {self._key}"})
        if r.status_code != 200:
            raise MistralError(f"HTTP {r.status_code}: {self._error_text(r)}", r.status_code)
        out = []
        for m in r.json().get("data", []):
            caps = m.get("capabilities") or {}
            if caps.get("completion_chat", True) and caps.get("function_calling", True):
                out.append(m["id"])
        return sorted(set(out))
