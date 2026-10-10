"""Gemini adapter: the small slice of the Anthropic client that the planner and analyzer use, backed by the
Gemini REST API (Google AI Studio keys).

The planner and analyzer call `client.beta.messages.create(...)` and read `.content` blocks (`text` /
`tool_use`), `.stop_reason` and `.usage`. GeminiClient accepts exactly those calls, translates them to
`generateContent`, and returns objects of the same shape, so the tool loop, the evidence validator, the
caches and the rule-based fallback work unchanged. Nothing here decides anything: a model answer still has to
pass the same validation as an Anthropic one, and any failure raises (callers fall back to the rules).

Plain HTTPS through httpx, which the project already depends on: no new dependency. The key travels in the
`x-goog-api-key` header, never in the URL, and is never logged or put into an error message.

Translation notes
  * system prompt          -> systemInstruction
  * tools                  -> one functionDeclarations list; JSON Schema is reduced to the subset Gemini accepts
                              (no additionalProperties; a ["boolean", "null"] union becomes nullable)
  * tool_use / tool_result -> functionCall / functionResponse parts. The model's own parts are replayed verbatim,
                              because Gemini 3 requires its `thoughtSignature` to come back unchanged
  * output_config.format   -> responseMimeType + responseSchema (the planner's structured JSON)
  * effort                 -> thinkingConfig.thinkingLevel (low / medium / high); if the model rejects the field the
                              call is repeated without it. Thinking tokens count against max_tokens
  * betas, fallbacks are Anthropic-specific and ignored
  * 429 / 5xx              -> retried with backoff (honouring the server's suggested delay), then GeminiError
"""
from __future__ import annotations

import json
import os
import re
import time
from types import SimpleNamespace
from typing import Callable, Optional

import httpx

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_BACKOFF_S = 30.0
THINKING_LEVEL = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
REFUSAL_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY", "LANGUAGE"}
_SCHEMA_KEYS = {"type", "format", "description", "nullable", "enum", "properties", "required", "items",
                "minItems", "maxItems", "minimum", "maximum"}


class GeminiError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s       # how long a caller should leave this model alone, when known


def api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def to_gemini_schema(s):
    """JSON Schema -> the OpenAPI subset Gemini's functionDeclarations / responseSchema accept."""
    if isinstance(s, list):
        return [to_gemini_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out: dict = {}
    for k, v in s.items():
        if k == "type" and isinstance(v, list):
            real = [t for t in v if t != "null"]
            out["type"] = real[0] if real else "string"
            if "null" in v:
                out["nullable"] = True
        elif k == "properties":
            out[k] = {name: to_gemini_schema(p) for name, p in v.items()}     # property NAMES are never filtered
        elif k == "items":
            out[k] = to_gemini_schema(v)
        elif k in _SCHEMA_KEYS:
            out[k] = v
    return out


def _get(b, name, default=None):
    return b.get(name, default) if isinstance(b, dict) else getattr(b, name, default)


class GeminiClient:
    """Duck-types the parts of anthropic.Anthropic() the planner and analyzer use."""

    def __init__(self, key: Optional[str] = None, transport: Optional[httpx.BaseTransport] = None,
                 sleep: Callable[[float], None] = time.sleep, max_retries: int = 4, timeout_s: float = 120.0):
        self._key = key or api_key()
        if not self._key:
            raise GeminiError("no Gemini credentials found (set GEMINI_API_KEY in .env or the environment)")
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self._sleep, self.max_retries = sleep, max_retries
        self._names: dict[str, str] = {}          # tool_use id -> function name, to label functionResponse parts
        self._n = 0
        self.calls = 0
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    # ------------------------------------------------------------ request
    def _contents(self, messages: list[dict]) -> list[dict]:
        for m in messages:                          # function names come from the history, so tool calls another provider issued work too
            if m["role"] == "assistant" and not isinstance(m["content"], str):
                for b in m["content"]:
                    if _get(b, "type") == "tool_use":
                        self._names[_get(b, "id")] = _get(b, "name")
        out = []
        for m in messages:
            role, content = "model" if m["role"] == "assistant" else "user", m["content"]
            if isinstance(content, str):
                parts = [{"text": content}]
            else:
                parts = []
                for b in content:
                    raw = getattr(b, "_raw", None) if not isinstance(b, dict) else None
                    kind = _get(b, "type")
                    if raw is not None:                                   # our own earlier output: replay verbatim
                        parts.append(raw)
                    elif kind == "text":
                        parts.append({"text": _get(b, "text") or ""})
                    elif kind == "tool_use":
                        parts.append({"functionCall": {"name": _get(b, "name"), "args": _get(b, "input") or {}}})
                    elif kind == "tool_result":
                        body = _get(b, "content")
                        try:
                            body = json.loads(body) if isinstance(body, str) else body
                        except ValueError:
                            pass
                        if _get(b, "is_error"):
                            body = {"error": body}
                        elif not isinstance(body, dict):
                            body = {"result": body}
                        parts.append({"functionResponse": {"name": self._names.get(_get(b, "tool_use_id"), "tool"), "response": body}})
            out.append({"role": role, "parts": parts or [{"text": " "}]})
        return out

    def _body(self, max_tokens, messages, system, tools, output_config, thinking: bool = True) -> dict:
        body: dict = {"contents": self._contents(messages), "generationConfig": {"maxOutputTokens": max_tokens}}
        level = THINKING_LEVEL.get((output_config or {}).get("effort", ""))
        if thinking and level:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": level}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t.get("description", ""), "parameters": to_gemini_schema(t["input_schema"])}
                for t in tools]}]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        fmt = (output_config or {}).get("format")
        if fmt and fmt.get("type") == "json_schema":
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = to_gemini_schema(fmt["schema"])
        return body

    # ------------------------------------------------------------ transport
    @staticmethod
    def _error_text(r: httpx.Response) -> tuple[str, str]:
        try:
            e = r.json().get("error", {})
            return str(e.get("status", "")), str(e.get("message", ""))[:300]
        except (ValueError, AttributeError):
            return "", r.text[:300]

    @staticmethod
    def _delay(r: httpx.Response, attempt: int) -> float:
        after = r.headers.get("retry-after")
        if after and after.replace(".", "", 1).isdigit():
            return min(float(after), MAX_BACKOFF_S)
        m = re.search(r'"retryDelay":\s*"([\d.]+)s"', r.text)           # Gemini's RetryInfo detail
        if m:
            return min(float(m.group(1)) + 0.5, MAX_BACKOFF_S)
        return min(2.0 ** (attempt + 1), MAX_BACKOFF_S)

    @staticmethod
    def _quota_exhausted(r: httpx.Response) -> Optional[GeminiError]:
        """A 429 whose suggested wait is longer than we would ever sleep is a spent quota (e.g. the free tier's
        daily request limit), not a burst: retrying is pointless, so say what ran out and when it resets."""
        m = re.search(r'"retryDelay":\s*"([\d.]+)s"', r.text)
        if r.status_code != 429 or not m or float(m.group(1)) <= MAX_BACKOFF_S:
            return None
        qid = re.search(r'"quotaId":\s*"([^"]+)"', r.text)
        qv = re.search(r'"quotaValue":\s*"?(\d+)', r.text)
        qm = re.search(r'"model":\s*"([^"]+)"', r.text)
        secs = float(m.group(1))
        return GeminiError("quota exhausted (" + (qid.group(1) if qid else "unknown quota") + (f", limit {qv.group(1)}" if qv else "") + ")"
                           + (f" for model {qm.group(1)}" if qm else "") + f"; resets in about {secs / 3600:.1f} h", 429, secs)

    def _post(self, model: str, body: dict) -> dict:
        url = f"{BASE_URL}/models/{model.removeprefix('models/')}:generateContent"
        headers = {"x-goog-api-key": self._key, "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            try:
                r = self._http.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    self._sleep(min(2.0 ** (attempt + 1), MAX_BACKOFF_S))
                    continue
                raise GeminiError(f"cannot reach the Gemini API ({type(e).__name__})", retry_after_s=60) from None
            if r.status_code == 200:
                return r.json()
            spent = self._quota_exhausted(r)
            if spent:
                raise spent
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(self._delay(r, attempt))
                continue
            status, message = self._error_text(r)
            raise GeminiError(f"HTTP {r.status_code} {status}: {message}".strip(), r.status_code)
        raise GeminiError("retries exhausted")                           # unreachable; keeps type checkers calm

    # ------------------------------------------------------------ response
    def _response(self, data: dict, model: str):
        usage = data.get("usageMetadata") or {}
        u = SimpleNamespace(input_tokens=usage.get("promptTokenCount", 0) or 0,
                            output_tokens=(usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0))

        def done(content, stop):
            return SimpleNamespace(content=content, stop_reason=stop, usage=u, model=model)

        if (data.get("promptFeedback") or {}).get("blockReason"):
            return done([], "refusal")
        cands = data.get("candidates") or []
        if not cands:
            raise GeminiError("the Gemini API returned no candidates")
        cand = cands[0]
        finish = cand.get("finishReason") or "STOP"
        blocks, has_call = [], False
        for part in (cand.get("content") or {}).get("parts") or []:
            if part.get("thought") is True:
                continue                                                 # thought summaries are not part of the answer
            if "functionCall" in part:
                fc, has_call = part["functionCall"], True
                self._n += 1
                uid = f"call_{self._n}"
                self._names[uid] = fc.get("name", "")
                blocks.append(SimpleNamespace(type="tool_use", id=uid, name=fc.get("name", ""), input=fc.get("args") or {}, _raw=part))
            elif "text" in part:
                blocks.append(SimpleNamespace(type="text", text=part["text"], _raw=part))
        if finish in REFUSAL_FINISH:
            return done(blocks, "refusal")
        if finish == "MAX_TOKENS":
            return done(blocks, "max_tokens")
        if finish == "MALFORMED_FUNCTION_CALL" and not has_call:          # a generation hiccup: let the loop nudge and retry
            blocks = [SimpleNamespace(type="text", text="(the previous function call was malformed)",
                                      _raw={"text": "(the previous function call was malformed)"})]
        if not blocks:
            blocks = [SimpleNamespace(type="text", text="", _raw={"text": " "})]
        return done(blocks, "tool_use" if has_call else "end_turn")

    # ------------------------------------------------------------ the Anthropic-shaped entry point
    def create(self, *, model: str, max_tokens: int, messages: list[dict], system: str | None = None,
               tools: list[dict] | None = None, output_config: dict | None = None, **_ignored):
        try:
            data = self._post(model, self._body(max_tokens, messages, system, tools, output_config))
        except GeminiError as e:
            if e.status != 400 or "think" not in str(e).lower():
                raise
            data = self._post(model, self._body(max_tokens, messages, system, tools, output_config, thinking=False))
        return self._response(data, model)

    def list_models(self) -> list[str]:
        """Models this key can call generateContent on (used by `chaos llm-check` to pick a valid name)."""
        r = self._http.get(f"{BASE_URL}/models", params={"pageSize": 200}, headers={"x-goog-api-key": self._key})
        if r.status_code != 200:
            status, message = self._error_text(r)
            raise GeminiError(f"HTTP {r.status_code} {status}: {message}".strip(), r.status_code)
        return sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
