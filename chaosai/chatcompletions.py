"""Shared adapter for OpenAI-style chat completions APIs (Mistral, OpenRouter): the same slice of the Anthropic client
that the planner and analyzer use (see gemini.py), backed by `POST {base}/chat/completions`.

`client.beta.messages.create(...)` is translated to a chat completion and the answer is returned as Anthropic-shaped
`text` / `tool_use` blocks, so the tool loop, the evidence validator, the caches and the rule-based fallback work
unchanged. Nothing here decides anything: an answer has to pass the same validation as any other, and every failure
raises (callers fall back to the rules).

Plain HTTPS through httpx (no new dependency). The key goes in the `Authorization` header, never in the URL, and is never
logged or put into an error message. Subclasses set the endpoint, the key variable, and the few provider quirks.

Translation notes
  * system prompt          -> a leading `system` message
  * tools                  -> `tools: [{type: function, function: {name, description, parameters}}]`, tool_choice auto
  * tool_use / tool_result -> assistant `tool_calls` (arguments are a JSON string) / `tool` messages carrying
                              tool_call_id and the function name. The model's own assistant message is replayed
                              unchanged, with the ids the API issued
  * output_config.format   -> response_format json_schema (strict); if the model rejects it, the call is repeated in
                              json_object mode with the schema spelled out in the system prompt
  * effort, betas, fallbacks are Anthropic-specific and ignored
  * 429 / 5xx              -> retried with backoff (honouring Retry-After); a wait longer than we would ever sleep, or a
                              limit the subclass recognises as final, fails immediately with a readable message
  * history from another provider (the pool fails over mid-run) is rebuilt from the generic blocks; tool-call ids the
    API would refuse are mapped to valid ones where the subclass says so (NORMALIZE_IDS)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from types import SimpleNamespace
from typing import Callable, Optional

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_BACKOFF_S = 30.0
_VALID_ID = re.compile(r"[A-Za-z0-9]{9}")


class ChatCompletionsError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s       # how long a caller should leave this endpoint alone, when known


def _get(b, name, default=None):
    return b.get(name, default) if isinstance(b, dict) else getattr(b, name, default)


def _text_of(content) -> str:
    """Message content is a string, or a list of chunks (reasoning models add `thinking` chunks, which are not the answer)."""
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


class ChatCompletionsClient:
    """Duck-types the parts of anthropic.Anthropic() the planner and analyzer use. Subclass and set the class fields."""

    PROVIDER = "chat completions"                # how errors name the service
    BASE_URL = ""
    KEY_ENV = ""
    NORMALIZE_IDS = False                        # the API only accepts 9-character alphanumeric tool-call ids
    MAX_WAIT_S = MAX_BACKOFF_S                   # a 429 asking for a longer wait than this fails at once (the pool moves on)
    REPLAY_FIELDS: tuple = ()                    # extra assistant-message fields to send back unchanged (e.g. reasoning)

    def __init__(self, key: Optional[str] = None, transport: Optional[httpx.BaseTransport] = None,
                 sleep: Callable[[float], None] = time.sleep, max_retries: int = 4, timeout_s: float = 120.0):
        self._key = key or os.environ.get(self.KEY_ENV) or None
        if not self._key:
            raise ChatCompletionsError(f"no {self.PROVIDER} credentials found (set {self.KEY_ENV} in .env or the environment)")
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self._sleep, self.max_retries = sleep, max_retries
        self._names: dict[str, str] = {}          # tool_call id -> function name, to label `tool` messages
        self.calls = 0
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    # ------------------------------------------------------------ hooks for subclasses
    def extra_headers(self) -> dict:
        return {}

    def extra_body(self) -> dict:
        return {}

    def final_429(self, r: httpx.Response) -> Optional[ChatCompletionsError]:
        """A 429 that waiting cannot fix (spent quota, a plan with no allowance): return the error to raise at once."""
        return None

    # ------------------------------------------------------------ request
    def _wire_id(self, tool_call_id: str) -> str:
        if not self.NORMALIZE_IDS or _VALID_ID.fullmatch(tool_call_id or ""):
            return tool_call_id
        return hashlib.sha1((tool_call_id or "").encode()).hexdigest()[:9]

    def _remember_names(self, messages: list[dict]) -> None:
        """Function names come from the history itself, so this works for tool calls another provider issued."""
        for m in messages:
            if m["role"] == "assistant" and not isinstance(m["content"], str):
                for b in m["content"]:
                    if _get(b, "type") == "tool_use":
                        self._names[_get(b, "id")] = _get(b, "name")

    def _assistant(self, content) -> dict:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        for b in content:
            mine = not isinstance(b, dict) and getattr(b, "_owner", None) == self.PROVIDER     # never replay another provider's raw message
            raw = getattr(b, "_msg", None) if mine else None
            if raw is not None:                                           # our own earlier output: replay unchanged
                return raw
        calls = [{"id": self._wire_id(_get(b, "id")), "type": "function",
                  "function": {"name": _get(b, "name"), "arguments": json.dumps(_get(b, "input") or {})}}
                 for b in content if _get(b, "type") == "tool_use"]
        msg = {"role": "assistant", "content": "".join(_get(b, "text") or "" for b in content if _get(b, "type") == "text")}
        if calls:
            msg["tool_calls"] = calls
        return msg

    def _messages(self, messages: list[dict], system: Optional[str]) -> list[dict]:
        self._remember_names(messages)
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            content = m["content"]
            if m["role"] == "assistant":
                out.append(self._assistant(content))
            elif isinstance(content, str):
                out.append({"role": "user", "content": content})
            else:
                texts = []
                for b in content:
                    if _get(b, "type") == "tool_result":
                        body = _get(b, "content")
                        out.append({"role": "tool", "tool_call_id": self._wire_id(_get(b, "tool_use_id")),
                                    "name": self._names.get(_get(b, "tool_use_id"), "tool"),
                                    "content": body if isinstance(body, str) else json.dumps(body)})
                    elif _get(b, "text"):
                        texts.append(_get(b, "text"))
                if texts:
                    out.append({"role": "user", "content": "\n".join(texts)})
        return out

    def _body(self, model, max_tokens, messages, system, tools, output_config, json_object_fallback=False) -> dict:
        fmt = (output_config or {}).get("format")
        if json_object_fallback and fmt and fmt.get("type") == "json_schema":
            system = (system or "") + "\n\nReply with a single JSON object that matches this JSON Schema, and nothing else:\n" + json.dumps(fmt["schema"])
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": self._messages(messages, system)}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                               "parameters": t["input_schema"]}} for t in tools]
            body["tool_choice"] = "auto"
        if fmt and fmt.get("type") == "json_schema":
            body["response_format"] = ({"type": "json_object"} if json_object_fallback else
                                       {"type": "json_schema", "json_schema": {"name": "answer", "schema": fmt["schema"], "strict": True}})
        body.update(self.extra_body())
        return body

    # ------------------------------------------------------------ transport
    @staticmethod
    def _error_text(r: httpx.Response) -> str:
        try:
            d = r.json()
            err = d.get("error") if isinstance(d, dict) else None
            msg = (err.get("message") if isinstance(err, dict) else None) or d.get("message") or d.get("detail") or err or r.text
        except (ValueError, AttributeError):
            msg = r.text
        return (msg if isinstance(msg, str) else json.dumps(msg))[:300]

    @staticmethod
    def _retry_after(r: httpx.Response) -> Optional[float]:
        v = r.headers.get("retry-after")
        return float(v) if v and v.replace(".", "", 1).isdigit() else None

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "application/json", **self.extra_headers()}
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            try:
                r = self._http.post(f"{self.BASE_URL}/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    self._sleep(min(2.0 ** (attempt + 1), MAX_BACKOFF_S))
                    continue
                raise ChatCompletionsError(f"cannot reach the {self.PROVIDER} API ({type(e).__name__})", retry_after_s=60) from None
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                final = self.final_429(r)
                if final:
                    raise final
            wait = self._retry_after(r)
            if r.status_code == 429 and wait is not None and wait > self.MAX_WAIT_S:
                raise ChatCompletionsError(f"rate limit exceeded; the API asks for a wait of about {wait / 60:.0f} min ({self._error_text(r)})", 429, wait)
            if r.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(min(wait if wait is not None else 2.0 ** (attempt + 1), max(MAX_BACKOFF_S, self.MAX_WAIT_S)))
                continue
            raise ChatCompletionsError(f"HTTP {r.status_code}: {self._error_text(r)}", r.status_code, wait)
        raise ChatCompletionsError("retries exhausted")                    # unreachable; keeps type checkers calm

    # ------------------------------------------------------------ response
    def _response(self, data: dict, model: str):
        usage = data.get("usage") or {}
        u = SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0) or 0, output_tokens=usage.get("completion_tokens", 0) or 0)

        def done(content, stop):
            return SimpleNamespace(content=content, stop_reason=stop, usage=u, model=data.get("model") or model)

        if isinstance(data.get("error"), dict) and not data.get("choices"):    # some gateways return errors with HTTP 200
            e = data["error"]
            code = e.get("code") if isinstance(e.get("code"), int) else None
            raise ChatCompletionsError(f"{self.PROVIDER} error{f' {code}' if code else ''}: {str(e.get('message', ''))[:300]}", code)
        choices = data.get("choices") or []
        if not choices:
            raise ChatCompletionsError(f"the {self.PROVIDER} API returned no choices")
        choice, msg = choices[0], choices[0].get("message") or {}
        finish = choice.get("finish_reason") or "stop"
        if finish == "error":
            raise ChatCompletionsError("the model reported an error while generating")
        text, blocks, calls, malformed = _text_of(msg.get("content")), [], [], False
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else (args or {})
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                malformed = True
                continue
            self._names[tc.get("id", "")] = fn.get("name", "")
            calls.append({"id": tc.get("id"), "type": "function", "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(args)}})
            blocks.append(SimpleNamespace(type="tool_use", id=tc.get("id"), name=fn.get("name", ""), input=args))
        if malformed and not blocks:                                      # a generation hiccup: let the loop nudge and retry
            text = text or "(the previous tool call had malformed arguments)"
        replay: dict = {"role": "assistant", "content": msg.get("content") if msg.get("content") not in (None, []) else ""}
        if calls:
            replay["tool_calls"] = calls
        for field in self.REPLAY_FIELDS:                                  # e.g. reasoning the API wants back on the next turn
            if msg.get(field) is not None:
                replay[field] = msg[field]
        if text:
            blocks.insert(0, SimpleNamespace(type="text", text=text))
        if not blocks:
            blocks = [SimpleNamespace(type="text", text="")]
        for b in blocks:
            b._msg, b._owner = replay, self.PROVIDER                      # replayed verbatim next turn, by this provider only (see _assistant)
        if finish in ("length", "model_length"):
            return done(blocks, "max_tokens")
        return done(blocks, "tool_use" if calls else "end_turn")

    # ------------------------------------------------------------ the Anthropic-shaped entry point
    def create(self, *, model: str, max_tokens: int, messages: list[dict], system: str | None = None,
               tools: list[dict] | None = None, output_config: dict | None = None, **_ignored):
        try:
            data = self._post(self._body(model, max_tokens, messages, system, tools, output_config))
        except ChatCompletionsError as e:
            fmt = (output_config or {}).get("format")
            if e.status not in (400, 422) or not fmt or not any(w in str(e).lower() for w in ("response_format", "json_schema", "schema", "strict")):
                raise
            data = self._post(self._body(model, max_tokens, messages, system, tools, output_config, json_object_fallback=True))
        return self._response(data, model)

    def list_models(self) -> list[str]:
        raise NotImplementedError
