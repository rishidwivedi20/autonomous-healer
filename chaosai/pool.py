"""A pool of LLM clients that looks like one client: spreads runs across members and fails over when one is out.

Why: free tiers are small (Gemini allows about 20 requests per model per day, OpenRouter's free models have a daily
cap, Mistral's depends on the plan), and one analysis takes about 10 requests. A pool of several models or keys turns
those small allowances into one bigger one, and means one exhausted quota does not end the AI step.

How it behaves
  * A NEW conversation (one message: the planner's single call, or the analyzer's first) starts on the next member in
    rotation. The rotation position is saved in `results/llm_pool.json`, so successive command-line runs really
    spread across members instead of always starting on the first.
  * A conversation then STAYS on its member for every turn: a tool loop replays provider-specific parts (Gemini's thought
    signatures), so hopping between members mid-run would break them.
  * If a member fails (rate limit, spent quota, overload, bad key, unknown model, network), it is put on a cooldown
    (the API's own "retry in ..." when it gave one) and the SAME request goes to the next member. Cooldowns are saved too,
    so a member known to be out is not tried again by the next run. A conversation that is already under way is never
    moved INTO Gemini or Anthropic, because their histories need parts only they issued.
  * Anything the model returns (a refusal, a cut-off answer, an invalid analysis) is not a failure of the member: it goes
    to the caller as usual and the existing validation and fallbacks decide.
  * If every member is unavailable, PoolError lists each reason, and the planner or analyzer falls back to the
    heuristic / rule-based path and records why.

Several keys help only if they have separate quotas (Gemini's quota is per Google project, not per key); several
models of one provider each have their own.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from .config import PoolMember, pool_label

MAX_COOLDOWN_S = 24 * 3600.0
DEFAULT_COOLDOWN_S = {429: 120.0, 401: 3600.0, 402: 3600.0, 403: 3600.0, 404: 6 * 3600.0}
OTHER_COOLDOWN_S = 60.0
NO_MID_RUN_ENTRY = {"gemini", "anthropic"}       # their histories carry parts only they issue
BUG_TYPES = (TypeError, AttributeError, NameError, AssertionError, KeyError)


class PoolError(Exception):
    pass


def default_state_path() -> Path:
    return Path(os.environ.get("CHAOSAI_RESULTS", "results")) / "llm_pool.json"


def make_member_client(m: PoolMember):
    """The real client for one member, built with its own key variable when it names one."""
    key = os.environ.get(m.key_env) if m.key_env else None
    if m.key_env and not key:
        raise PoolError(f"{m.key_env} is not set")
    if m.provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(key=key)
    if m.provider == "mistral":
        from .mistral import MistralClient
        return MistralClient(key=key)
    if m.provider == "openrouter":
        from .openrouter import OpenRouterClient
        return OpenRouterClient(key=key)
    if not (key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise PoolError("no Anthropic credentials found (set ANTHROPIC_API_KEY)")
    import anthropic
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


class PoolClient:
    """Duck-types the parts of anthropic.Anthropic() the planner and analyzer use."""

    def __init__(self, members: list[PoolMember], state_path: Path | str | None = None,
                 clock: Callable[[], float] = time.time, make: Callable = make_member_client):
        if not members:
            raise PoolError("the pool has no members (set `members:` in config.yaml)")
        self.members, self.clock, self._make = list(members), clock, make
        self.state_path = Path(state_path) if state_path else default_state_path()
        self._clients: dict[int, object] = {}
        self._current: Optional[int] = None
        self.used_models: list[str] = []
        self.calls = 0
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    # ------------------------------------------------------------ saved state (rotation position and cooldowns)
    def _load(self) -> dict:
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            return s if isinstance(s, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        except OSError:
            pass                                                             # losing the rotation hint must never break a run

    def _cooldown(self, label: str, seconds: float, reason: str) -> None:
        state = self._load()
        state.setdefault("cooldown", {})[label] = {"until": self.clock() + min(seconds, MAX_COOLDOWN_S), "reason": reason[:200]}
        self._save(state)

    def _cooling(self, state: dict) -> dict[str, dict]:
        now = self.clock()
        return {k: v for k, v in (state.get("cooldown") or {}).items() if isinstance(v, dict) and v.get("until", 0) > now}

    # ------------------------------------------------------------ choosing and calling members
    def _client(self, idx: int):
        if idx not in self._clients:
            self._clients[idx] = self._make(self.members[idx])
        return self._clients[idx]

    @staticmethod
    def _cool_for(exc: Exception) -> float:
        secs = getattr(exc, "retry_after_s", None)
        if isinstance(secs, (int, float)) and secs > 0:
            return float(secs)
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        return DEFAULT_COOLDOWN_S.get(status, OTHER_COOLDOWN_S)

    def create(self, *, messages: list, model=None, **kw):
        """`model` is ignored: each member brings its own."""
        mid_run = len(messages) > 1
        state = self._load()
        cooling = self._cooling(state)
        n = len(self.members)
        if mid_run and self._current is not None:
            order = [(self._current + i) % n for i in range(n)]            # stay on this member while it works
        else:
            start = int(state.get("next", 0)) % n
            order = [(start + i) % n for i in range(n)]
        problems: list[str] = []
        for idx in order:
            m = self.members[idx]
            label = pool_label(m)
            if mid_run and m.provider in NO_MID_RUN_ENTRY and idx != self._current:
                problems.append(f"{label}: cannot take over a run that is already under way")
                continue
            if label in cooling:
                left = max(0.0, cooling[label]["until"] - self.clock())
                problems.append(f"{label}: unavailable for about {left / 60:.0f} more min ({cooling[label].get('reason', '')})")
                continue
            try:
                client = self._client(idx)
            except Exception as e:                                           # no key, SDK missing, ...
                problems.append(f"{label}: {str(e)[:160]}")
                self._cooldown(label, 3600.0, str(e))
                continue
            self.calls += 1
            try:
                resp = client.beta.messages.create(model=m.model, messages=messages, **kw)
            except BUG_TYPES:
                raise                                                        # a defect in our own request building: do not mask it
            except Exception as e:
                problems.append(f"{label}: {str(e)[:160]}")
                self._cooldown(label, self._cool_for(e), str(e))
                continue
            self._current = idx
            if label not in self.used_models:
                self.used_models.append(label)
            if not mid_run:
                state = self._load()
                state["next"] = (idx + 1) % n                                # the next new run starts on the member after this one
                self._save(state)
            return resp
        raise PoolError("no pool member could answer: " + "; ".join(problems))

    # ------------------------------------------------------------ for `chaos llm-check`
    def status(self) -> list[tuple[str, float, str]]:
        """(label, seconds left on its cooldown or 0, why) for every member, from the saved state."""
        cooling, now = self._cooling(self._load()), self.clock()
        return [(pool_label(m), max(0.0, cooling[pool_label(m)]["until"] - now) if pool_label(m) in cooling else 0.0,
                 cooling.get(pool_label(m), {}).get("reason", "")) for m in self.members]

    def check(self) -> list[tuple[str, bool, str]]:
        """One tiny request per member (bypassing rotation and cooldowns): (label, worked, detail)."""
        out = []
        for idx, m in enumerate(self.members):
            label = pool_label(m)
            try:
                r = self._client(idx).beta.messages.create(model=m.model, max_tokens=256,
                                                           messages=[{"role": "user", "content": "Reply with the single word: ok"}])
                text = next((b.text for b in r.content if getattr(b, "type", None) == "text"), "")
                out.append((label, True, f"answered {text.strip()[:20]!r}; tokens in={r.usage.input_tokens} out={r.usage.output_tokens}"))
            except Exception as e:
                out.append((label, False, str(e)[:200]))
        return out
