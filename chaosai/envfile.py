"""Tiny `.env` loader so the Anthropic key can live in a git-ignored file instead of being retyped.

Safety rules: it never overrides a variable that is already set, it only accepts names starting with
ANTHROPIC_, GEMINI_, MISTRAL_, OPENROUTER_ or CHAOSAI_ (a stray `.env` line must not be able to change PATH or similar), and it never
prints a value. `.env` is listed in .gitignore.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ALLOWED_PREFIXES = ("ANTHROPIC_", "GEMINI_", "MISTRAL_", "OPENROUTER_", "CHAOSAI_")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | str | None = None) -> list[str]:
    """Load KEY=VALUE lines into os.environ. Returns the names that were set (never the values)."""
    p = Path(path) if path else ROOT / ".env"
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    loaded = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not _NAME.match(name) or not name.startswith(ALLOWED_PREFIXES) or not value:
            continue
        if name in os.environ:                       # real environment wins
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


def mask(secret: str) -> str:
    """Show only enough of a key to recognise it."""
    return "(empty)" if not secret else (secret[:7] + "..." + secret[-4:] if len(secret) > 14 else "***")
