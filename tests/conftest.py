"""Every test runs hermetically with respect to the developer's real setup.

The CLI loads `.env` at import time and reads config.yaml, so without this a test could pick up a real API key
(and spend quota calling a live LLM) or behave differently depending on which provider config.yaml selects.
Here the config path points at a file that does not exist (so defaults apply) and every API-key variable is
removed; a test that needs one sets it itself with monkeypatch.
"""
import pytest

KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _hermetic_llm_environment(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("CHAOSAI_CONFIG", str(tmp_path_factory.getbasetemp() / "no-such-config.yaml"))
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("CHAOSAI_NO_DOTENV", raising=False)       # the acceptance run sets it for its subprocesses; a test must not inherit it
