import os

import pytest

from chaosai.envfile import load_env, mask


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_FOO", "CHAOSAI_X", "EVIL_VAR", "PATH_BACKUP"):
        monkeypatch.delenv(k, raising=False)


def write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_allowed_names_and_strips_quotes_and_export(tmp_path):
    p = write(tmp_path, 'ANTHROPIC_API_KEY="sk-ant-abc123"\nexport CHAOSAI_X=1\n# comment\n\nANTHROPIC_FOO = \'bar\'\n')
    assert load_env(p) == ["ANTHROPIC_API_KEY", "CHAOSAI_X", "ANTHROPIC_FOO"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc123" and os.environ["CHAOSAI_X"] == "1" and os.environ["ANTHROPIC_FOO"] == "bar"


def test_other_names_are_ignored_so_a_stray_line_cannot_change_the_environment(tmp_path):
    before = os.environ.get("PATH")
    p = write(tmp_path, "PATH=/evil\nEVIL_VAR=1\nANTHROPIC_API_KEY=sk-ant-x1\n")
    assert load_env(p) == ["ANTHROPIC_API_KEY"]
    assert os.environ.get("PATH") == before and "EVIL_VAR" not in os.environ


def test_existing_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-real-env")
    assert load_env(write(tmp_path, "ANTHROPIC_API_KEY=from-file\n")) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "from-real-env"


def test_missing_file_and_empty_values_are_fine(tmp_path):
    assert load_env(tmp_path / "nope") == []
    assert load_env(write(tmp_path, "ANTHROPIC_API_KEY=\nANTHROPIC_FOO\n")) == []


def test_bom_from_windows_editors_is_tolerated(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes(b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-ant-bom1\r\n")
    assert load_env(p) == ["ANTHROPIC_API_KEY"] and os.environ["ANTHROPIC_API_KEY"] == "sk-ant-bom1"


def test_mask_never_reveals_the_middle_of_a_key():
    m = mask("sk-ant-api03-SECRETSECRETSECRET-wxyz")
    assert m == "sk-ant-...wxyz" and "SECRET" not in m
    assert mask("") == "(empty)" and mask("short") == "***"


def test_chaosai_no_dotenv_skips_the_file_entirely(tmp_path, monkeypatch):
    p = write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-x1\nGEMINI_API_KEY=g1\n")
    monkeypatch.setenv("CHAOSAI_NO_DOTENV", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_env(p) == [] and "ANTHROPIC_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ


def test_any_other_value_of_chaosai_no_dotenv_still_loads_the_file(tmp_path, monkeypatch):
    p = write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-x1\n")
    monkeypatch.setenv("CHAOSAI_NO_DOTENV", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        assert load_env(p) == ["ANTHROPIC_API_KEY"]
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
