import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import urllib.error

from run_live import Client


def reply(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 10}}


def test_fence_is_stripped_and_counted():
    c = Client("m", "k", post=lambda b: reply('```json\n{"a": 1}\n```'))
    assert json.loads(c.complete("s", "u")) == {"a": 1}
    assert c.stats["needed_fence_strip"] == 1


def test_retries_then_succeeds():
    calls = {"n": 0}

    def post(b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return reply("{}")

    c = Client("m", "k", post=post, sleep=lambda s: None)
    assert c.complete("s", "u") == "{}" and calls["n"] == 3


def test_persistent_error_returns_empty_and_is_counted():
    def post(b):
        raise urllib.error.HTTPError("u", 401, "no", {}, None)

    c = Client("m", "k", post=post, sleep=lambda s: None)
    assert c.complete("s", "u") == "" and c.stats["api_errors"] == 1
