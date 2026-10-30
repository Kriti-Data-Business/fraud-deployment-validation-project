import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import urllib.error

from run_live import Client


def reply(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 10}}


def test_fenced_or_wrapped_json_is_extracted_and_counted():
    c = Client("m", "k", post=lambda b: reply('```json\n{"a": 1}\n```\nHope this helps!'))
    assert json.loads(c.complete("s", "u")) == {"a": 1}
    assert c.stats["needed_json_extraction"] == 1
    c2 = Client("m", "k", post=lambda b: reply('{"a": 2}'))
    assert json.loads(c2.complete("s", "u")) == {"a": 2} and c2.stats["needed_json_extraction"] == 0


def test_text_with_no_object_still_fails_closed():
    from twohats.triage.assistant import LLMAssistant
    c = Client("m", "k", post=lambda b: reply("I cannot help with that."))
    assert "_unparseable" in LLMAssistant(c.complete).generate(__import__("types").SimpleNamespace(facts={}, note=""))


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
