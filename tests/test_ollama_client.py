from __future__ import annotations

from types import SimpleNamespace

import pytest

from nova_mem.models.client import OllamaClient, client_from_config


def test_client_from_config_sends_num_ctx(monkeypatch):
    sent = {}

    def fake_post(url, *, json, timeout):
        sent.update(url=url, payload=json, timeout=timeout)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {"content": "answer"},
                "prompt_eval_count": 4,
                "eval_count": 1,
                "context": [1, 2, 3, 4],
                "total_duration": 6_000_000_000,
                "prompt_eval_duration": 3_000_000_000,
                "eval_duration": 2_000_000_000,
            },
        )

    monkeypatch.setattr("nova_mem.models.client.requests.post", fake_post)
    client = client_from_config(
        {
            "backend": "ollama",
            "model_id": "llama3.2:3b-instruct-q4_K_M",
            "num_ctx": 16384,
        }
    )

    result = client.generate("question", seed=0)

    assert sent["payload"]["options"]["num_ctx"] == 16384
    assert result.extra["num_ctx_requested"] == 16384
    assert result.extra["server_timings_s"] == {
        "total": 6.0,
        "prompt_eval": 3.0,
        "generation_eval": 2.0,
    }


def test_ollama_client_rejects_invalid_server_duration(monkeypatch):
    class Response:
        raise_for_status = lambda self: None

        def json(self):
            return {
                "message": {"content": "answer"},
                "prompt_eval_duration": -1,
            }

    monkeypatch.setattr("nova_mem.models.client.requests.post", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="prompt_eval_duration"):
        OllamaClient("model").generate("question")


@pytest.mark.parametrize("num_ctx", [0, -1, True, "16384"])
def test_ollama_client_rejects_invalid_num_ctx(num_ctx):
    with pytest.raises(ValueError, match="num_ctx must be a positive integer"):
        OllamaClient("model", num_ctx=num_ctx)
