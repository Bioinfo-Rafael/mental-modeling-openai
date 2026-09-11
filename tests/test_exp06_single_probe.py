from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import httpx
import pytest
from openai import APIStatusError
from tools import probe_exp06_query as probe


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("alias", ["sol", "terra", "luna"])
def test_exactly_one_create_and_separate_logs(tmp_path, monkeypatch, failure, alias):
    kwargs = {"model": "gpt-5.6-sol", "reasoning_effort": "medium", "messages": [
        {"role": "system", "content": "unchanged system"}, {"role": "user", "content": "unchanged history/question"}]}
    expected = {**kwargs, "model": probe.common.MODELS[alias]}
    monkeypatch.setattr(probe, "load_request", lambda: kwargs)
    monkeypatch.setattr(probe, "SOURCE", tmp_path)
    monkeypatch.setattr(probe, "OUTPUT", tmp_path / "outputs")
    (tmp_path / "requests.jsonl").write_text("fixture\n")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    factory = MagicMock()
    client = factory.return_value.__enter__.return_value
    create = client.chat.completions.create
    if failure:
        response = httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
                                  headers={"x-request-id": "req_failure"})
        create.side_effect = APIStatusError("server error", response=response, body={"error": "fixture"})
    else:
        create.return_value = SimpleNamespace(_request_id="req_success", model_dump=lambda **kw: {"usage": {"prompt_tokens": 1}})
    monkeypatch.setattr(probe, "OpenAI", factory)
    assert probe.main(["--model", alias]) == 0
    factory.assert_not_called()
    assert probe.main(["--model", alias, "--execute", "--confirm-paid-api"]) == (2 if failure else 0)
    create.assert_called_once_with(**expected)
    assert kwargs["model"] == "gpt-5.6-sol"  # Do not mutate the loaded source request.
    assert factory.call_args.kwargs["max_retries"] == 0
    directory = next((tmp_path / "outputs").iterdir())
    result = json.loads((directory / "result.json").read_text())
    assert result["request_id"] == ("req_failure" if failure else "req_success")
    assert not result["included_in_experiment"]
    assert json.loads((directory / "request.json").read_text())["kwargs"] == expected
    assert result["model"] == expected["model"] and result["source_query_id"] == probe.QUERY_ID
    assert (tmp_path / "requests.jsonl").read_text() == "fixture\n"
