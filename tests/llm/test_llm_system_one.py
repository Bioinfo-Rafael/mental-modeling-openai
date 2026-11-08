"""System One part: questions, request body, parsing, budget, status/transport mapping (MockTransport)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from translate.llm.system_one import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneClient,
    SystemOneConfigError,
    SystemOneModelError,
    SystemOneProtocolError,
    SystemOneUsage,
    question_set_sha256,
    questions_payload,
)

from .conftest import SYSTEMONE_KEY_ENV, SYSTEMONE_KEY_VALUE
from .llm_fakes import (
    SystemOneServer,
    choice_answer,
    json_response,
    models_body,
    noul_answer,
    score_answer,
    systemone_body,
)

BASE_URL = "http://127.0.0.1:8000"
MODEL = "clef-flash"
BUDGET = 16384
STATE = "STEP 0 aircraft A0100 alive=1 lon=1.0 lat=2.0 [customer text]"
CHANGED = NoulQuestion("changed", "Did the motion change beyond the previous intent?", "yes, changed", "no change")
KIND = ChoiceQuestion("kind", "Which manoeuvre?", (("turn", "a turn"), ("climb", None)))
LEVEL = ScoreQuestion("level", "How strong is the change?", ("low", "mid", "high"))
ALL_QUESTIONS = (CHANGED, KIND, LEVEL)

# Every test has the local Bearer key in the environment; TCP clients require it.
pytestmark = pytest.mark.usefixtures("systemone_key")


def make_client(server: SystemOneServer, **overrides: Any) -> SystemOneClient:
    fields: dict[str, Any] = {
        "base_url": BASE_URL,
        "uds_path": None,
        "model": MODEL,
        "questions": (CHANGED,),
        "input_token_budget": BUDGET,
        "timeout_s": 5.0,
        "api_key_env": SYSTEMONE_KEY_ENV,
        "transport": server.transport,
    }
    return SystemOneClient(**{**fields, **overrides})


def ok_response(input_tokens: int = 500, **extras: Any) -> httpx.Response:
    return json_response(200, systemone_body({"changed": noul_answer(0.8, **extras)}, input_tokens=input_tokens))


# --- questions and payload ------------------------------------------------------------------------


def test_questions_payload_serialises_noul_choice_and_score_in_order() -> None:
    assert questions_payload(ALL_QUESTIONS) == {
        "changed": {
            "type": "noul",
            "instructions": "Did the motion change beyond the previous intent?",
            "criteria": {"true": "yes, changed", "false": "no change"},
        },
        "kind": {"type": "choice", "instructions": "Which manoeuvre?", "criteria": {"turn": "a turn", "climb": None}},
        "level": {"type": "score", "instructions": "How strong is the change?", "criteria": ["low", "mid", "high"]},
    }
    assert list(questions_payload(ALL_QUESTIONS)) == ["changed", "kind", "level"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: NoulQuestion("", "i", "t", "f"),
        lambda: NoulQuestion("q", "  ", "t", "f"),
        lambda: NoulQuestion("q", "i", "", "f"),
        lambda: NoulQuestion("q", "lone \ud800 surrogate", "t", "f"),
        lambda: ChoiceQuestion("q", "i", ()),
        lambda: ChoiceQuestion("q", "i", [("a", None)]),
        lambda: ChoiceQuestion("q", "i", (("a", None), ("a", "dup"))),
        lambda: ChoiceQuestion("q", "i", (("", None),)),
        lambda: ChoiceQuestion("q", "i", (("a", 1),)),
        lambda: ChoiceQuestion("q", "i", (("a", "lone \ud800 surrogate"),)),
        lambda: ChoiceQuestion("q", "i", tuple((f"o{i}", None) for i in range(27))),
        lambda: ScoreQuestion("q", "i", ("only",)),
        lambda: ScoreQuestion("q", "i", tuple(f"l{i}" for i in range(11))),
        lambda: ScoreQuestion("q", "i", ("a", "a")),
    ],
)
def test_invalid_question_raises_config_error(build: Any) -> None:
    with pytest.raises(SystemOneConfigError):
        build()


def test_question_limits_accept_the_boundaries() -> None:
    choice = ChoiceQuestion("q", "i", tuple((f"o{i}", None) for i in range(26)))
    score_min = ScoreQuestion("s", "i", ("a", "b"))
    score_max = ScoreQuestion("t", "i", tuple(f"l{i}" for i in range(10)))
    many = tuple(NoulQuestion(f"q{i}", "i", "t", "f") for i in range(64))

    assert len(questions_payload((choice, score_min, score_max))) == 3
    assert len(questions_payload(many)) == 64


@pytest.mark.parametrize(
    "questions",
    [(), (CHANGED, CHANGED), tuple(NoulQuestion(f"q{i}", "i", "t", "f") for i in range(65)), ("not a question",)],
)
def test_invalid_question_set_raises_config_error(questions: tuple[Any, ...]) -> None:
    with pytest.raises(SystemOneConfigError):
        questions_payload(questions)


def test_question_set_hash_is_stable_and_equals_hash_of_sent_questions() -> None:
    server = SystemOneServer([ok_response(), ok_response()])
    client = make_client(server)

    client.decide(STATE)
    client.decide(STATE + " later")

    sent = [json.dumps(r.body["questions"], ensure_ascii=False, separators=(",", ":")) for r in server.requests]
    assert sent[0] == sent[1]
    assert client.question_set_sha256 == hashlib.sha256(sent[0].encode("utf-8")).hexdigest()
    assert client.question_set_sha256 == question_set_sha256((CHANGED,))


def test_question_set_hash_changes_with_content_or_order() -> None:
    other = NoulQuestion("changed", "Different instructions", "yes, changed", "no change")

    assert question_set_sha256((CHANGED,)) != question_set_sha256((other,))
    assert question_set_sha256((CHANGED, KIND)) != question_set_sha256((KIND, CHANGED))


# --- happy paths ------------------------------------------------------------------------------------


def test_decide_over_tcp_sends_exact_body_with_bearer_key() -> None:
    server = SystemOneServer([ok_response()])
    client = make_client(server)

    client.decide(STATE)

    request = server.requests[0]
    assert (request.method, request.path) == ("POST", "/v1/systemone")
    assert request.headers["authorization"] == f"Bearer {SYSTEMONE_KEY_VALUE}"
    assert request.headers["content-type"] == "application/json"
    assert request.body == {"model": MODEL, "state": STATE, "questions": questions_payload((CHANGED,))}


def test_decide_parses_all_answer_types_and_ignores_vllm_only_fields() -> None:
    answers = {
        "changed": noul_answer(0.8, vllm_extras=True),
        "kind": choice_answer("climb", {"turn": 0.25, "climb": 0.75}),
        "level": score_answer(1.5, {"0": 0.1, "1": 0.3, "2": 0.6}),
    }
    body = systemone_body(answers, input_tokens=900, vllm_extras=True)
    server = SystemOneServer([json_response(200, body)])

    result = make_client(server, questions=ALL_QUESTIONS).decide(STATE)

    assert result.model == MODEL
    assert result.answers == (
        NoulAnswer("changed", 0.8),
        ChoiceAnswer("kind", "climb", (("turn", 0.25), ("climb", 0.75))),
        ScoreAnswer("level", 1.5, (("0", 0.1), ("1", 0.3), ("2", 0.6))),
    )
    assert result.usage == SystemOneUsage(input_tokens=900, output_tokens=0)
    assert result.raw == body
    assert result.answer("kind") == ChoiceAnswer("kind", "climb", (("turn", 0.25), ("climb", 0.75)))


def test_decide_parses_openjev_style_noul_without_probabilities() -> None:
    server = SystemOneServer([json_response(200, {**systemone_body({"changed": {"type": "noul", "noul": 1}})})])

    result = make_client(server).decide(STATE)

    assert result.answers == (NoulAnswer("changed", 1.0),)
    with pytest.raises(KeyError):
        result.answer("missing")


def test_decide_over_uds_without_key_sends_no_authorization(secure_uds_path: Path) -> None:
    server = SystemOneServer([ok_response()])
    client = make_client(server, base_url=None, uds_path=secure_uds_path, api_key_env=None)

    client.decide(STATE)

    assert server.requests[0].path == "/v1/systemone"
    assert "authorization" not in server.requests[0].headers


def test_decide_over_uds_with_key_sends_bearer(secure_uds_path: Path) -> None:
    server = SystemOneServer([ok_response()])

    make_client(server, base_url=None, uds_path=secure_uds_path).decide(STATE)

    assert server.requests[0].headers["authorization"] == f"Bearer {SYSTEMONE_KEY_VALUE}"


def test_ipv6_loopback_base_url_is_accepted() -> None:
    server = SystemOneServer([ok_response()])

    make_client(server, base_url="http://[::1]:8000/").decide(STATE)

    assert server.requests[0].path == "/v1/systemone"


def test_health_sends_authenticated_models_request() -> None:
    server = SystemOneServer([json_response(200, models_body("other", MODEL))])

    make_client(server).health()

    request = server.requests[0]
    assert (request.method, request.path) == ("GET", "/v1/models")
    assert request.headers["authorization"] == f"Bearer {SYSTEMONE_KEY_VALUE}"


def test_health_without_the_configured_model_raises_model_error() -> None:
    server = SystemOneServer([json_response(200, models_body("other"))])

    with pytest.raises(SystemOneModelError):
        make_client(server).health()


def test_health_ignores_non_string_model_ids() -> None:
    server = SystemOneServer([json_response(200, {"data": [{"id": ["clef-flash"]}, {"id": None}, "x"]})])

    with pytest.raises(SystemOneModelError):
        make_client(server).health()


def test_choice_option_with_empty_description_is_accepted() -> None:
    question = ChoiceQuestion("q", "i", (("a", ""), ("b", None)))

    assert questions_payload((question,))["q"]["criteria"] == {"a": "", "b": None}


def test_health_with_unexpected_body_raises_protocol_error() -> None:
    server = SystemOneServer([json_response(200, {"models": []})])

    with pytest.raises(SystemOneProtocolError):
        make_client(server).health()


def test_client_as_context_manager_closes() -> None:
    server = SystemOneServer([ok_response()])

    with make_client(server) as client:
        client.decide(STATE)

    assert len(server.requests) == 1
