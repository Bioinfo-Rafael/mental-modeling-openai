"""System One part: token budget, HTTP status / transport errors, protocol errors on 200."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from translate.llm import system_one
from translate.llm.system_one import (
    FIXED_FRAMING_RESERVE_TOKENS,
    PER_QUESTION_FRAMING_RESERVE_TOKENS,
    SystemOneAuthError,
    SystemOneBudgetError,
    SystemOneConfigError,
    SystemOneError,
    SystemOneModelError,
    SystemOneProtocolError,
    SystemOneRequestError,
    SystemOneServerError,
    SystemOneTransportError,
    SystemOneUnsupportedError,
    conservative_input_token_bound,
    questions_payload,
)

from .conftest import SYSTEMONE_KEY_VALUE
from .llm_fakes import SystemOneServer, json_response, noul_answer, raise_transport, systemone_body
from .test_llm_system_one import ALL_QUESTIONS, CHANGED, STATE, make_client, ok_response

pytestmark = pytest.mark.usefixtures("systemone_key")


def _never_called() -> SystemOneServer:
    return SystemOneServer([])


# --- token budget ----------------------------------------------------------------------------------


def test_conservative_bound_is_state_bytes_plus_question_bytes_plus_fixed_reserves() -> None:
    state = "héllo"  # 6 UTF-8 bytes
    question_bytes = len(system_one._canonical_json(questions_payload(ALL_QUESTIONS)).encode("utf-8"))

    bound = conservative_input_token_bound(state=state, questions=ALL_QUESTIONS)

    assert bound == 6 + question_bytes + FIXED_FRAMING_RESERVE_TOKENS + 3 * PER_QUESTION_FRAMING_RESERVE_TOKENS


def test_bound_one_below_budget_is_sent() -> None:
    server = SystemOneServer([ok_response(input_tokens=10)])
    budget = conservative_input_token_bound(state=STATE, questions=(CHANGED,)) + 1

    make_client(server, input_token_budget=budget).decide(STATE)

    assert len(server.requests) == 1


def test_bound_equal_to_budget_is_rejected_before_transport() -> None:
    server = _never_called()
    budget = conservative_input_token_bound(state=STATE, questions=(CHANGED,))
    client = make_client(server, input_token_budget=budget)

    with pytest.raises(SystemOneBudgetError) as excinfo:
        client.decide(STATE)

    assert server.requests == []
    assert STATE not in str(excinfo.value)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_non_positive_or_non_int_budget_is_rejected(budget: Any) -> None:
    with pytest.raises(SystemOneConfigError):
        make_client(_never_called(), input_token_budget=budget)


def test_budget_without_room_for_any_state_is_rejected_at_construction() -> None:
    empty_bound = conservative_input_token_bound(state="", questions=(CHANGED,))

    with pytest.raises(SystemOneConfigError, match="input_token_budget"):
        make_client(_never_called(), input_token_budget=empty_bound)


def test_usage_at_budget_raises_budget_error_keeping_raw() -> None:
    budget = 16384
    server = SystemOneServer([ok_response(input_tokens=budget)])

    with pytest.raises(SystemOneBudgetError) as excinfo:
        make_client(server, input_token_budget=budget).decide(STATE)

    assert excinfo.value.raw is not None
    assert excinfo.value.raw["usage"]["input_tokens"] == budget


def test_usage_one_below_budget_is_accepted() -> None:
    budget = 16384
    server = SystemOneServer([ok_response(input_tokens=budget - 1)])

    result = make_client(server, input_token_budget=budget).decide(STATE)

    assert result.usage.input_tokens == budget - 1


@pytest.mark.parametrize("state", ["", 42, "\ud800 lone surrogate"])
def test_invalid_state_is_rejected_before_transport(state: Any) -> None:
    server = _never_called()

    with pytest.raises(SystemOneRequestError):
        make_client(server).decide(state)

    assert server.requests == []


# --- HTTP status and transport mapping -------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, SystemOneRequestError),
        (422, SystemOneRequestError),
        (403, SystemOneRequestError),
        (409, SystemOneRequestError),
        (413, SystemOneRequestError),
        (401, SystemOneAuthError),
        (404, SystemOneModelError),
        (500, SystemOneServerError),
        (502, SystemOneServerError),
        (503, SystemOneServerError),
        (501, SystemOneUnsupportedError),
        (204, SystemOneProtocolError),
        (302, SystemOneProtocolError),
    ],
)
def test_http_status_maps_to_typed_error_without_body_or_key(status: int, error_type: type[SystemOneError]) -> None:
    echo = {"error": {"message": f"echo {STATE} {SYSTEMONE_KEY_VALUE}", "type": "BadRequestError", "code": status}}
    server = SystemOneServer([json_response(status, echo)])

    with pytest.raises(error_type) as excinfo:
        make_client(server).decide(STATE)

    message = str(excinfo.value)
    assert excinfo.value.status_code == status
    assert str(status) in message
    assert STATE not in message
    assert SYSTEMONE_KEY_VALUE not in message
    assert "customer text" not in message


@pytest.mark.parametrize("echoed", [SYSTEMONE_KEY_VALUE, "STEP0", "BadRequestError"])
def test_error_type_and_code_never_reach_the_message(echoed: str) -> None:
    body = {"error": {"message": "secret", "type": echoed, "code": echoed}}
    server = SystemOneServer([json_response(400, body)])

    with pytest.raises(SystemOneRequestError) as excinfo:
        make_client(server).decide("STEP0")

    assert str(excinfo.value) == "System One /v1/systemone returned HTTP 400"
    assert excinfo.value.raw == body


def test_openjev_422_detail_body_maps_to_request_error() -> None:
    server = SystemOneServer([json_response(422, {"detail": [{"msg": f"ContextTooLong {STATE}"}]})])

    with pytest.raises(SystemOneRequestError) as excinfo:
        make_client(server).decide(STATE)

    assert STATE not in str(excinfo.value)


def test_non_json_error_body_keeps_no_raw() -> None:
    server = SystemOneServer([httpx.Response(500, text=f"Internal error {SYSTEMONE_KEY_VALUE}")])

    with pytest.raises(SystemOneServerError) as excinfo:
        make_client(server).decide(STATE)

    assert excinfo.value.raw is None
    assert SYSTEMONE_KEY_VALUE not in str(excinfo.value)


@pytest.mark.parametrize(
    ("error_type", "timed_out"),
    [
        (httpx.ConnectError, False),
        (httpx.RemoteProtocolError, False),
        (httpx.ConnectTimeout, True),
        (httpx.ReadTimeout, True),
        (httpx.WriteTimeout, True),
    ],
)
def test_transport_failure_maps_to_transport_error(error_type: type[httpx.TransportError], timed_out: bool) -> None:
    server = SystemOneServer([raise_transport(error_type)])

    with pytest.raises(SystemOneTransportError) as excinfo:
        make_client(server).decide(STATE)

    assert excinfo.value.timed_out is timed_out
    assert excinfo.value.status_code is None
    assert SYSTEMONE_KEY_VALUE not in str(excinfo.value)


def test_health_maps_401_to_auth_error() -> None:
    server = SystemOneServer([json_response(401, {"error": {"message": "unauthorized"}})])

    with pytest.raises(SystemOneAuthError):
        make_client(server).health()


# --- protocol errors on 200 ------------------------------------------------------------------------


def _answers(**answers: Any) -> dict[str, Any]:
    return systemone_body(answers)


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {"answers": {"changed": noul_answer(0.5)}, "usage": {"input_tokens": 1, "output_tokens": 0}},
        {**_answers(changed=noul_answer(0.5)), "model": ""},
        {**_answers(changed=noul_answer(0.5)), "answers": []},
        _answers(),
        _answers(changed=noul_answer(0.5), extra=noul_answer(0.5)),
        _answers(changed={"noul": 0.5}),
        _answers(changed={"type": "choice", "noul": 0.5}),
        _answers(changed={"type": "noul"}),
        _answers(changed={"type": "noul", "noul": 1.5}),
        _answers(changed={"type": "noul", "noul": -0.1}),
        _answers(changed={"type": "noul", "noul": True}),
        _answers(changed={"type": "noul", "noul": "0.5"}),
        _answers(changed=[0.5]),
        {**_answers(changed=noul_answer(0.5)), "usage": None},
        {**_answers(changed=noul_answer(0.5)), "usage": {"input_tokens": 1}},
        {**_answers(changed=noul_answer(0.5)), "usage": {"input_tokens": True, "output_tokens": 0}},
        {**_answers(changed=noul_answer(0.5)), "usage": {"input_tokens": -1, "output_tokens": 0}},
    ],
)
def test_malformed_200_body_raises_protocol_error(body: Any) -> None:
    server = SystemOneServer([json_response(200, body)])

    with pytest.raises(SystemOneProtocolError):
        make_client(server).decide(STATE)


@pytest.mark.parametrize(
    "answers",
    [
        {"kind": {"type": "choice", "choice": "dive", "probabilities": {"turn": 0.5, "climb": 0.5}}},
        {"kind": {"type": "choice", "choice": "turn", "probabilities": {"turn": 1.0}}},
        {"kind": {"type": "choice", "choice": "turn", "probabilities": {"turn": 0.5, "climb": float("nan")}}},
        {"level": {"type": "score", "score": 2.5, "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}},
        {"level": {"type": "score", "score": -0.5, "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}},
        {"level": {"type": "score", "score": 1.0, "probabilities": {"0": 0.5, "1": 0.5}}},
        {"level": {"type": "score", "score": False, "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}},
    ],
)
def test_malformed_choice_or_score_answer_raises_protocol_error(answers: dict[str, Any]) -> None:
    valid = {
        "changed": noul_answer(0.5),
        "kind": {"type": "choice", "choice": "turn", "probabilities": {"turn": 0.5, "climb": 0.5}},
        "level": {"type": "score", "score": 1.0, "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}},
    }
    server = SystemOneServer([json_response(200, systemone_body({**valid, **answers}))])

    with pytest.raises(SystemOneProtocolError):
        make_client(server, questions=ALL_QUESTIONS).decide(STATE)


def test_non_json_200_body_raises_protocol_error() -> None:
    server = SystemOneServer([httpx.Response(200, text="not json")])

    with pytest.raises(SystemOneProtocolError):
        make_client(server).decide(STATE)
