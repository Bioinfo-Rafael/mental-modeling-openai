"""Detector and translation layers: exact inputs, threshold, strict parsing, production adapters."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mtd_rl_xavier.llm.openai_responses import OpenAIResponsesClient
from mtd_rl_xavier.llm.system_one import (
    ChoiceQuestion,
    NoulQuestion,
    SystemOneClient,
    conservative_input_token_bound,
)
from mtd_rl_xavier.translator.cli import build_system_one_client
from mtd_rl_xavier.translator.detector import (
    DetectionContext,
    DetectorLayer,
    SystemOneDetectionModel,
    detector_state_bound,
    noul_question,
    render_detector_state,
)
from mtd_rl_xavier.translator.schema import (
    Instruction,
    IntentTranslation,
    ModelCallError,
    PreviousTranslation,
    TargetRef,
)
from mtd_rl_xavier.translator.translation import (
    OpenAIIntentModel,
    TranslationLayer,
    TranslationModelRequest,
    TranslationOutputError,
    parse_intent_output,
    system_prompt,
)
from tests.llm.llm_fakes import (
    APITimeoutError,
    DumpableResponse,
    ScriptedResponsesClient,
    SystemOneServer,
    json_response,
    message_item,
    noul_answer,
    raise_transport,
    refusal_item,
    responses_body,
    systemone_body,
)

from .tr_fakes import (
    FakeDetectionModel,
    FakeIntentModel,
    fake_detector_renderer,
    fake_translation_renderer,
    make_config,
)

KEY_ENV = "TEST_TRANSLATOR_DETECTOR_KEY"
TEAM_A = TargetRef("team", "A")
AIRCRAFT_A1 = TargetRef("aircraft", "A0100")
PREVIOUS = PreviousTranslation("s:team:A:000000", 0, IntentTranslation("[FAKE] hold formation", ("[FAKE] e1",)))


def _context(previous: PreviousTranslation = PREVIOUS) -> DetectionContext:
    trajectory = fake_detector_renderer(
        None,
        start_step=1,
        end_step=10,
        anchor_step=10,
        object_ids=("A0100", "A0200", "B0100", "B0200"),
        fields=("alive", "lon_deg"),
        float_precision=5,
    )
    return DetectionContext(TEAM_A, 10, ("A0100", "A0200"), ("B0100", "B0200"), previous, trajectory)


def test_detector_state_is_target_previous_intent_and_compact_window_only() -> None:
    state = render_detector_state(_context())

    assert state == (
        "Target: team A (aircraft A0100, A0200). Opponents: B0100, B0200.\n"
        "Previous intent translation for this target (from step 0):\n"
        "[FAKE] hold formation\n"
        "\n"
        "Trajectory window: steps 1-10, 0.2 s per step, one JSON line per step, newest last; "
        "relative_step = step - 10. Fields: alive, lon_deg. null = unavailable.\n"
        "FAKE-COMPACT anchor=10 start=1 objects=A0100,A0200,B0100,B0200"
    )
    assert "[FAKE] e1" not in state  # evidence is not part of the detector input


@pytest.mark.parametrize(("probability", "changed"), [(0.64, False), (0.65, True), (0.66, True)])
def test_threshold_below_at_and_above(probability: float, changed: bool) -> None:
    layer = DetectorLayer(FakeDetectionModel(probability), threshold=0.65)

    result = layer.detect("state")

    assert (result.changed, result.probability, result.threshold) == (changed, probability, 0.65)


def test_detector_state_bound_covers_any_intent_up_to_the_limit() -> None:
    questions = (noul_question(make_config().detector.question),)
    worst_intent = "\U0001f600" * 25  # 4 UTF-8 bytes per character
    context = _context(PreviousTranslation("e", 0, IntentTranslation(worst_intent, ("x",))))

    bound = detector_state_bound(context, questions=questions, max_intent_chars=25)

    assert bound == conservative_input_token_bound(state=render_detector_state(context), questions=questions)
    assert bound > detector_state_bound(context, questions=questions, max_intent_chars=24)


def test_committed_config_writes_japanese_and_the_prompt_asks_for_it() -> None:
    config = make_config().translation  # D42

    assert config.output_language == "Japanese"
    assert "Write both in Japanese." in system_prompt(
        output_language=config.output_language, max_intent_chars=config.max_intent_chars
    )


def test_detector_state_bound_holds_for_a_longest_japanese_intent_at_the_default_limit() -> None:
    config = make_config()
    questions = (noul_question(config.detector.question),)
    limit = config.translation.max_intent_chars
    japanese = "航空機は編隊を維持して北へ向かう。" * limit  # 3 UTF-8 bytes per character
    intent = japanese[:limit]
    context = _context(PreviousTranslation("e", 0, IntentTranslation(intent, ("x",))))

    actual = conservative_input_token_bound(state=render_detector_state(context), questions=questions)
    bound = detector_state_bound(context, questions=questions, max_intent_chars=limit)

    assert len(intent) == limit and len(intent.encode("utf-8")) == 3 * limit
    assert actual <= bound
    assert bound - actual >= limit  # the reserve (4 bytes/char) leaves at least one byte per character spare


# --- System One adapter ---------------------------------------------------------------------------


def _system_one_client(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> tuple[SystemOneClient, SystemOneServer]:
    monkeypatch.setenv(KEY_ENV, "local-test-key")
    connection = {"base_url": "http://127.0.0.1:8000", "uds_path": None, "api_key_env": KEY_ENV}
    config = make_config(detector__connection=connection)
    server = SystemOneServer(outcomes)
    return build_system_one_client(config.detector, transport=server.transport), server


def test_system_one_adapter_sends_exactly_the_logged_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = systemone_body({"unexplained_change": noul_answer(0.72, vllm_extras=True)}, vllm_extras=True)
    client, server = _system_one_client(monkeypatch, [json_response(200, body)])
    model = SystemOneDetectionModel(client, question_id="unexplained_change")
    state = render_detector_state(_context())

    reply = model.decide(state)

    assert server.requests[0].body == model.request_payload(state)
    assert server.requests[0].body["state"] == state
    assert model.question == {**make_config().detector.question.to_json()}
    assert (reply.probability, reply.model, reply.raw) == (0.72, "clef-flash", body)
    assert reply.usage == {"input_tokens": 500, "output_tokens": 0}


def test_system_one_timeout_is_uncertain_and_server_error_keeps_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _system_one_client(
        monkeypatch, [raise_transport(httpx.ReadTimeout), json_response(500, {"error": "boom"})]
    )
    model = SystemOneDetectionModel(client, question_id="unexplained_change")

    with pytest.raises(ModelCallError) as timeout:
        model.decide("state")
    with pytest.raises(ModelCallError) as server_error:
        model.decide("state")

    assert (timeout.value.error_type, timeout.value.uncertain) == ("SystemOneTransportError", True)
    assert (server_error.value.error_type, server_error.value.uncertain) == ("SystemOneServerError", False)
    assert server_error.value.raw == {"error": "boom"}
    assert "local-test-key" not in str(timeout.value) + str(server_error.value)


def test_system_one_adapter_requires_one_noul_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "local-test-key")
    choice = ChoiceQuestion("pick", "Pick one.", (("a", None), ("b", None)))
    noul = NoulQuestion("unexplained_change", "Q", "yes", "no")
    clients = [
        SystemOneClient(
            base_url="http://127.0.0.1:8000",
            uds_path=None,
            model="clef-flash",
            questions=questions,
            input_token_budget=16384,
            timeout_s=5,
            api_key_env=KEY_ENV,
            transport=SystemOneServer([]).transport,
        )
        for questions in ((choice,), (noul, choice))
    ]

    for client in clients:
        with pytest.raises(ValueError, match="exactly one noul"):
            SystemOneDetectionModel(client, question_id="unexplained_change")


# --- translation layer ----------------------------------------------------------------------------


def _translation_request(previous: PreviousTranslation | None = PREVIOUS) -> TranslationModelRequest:
    instructions = (
        Instruction("ia", 0, AIRCRAFT_A1, "[PLACEHOLDER] aircraft task"),
        Instruction("ta", 20, TEAM_A, "[PLACEHOLDER] revised team task"),
    )
    trajectory = fake_translation_renderer(None, start_step=0, end_step=25, anchor_step=25, episode_alias="episode_x")
    return TranslationModelRequest(
        TEAM_A, ("A0100", "A0200"), 25, "episode_x", instructions, "ta", trajectory, previous
    )


def test_translation_prompt_carries_all_instructions_primary_trajectory_previous_and_language() -> None:
    layer = TranslationLayer(FakeIntentModel(), output_language="Japanese", max_intent_chars=321)

    prompt = layer.build_prompt(_translation_request())

    assert "[PLACEHOLDER] aircraft task" in prompt.user_text
    assert (
        "- issued at step 20 for team A | THIS TARGET'S INSTRUCTION:\n  [PLACEHOLDER] revised team task"
        in prompt.user_text
    )
    assert "- issued at step 0 for aircraft A0100:\n" in prompt.user_text
    assert "BEGIN WINDOW DATA\nFAKE-LAG17 0..25 anchor=25\nEND WINDOW DATA" in prompt.user_text
    assert "From step 0:\nintent: [FAKE] hold formation\nevidence:\n  - [FAKE] e1" in prompt.user_text
    assert "Write both in Japanese." in prompt.instructions and "at most 321 characters" in prompt.instructions
    assert prompt.schema["required"] == ["intent", "evidence"] and prompt.schema["additionalProperties"] is False
    assert "s:team:A" not in prompt.user_text  # no event / scenario ids reach the model


def test_first_translation_prompt_says_there_is_no_previous_translation() -> None:
    layer = TranslationLayer(FakeIntentModel(), output_language="English", max_intent_chars=100)

    prompt = layer.build_prompt(_translation_request(previous=None))

    assert "(none: this is the first translation for this target)" in prompt.user_text


def test_parse_accepts_exactly_one_valid_object_and_keeps_text_unchanged() -> None:
    parsed = parse_intent_output('{"intent": " Climb. ", "evidence": ["a", "b"]}', max_intent_chars=8)

    assert parsed == IntentTranslation(" Climb. ", ("a", "b"))


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"intent": "x", "evidence": ["e"]} {"intent": "y"}',
        '[{"intent": "x", "evidence": ["e"]}]',
        '{"intent": "x"}',
        '{"intent": "x", "evidence": ["e"], "extra": 1}',
        '{"intent": "  ", "evidence": ["e"]}',
        '{"intent": 3, "evidence": ["e"]}',
        '{"intent": "x", "evidence": []}',
        '{"intent": "x", "evidence": "e"}',
        '{"intent": "x", "evidence": ["e", " "]}',
        '{"intent": "x", "evidence": [1]}',
        '{"intent": "123456789", "evidence": ["e"]}',
        '{"intent": "x\\ud800", "evidence": ["e"]}',
        '{"intent": "x", "evidence": ["\\udfff"]}',
    ],
)
def test_parse_rejects_anything_else(text: str) -> None:
    with pytest.raises(TranslationOutputError):
        parse_intent_output(text, max_intent_chars=8)


def _openai_model(client: ScriptedResponsesClient) -> OpenAIIntentModel:
    return OpenAIIntentModel(
        OpenAIResponsesClient(client), model="gpt-6.1-terra", reasoning_effort="medium", max_output_tokens=4000
    )


def test_openai_adapter_sends_exactly_the_logged_payload() -> None:
    answer = json.dumps({"intent": "[FAKE] intent", "evidence": ["[FAKE] e"]})
    client = ScriptedResponsesClient([DumpableResponse(responses_body(output=[message_item(answer)]))])
    model = _openai_model(client)
    layer = TranslationLayer(model, output_language="English", max_intent_chars=1000)
    prompt = layer.build_prompt(_translation_request())

    output = layer.generate(prompt)

    assert client.requests == [layer.request_payload(prompt)]
    sent = client.requests[0]
    assert (sent["model"], sent["reasoning"], sent["max_output_tokens"], sent["store"]) == (
        "gpt-6.1-terra",
        {"effort": "medium"},
        4000,
        False,
    )
    assert sent["text"]["format"]["strict"] is True
    assert sent["input"][0]["content"][0]["text"] == prompt.user_text
    assert "temperature" not in sent
    assert layer.parse(output) == IntentTranslation("[FAKE] intent", ("[FAKE] e",))
    assert output.usage == {"input_tokens": 120, "cached_input_tokens": 20, "output_tokens": 40, "reasoning_tokens": 30}


def test_openai_refusal_keeps_raw_and_timeout_is_uncertain() -> None:
    refusal = responses_body(output=[refusal_item("[FAKE] no")])
    client = ScriptedResponsesClient([DumpableResponse(refusal), APITimeoutError("slow")])
    layer = TranslationLayer(_openai_model(client), output_language="English", max_intent_chars=1000)
    prompt = layer.build_prompt(_translation_request())

    with pytest.raises(ModelCallError) as refused:
        layer.generate(prompt)
    with pytest.raises(ModelCallError) as timed_out:
        layer.generate(prompt)

    assert (refused.value.error_type, refused.value.uncertain, refused.value.raw) == (
        "OpenAIRefusalError",
        False,
        refusal,
    )
    assert (timed_out.value.error_type, timed_out.value.uncertain) == ("OpenAIClientError", True)


@pytest.mark.parametrize(
    ("text", "case"),
    [
        ('{"intent": "shown", "intent": "parsed", "evidence": ["e"]}', "repeated key"),
        ('{"intent": "x", "evidence": ["e"], "evidence": ["f"]}', "repeated key"),
        ("[" * 100_000, "deep nesting raises RecursionError inside json"),
    ],
)
def test_parse_rejects_repeated_keys_and_deep_nesting_as_output_errors(text: str, case: str) -> None:
    with pytest.raises(TranslationOutputError, match="repeated keys"):
        parse_intent_output(text, max_intent_chars=8)
