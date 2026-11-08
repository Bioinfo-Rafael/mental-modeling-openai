"""Run-config and scenario contracts: exact keys, no hidden model, phase limits, scenario v2 rules."""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from translate import config as translator_config
from translate import paths
from translate.config import (
    ConfigError,
    load_config,
    parse_config,
)
from translate.events import TargetRef
from translate.scenario import (
    PipelineError,
    ScenarioError,
    episode_facts,
    load_scenario,
    parse_scenario,
    validate_scenario,
)

from .tr_fakes import (
    DELETE,
    EXAMPLE_SCENARIO,
    LOSS01,
    RUN_CONFIG,
    WIN01,
    alarm,
    instruction,
    make_config,
    make_scenario,
    request,
    run_config_raw,
    scenario_raw,
    with_value,
)

LIMITS = make_config().scenario_limits


def _config_error(dotted: str, value: Any, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _parse_config(with_value(run_config_raw(), dotted, value))


def _parse_config(raw: object):
    return parse_config(raw, path=RUN_CONFIG, sha256="test")


def test_default_run_yaml_loads_with_explicit_values() -> None:
    config = load_config(RUN_CONFIG)

    assert config.translation.model == "gpt-6.1-terra"
    assert config.translation.backend == "openai_responses"
    assert config.detector.served_model_name == "clef-flash"
    assert config.detector.input_token_budget == 16384
    assert config.detector.window.float_precision == 5
    assert config.detector.window.past_steps == 10
    assert config.trajectory.past_steps == 60
    assert config.triggers.tick_interval_steps == 5
    assert config.detector is not None and config.code_session is False
    assert config.output_root == paths.ROOT / "outputs" / "translate"
    assert config.answer.limits.max_intent_chars == 600  # D61
    assert config.detector.messages_lookback_steps == 25
    assert (LIMITS.max_instruction_chars, LIMITS.max_alarm_chars, LIMITS.max_question_chars) == (300, 200, 200)
    assert (config.detector.question.question_id, config.detector.question.question_type) == ("point_kind", "choice")
    assert [label for label, _ in config.detector.question.criteria] == ["none", "start", "change"]
    assert config.detector.threshold == 0.65


@pytest.mark.parametrize(
    "section",
    [
        "scenario",
        "triggers",
        "visibility",
        "limits",
        "scenario_limits",
        "detector",
        "inputs",
        "trajectory",
        "policy",
        "prompt",
        "translation",
        "answer",
        "session",
        "output",
        "pricing",
    ],
)
def test_config_without_a_section_is_rejected(section: str) -> None:
    _config_error(section, DELETE, "exactly")


@pytest.mark.parametrize(
    "dotted",
    [
        "extra",
        "triggers.extra",
        "visibility.extra",
        "scenario_limits.extra",
        "detector.extra",
        "detector.question.criteria.extra",
        "detector.window.extra",
        "trajectory.extra",
    ],
)
def test_config_with_unknown_key_is_rejected(dotted: str) -> None:
    _config_error(dotted, 1, "unknown")


@pytest.mark.parametrize("value", [DELETE, "", "   ", None])
def test_translation_model_has_no_hidden_default(value: object) -> None:
    match = "exactly" if value is DELETE else "translation.model"
    _config_error("translation.model", value, match)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_caps_must_be_positive_integers_and_not_bool(value: object) -> None:
    _config_error("limits.max_translation_calls", value, "positive integer")


def test_retrospective_mode_is_rejected_as_not_in_this_phase() -> None:
    _config_error("visibility.mode", "retrospective", "visibility.mode")


def test_local_chat_translation_backend_is_rejected() -> None:
    _config_error("translation.backend", "local_chat", r"translation.backend must be one of \['openai_responses'\]")


def test_online_mode_requires_zero_future_steps() -> None:
    _config_error("visibility.future_steps", 5, "must be 0")


@pytest.mark.parametrize("value", [0, 0.0, 0.65, 1, 1.0])
def test_threshold_boundaries_are_accepted(value: float) -> None:
    raw = with_value(run_config_raw(), "detector.threshold", value)

    assert _parse_config(raw).detector.threshold == float(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, True, "0.5", float("nan")])
def test_threshold_outside_unit_interval_or_wrong_type_is_rejected(value: object) -> None:
    _config_error("detector.threshold", value, "threshold")


def test_detector_budget_above_serving_cap_is_rejected_and_cap_is_accepted() -> None:
    _config_error("detector.input_token_budget", 16385, "must not exceed 16384")

    assert (
        _parse_config(with_value(run_config_raw(), "detector.input_token_budget", 4096)).detector.input_token_budget
        == 4096
    )


@pytest.mark.parametrize("label", ["none", "start", "change"])
def test_detector_question_needs_every_choice_option(label: str) -> None:
    _config_error(f"detector.question.criteria.{label}", DELETE, "missing")


@pytest.mark.parametrize("value", ["noul", "score", None])
def test_detector_question_must_be_a_choice_question(value: object) -> None:
    _config_error("detector.question.type", value, "choice")


@pytest.mark.parametrize(("lookback", "accepted"), [(4, False), (5, True), (25, True)])
def test_messages_lookback_must_cover_the_tick_interval(lookback: int, accepted: bool) -> None:
    raw = with_value(run_config_raw(), "detector.messages_lookback_steps", lookback)

    if accepted:
        assert _parse_config(raw).detector.messages_lookback_steps == lookback
        return
    with pytest.raises(ConfigError, match="messages_lookback_steps must be >= triggers.tick_interval_steps"):
        _parse_config(raw)


@pytest.mark.parametrize("key", ["max_instruction_chars", "max_alarm_chars", "max_question_chars"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_scenario_limits_must_be_positive_integers(key: str, value: object) -> None:
    _config_error(f"scenario_limits.{key}", value, "positive integer")


def test_unquoted_true_false_criteria_keys_are_rejected_with_a_hint() -> None:
    raw = with_value(run_config_raw(), "detector.question.criteria", {True: "a", False: "b"})

    with pytest.raises(ConfigError, match="quote"):
        _parse_config(raw)


@pytest.mark.parametrize(
    ("connection", "match"),
    [
        ({"base_url": "http://127.0.0.1:8000", "uds_path": "/run/x.sock", "api_key_env": "K"}, "exactly one"),
        ({"base_url": None, "uds_path": None, "api_key_env": "K"}, "exactly one"),
        ({"base_url": "http://10.0.0.5:8000", "uds_path": None, "api_key_env": "K"}, "127.0.0.1"),
        ({"base_url": "http://0.0.0.0:8000", "uds_path": None, "api_key_env": "K"}, "127.0.0.1"),
        ({"base_url": "http://127.0.0.1:8000", "uds_path": None, "api_key_env": None}, "required"),
        ({"base_url": None, "uds_path": "relative.sock", "api_key_env": None}, "absolute"),
        ({"base_url": "http://127.0.0.1:8000", "uds_path": None, "api_key_env": "not a name"}, "api_key_env"),
    ],
)
def test_detector_connection_rules(connection: dict[str, Any], match: str) -> None:
    _config_error("detector.connection", connection, match)


def test_uds_connection_without_key_is_accepted() -> None:
    connection = {"base_url": None, "uds_path": "/run/user/vllm/clef.sock", "api_key_env": None}

    config = _parse_config(with_value(run_config_raw(), "detector.connection", connection))

    assert config.detector.connection.uds_path == Path("/run/user/vllm/clef.sock")


@pytest.mark.parametrize("fields", [["alive", "nope"], ["step"], ["alive", "alive"], [], "alive"])
def test_detector_fields_must_be_known_unique_and_not_reserved(fields: object) -> None:
    _config_error("detector.window.fields", fields, "fields")


def test_translation_connection_is_unknown_and_effort_must_be_known() -> None:
    _config_error("translation.connection", {"base_url": "x"}, "exactly")
    _config_error("translation.reasoning_effort", "extreme", "reasoning_effort")


def test_translation_model_needs_a_pricing_entry() -> None:
    _config_error("translation.model", "gpt-unpriced", "pricing has no entry")


def test_pricing_rejects_negative_and_accepts_null_cached_input() -> None:
    negative = {"gpt-6.1-terra": {"input": -1, "cached_input": None, "output": 5}}
    _config_error("pricing", negative, "non-negative")

    config = load_config(RUN_CONFIG)

    assert config.price("gpt-6.1-terra").cached_input is None


def test_absolute_output_root_is_kept(tmp_path: Path) -> None:
    raw = with_value(run_config_raw(), "output.root", str(tmp_path))

    assert _parse_config(raw).output_root == tmp_path


def test_schema_version_must_be_two_and_not_bool() -> None:
    _config_error("schema_version", True, "schema_version")
    _config_error("schema_version", 1, "schema_version")


# --- scenario -------------------------------------------------------------------------------------


def test_example_scenario_resolves_paths_relative_to_its_file() -> None:
    scenario = load_scenario(EXAMPLE_SCENARIO, limits=LIMITS)

    assert scenario.schema_version == 2
    assert scenario.episode.csv_path.is_file()
    assert scenario.episode.csv_path.name == f"{WIN01}.csv"
    assert all("[PLACEHOLDER]" in item.text for item in (*scenario.instructions, *scenario.alarms))
    assert all("[PLACEHOLDER]" in item.question for item in scenario.requests)
    assert [item.issue_step for item in scenario.instructions] == [0, 0, 20]
    assert [(item.alarm_id, item.step) for item in scenario.alarms] == [("placeholder-alarm-001", 12)]
    assert [(item.request_id, item.step) for item in scenario.requests] == [("placeholder-request-001", 10)]


def test_scenario_without_step_zero_instruction_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="step 0"):
        make_scenario([instruction("i1", 3, "aircraft", "A0100")])


@pytest.mark.parametrize(
    "duplicate",
    [
        {"instructions": [instruction("x", 0, "aircraft", "A0100"), instruction("x", 5, "team", "A")]},
        {
            "instructions": [instruction("x", 0, "aircraft", "A0100")],
            "requests": [request("x", 5, ("aircraft", "A0100"))],
        },
        {"instructions": [instruction("x", 0, "aircraft", "A0100")], "alarms": [alarm("x", 5, ("aircraft", "A0100"))]},
        {
            "instructions": [instruction("i1", 0, "aircraft", "A0100")],
            "requests": [request("x", 5, ("aircraft", "A0100"))],
            "alarms": [alarm("x", 6, ("aircraft", "A0100"))],
        },
    ],
)
def test_ids_must_be_unique_across_the_scenario(duplicate: dict[str, Any]) -> None:
    with pytest.raises(ScenarioError, match="unique"):
        make_scenario(**duplicate)


def test_two_instructions_for_one_target_at_one_step_are_rejected() -> None:
    with pytest.raises(ScenarioError, match="one instruction per exact target"):
        make_scenario([instruction("i1", 0, "team", "A"), instruction("i2", 0, "team", "A")])


def test_request_before_the_targets_first_instruction_is_rejected() -> None:
    instructions = [instruction("i1", 0, "aircraft", "A0100"), instruction("i2", 20, "team", "B")]

    with pytest.raises(ScenarioError, match="r1 references team B"):
        make_scenario(instructions, [request("r1", 19, ("team", "B"))])


def test_alarm_for_a_never_instructed_target_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="c1 references aircraft B0100"):
        make_scenario([instruction("i1", 0, "aircraft", "A0100")], alarms=[alarm("c1", 5, ("aircraft", "B0100"))])


def test_alarm_before_one_of_its_targets_first_instruction_is_rejected() -> None:
    instructions = [instruction("i1", 0, "aircraft", "A0100"), instruction("i2", 20, "team", "B")]

    with pytest.raises(ScenarioError, match="w1 references team B"):
        make_scenario(instructions, alarms=[alarm("w1", 19, ("aircraft", "A0100"), ("team", "B"))])
    assert make_scenario(instructions, alarms=[alarm("w1", 20, ("aircraft", "A0100"), ("team", "B"))]).alarms


def test_two_requests_for_one_target_at_one_step_are_rejected() -> None:
    instructions = [instruction("i1", 0, "aircraft", "A0100"), instruction("i2", 0, "team", "A")]
    one_per_target = [request("r1", 5, ("aircraft", "A0100")), request("r2", 5, ("team", "A"))]

    with pytest.raises(ScenarioError, match="one request per exact target per step"):
        make_scenario(instructions, [request("r1", 5, ("aircraft", "A0100")), request("r2", 5, ("aircraft", "A0100"))])
    assert len(make_scenario(instructions, one_per_target).requests) == 2


def test_v1_scenario_with_external_change_points_is_rejected() -> None:
    v1 = {**scenario_raw([instruction("i1", 0, "aircraft", "A0100")]), "schema_version": 1}
    v2_with_v1_field = {**scenario_raw([instruction("i1", 0, "aircraft", "A0100")]), "external_change_points": []}

    with pytest.raises(ScenarioError, match="schema_version must be 2"):
        parse_scenario(v1, base=paths.ROOT, limits=LIMITS)
    with pytest.raises(ScenarioError, match=r"unknown \['external_change_points'\]"):
        parse_scenario(v2_with_v1_field, base=paths.ROOT, limits=LIMITS)


@pytest.mark.parametrize(
    ("kind", "limit"),
    [("instruction", "max_instruction_chars"), ("alarm", "max_alarm_chars"), ("question", "max_question_chars")],
)
def test_scenario_texts_at_the_limit_load_and_one_more_character_fails(kind: str, limit: str) -> None:
    size = getattr(LIMITS, limit)

    def build(length: int) -> Any:
        text = "航" * length  # characters, not bytes
        instructions = [instruction("i1", 0, "aircraft", "A0100", text if kind == "instruction" else "")]
        alarms = [alarm("w1", 3, ("aircraft", "A0100"), text=text)] if kind == "alarm" else []
        requests = [request("r1", 3, ("aircraft", "A0100"), question=text)] if kind == "question" else []
        return make_scenario(instructions, requests, alarms)

    assert build(size).scenario_id == "test-scenario"
    with pytest.raises(ScenarioError, match=f"has {size + 1} characters; the limit is {size}"):
        build(size + 1)


@pytest.mark.parametrize(
    "alias",
    [
        "mhi_episode",
        "episode-v1",
        "ic100601",
        "replay.3",
        "MHI_x",
        "episode_LOSS",
        "Won",
        "lost_x",
    ],
)
def test_alias_sharing_an_episode_token_or_naming_an_outcome_is_rejected(alias: str) -> None:
    raw = with_value(scenario_raw([instruction("i1", 0, "aircraft", "A0100")]), "episode.alias", alias)

    with pytest.raises(ScenarioError, match="episode.alias must"):
        parse_scenario(raw, base=paths.ROOT, limits=LIMITS)


@pytest.mark.parametrize("alias", ["loss_01", "ep-win", "WIN", "x.Lose", "won-7", "LOST"])
def test_alias_with_an_outcome_word_token_is_rejected(alias: str) -> None:
    raw = scenario_raw([instruction("i1", 0, "aircraft", "A0100")])
    raw = with_value(with_value(raw, "episode.episode_id", "recording_07"), "episode.alias", alias)

    with pytest.raises(ScenarioError, match="outcome word"):
        parse_scenario(raw, base=paths.ROOT, limits=LIMITS)


@pytest.mark.parametrize(
    ("episode_id", "alias"),
    [(LOSS01, "seed100100_x"), (LOSS01, "x-ic100601"), (WIN01, "IC100601"), (WIN01, "replay_v1")],
)
def test_alias_reusing_an_episode_seed_or_ic_token_is_rejected(episode_id: str, alias: str) -> None:
    raw = scenario_raw([instruction("i1", 0, "aircraft", "A0100")], episode_id=episode_id)

    with pytest.raises(ScenarioError, match="share no token"):
        parse_scenario(with_value(raw, "episode.alias", alias), base=paths.ROOT, limits=LIMITS)


@pytest.mark.parametrize(
    "alias", ["episode_x", "episode_01", "replay-07", "e.1", "window_01", "close_call", "winner_01", "close_01"]
)
def test_neutral_alias_is_accepted(alias: str) -> None:
    raw = with_value(scenario_raw([instruction("i1", 0, "aircraft", "A0100")]), "episode.alias", alias)

    assert parse_scenario(raw, base=paths.ROOT, limits=LIMITS).episode.alias == alias


def test_request_at_the_targets_first_instruction_step_is_accepted() -> None:
    instructions = [instruction("i1", 0, "aircraft", "A0100"), instruction("i2", 20, "team", "B")]

    scenario = make_scenario(instructions, [request("r1", 20, ("team", "B"))])

    assert scenario.requests[0].targets == (TargetRef("team", "B"),)


@pytest.mark.parametrize(
    ("dotted", "value", "match"),
    [
        ("episode.alias", WIN01, "alias must differ"),
        ("extra", 1, "unknown"),
        ("instructions", [], "must not be empty"),
        ("requests", None, "list"),
        ("alarms", None, "list"),
        ("alarms", [{"alarm_id": "w1", "step": 1, "targets": [], "text": "x"}], "distinct"),
        (
            "alarms",
            [{"alarm_id": "w1", "step": 1, "targets": [{"kind": "aircraft", "id": "A0100"}], "text": " "}],
            "text",
        ),
        ("alarms", [{"alarm_id": "w1", "step": 1, "targets": [{"kind": "aircraft", "id": "A0100"}]}], "missing"),
        ("requests", [{**request("r1", 1, ("aircraft", "A0100")), "question": " "}], "question"),
        (
            "requests",
            [{k: v for k, v in request("r1", 1, ("aircraft", "A0100")).items() if k != "question"}],
            "missing",
        ),
        ("instructions.0.issue_step", True, "non-negative integer"),
        ("instructions.0.issue_step", -1, "non-negative integer"),
        ("instructions.0.text", " ", "text"),
        ("instructions.0.target", {"kind": "squadron", "id": "A"}, "kind"),
        ("scenario_id", "has space", "scenario_id"),
    ],
)
def test_malformed_scenarios_are_rejected(dotted: str, value: object, match: str) -> None:
    raw = scenario_raw([instruction("i1", 0, "aircraft", "A0100")])
    raw = (
        with_value(raw, dotted, value)
        if not dotted.startswith("instructions.0")
        else _set_instruction(raw, dotted, value)
    )

    with pytest.raises(ScenarioError, match=match):
        parse_scenario(raw, base=paths.ROOT, limits=LIMITS)


def _set_instruction(raw: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    key = dotted.split(".")[-1]
    return {**raw, "instructions": [{**raw["instructions"][0], key: value}]}


def test_request_with_no_targets_or_duplicate_targets_is_rejected() -> None:
    instructions = [instruction("i1", 0, "aircraft", "A0100")]

    with pytest.raises(ScenarioError, match="distinct"):
        make_scenario(instructions, [request("r1", 5)])
    with pytest.raises(ScenarioError, match="distinct"):
        make_scenario(instructions, [request("r1", 5, ("aircraft", "A0100"), ("aircraft", "A0100"))])


# --- episode-dependent checks ---------------------------------------------------------------------


def test_steps_are_checked_against_the_last_complete_step(frames: dict[str, pd.DataFrame]) -> None:
    facts = episode_facts(frames[WIN01])
    instructions = [instruction("i1", 0, "aircraft", "A0100")]

    validate_scenario(make_scenario(instructions, [request("r1", 340, ("aircraft", "A0100"))]), facts, LIMITS)
    validate_scenario(make_scenario(instructions, alarms=[alarm("w1", 340, ("aircraft", "A0100"))]), facts, LIMITS)
    with pytest.raises(PipelineError, match=r"\[0, 340\]"):
        validate_scenario(make_scenario(instructions, [request("r1", 341, ("aircraft", "A0100"))]), facts, LIMITS)
    with pytest.raises(PipelineError, match=r"\[0, 340\]"):
        validate_scenario(make_scenario(instructions, alarms=[alarm("w1", 341, ("aircraft", "A0100"))]), facts, LIMITS)


@pytest.mark.parametrize(("kind", "target_id"), [("aircraft", "A0300"), ("team", "C")])
def test_unknown_aircraft_or_team_is_rejected(frames: dict[str, pd.DataFrame], kind: str, target_id: str) -> None:
    scenario = make_scenario([instruction("i1", 0, kind, target_id)])

    with pytest.raises(PipelineError, match="unknown"):
        validate_scenario(scenario, episode_facts(frames[WIN01]), LIMITS)


def test_scenario_episode_id_must_match_the_loaded_episode(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("i1", 0, "aircraft", "A0100")])

    with pytest.raises(PipelineError, match="episode_id"):
        validate_scenario(scenario, episode_facts(frames["ours_seed100100_ic100601_loss"]), LIMITS)


def test_episode_facts_teams_last_steps_and_down_steps(frames: dict[str, pd.DataFrame]) -> None:
    facts = {episode_id: episode_facts(frame) for episode_id, frame in frames.items()}

    assert {k: f.last_step for k, f in facts.items()} == {
        "mhi_v1.3_ic100601_win": 340,
        "ours_seed100100_ic100601_loss": 293,
        "ours_seed100100_ic100606_win": 495,
    }
    assert dict(facts[WIN01].teams) == {"A": ("A0100", "A0200"), "B": ("B0100", "B0200")}
    assert dict(facts["ours_seed100100_ic100601_loss"].down_steps) == {
        "A0100": None,
        "A0200": 210,
        "B0100": None,
        "B0200": 137,
    }
    assert facts[WIN01].down_step(TargetRef("team", "B")) == 335  # B0200 at 136, B0100 at 335
    assert facts[WIN01].down_step(TargetRef("team", "A")) is None
    assert facts[WIN01].opponents(TargetRef("aircraft", "A0100")) == ("B0100", "B0200")


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    text = RUN_CONFIG.read_text(encoding="utf-8").replace(
        "  model: gpt-6.1-terra ", "  model: gpt-6.1-terra\n  model: gpt-other ", 1
    )
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate key 'model'"):
        load_config(path)


# --- detector endpoint (D38 security L1, D39) ------------------------------------------------------


def test_default_detector_endpoint_is_a_unix_socket_in_a_per_user_directory() -> None:
    connection = load_config(RUN_CONFIG).detector.connection

    assert connection.base_url is None and connection.api_key_env == "VLLM_API_KEY"
    assert connection.uds_path == Path(f"/tmp/{getpass.getuser()}-vllm/clef-flash.sock")


@pytest.mark.parametrize(
    "uds_path",
    ["/tmp/{USER}-vllm/x.sock", "/tmp/$USER-vllm/x.sock", "/tmp/${user}/x.sock", "/tmp/{user}/{home}/x.sock"],
)
def test_uds_path_supports_only_the_user_placeholder(uds_path: str) -> None:
    connection = {"base_url": None, "uds_path": uds_path, "api_key_env": None}

    _config_error("detector.connection", connection, "only the placeholder")


@pytest.mark.parametrize("user", ["../evil", "a/b", ""])
def test_unusable_user_name_for_the_placeholder_is_rejected(monkeypatch: pytest.MonkeyPatch, user: str) -> None:
    monkeypatch.setattr(translator_config.getpass, "getuser", lambda: user)

    _config_error("detector.connection.uds_path", "/tmp/{user}-vllm/x.sock", "user name")


def test_tcp_base_url_is_stored_as_the_validated_server_root() -> None:
    connection = {"base_url": "http://127.0.0.1:8000/", "uds_path": None, "api_key_env": "VLLM_API_KEY"}

    config = _parse_config(with_value(run_config_raw(), "detector.connection", connection))

    assert config.detector.connection.base_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "connection",
    [
        {"base_url": "http://user:SECRETPW@127.0.0.1:8000", "uds_path": None, "api_key_env": "K"},
        {"base_url": "http://127.0.0.1:8000/?token=SECRETPW", "uds_path": None, "api_key_env": "K"},
        {"base_url": "http://user" + chr(0xFF1A) + "SECRETPW@127.0.0.1:8000", "uds_path": None, "api_key_env": "K"},
        {"base_url": None, "uds_path": "/tmp/x.sock", "api_key_env": "SECRETPW-0123"},
        {"base_url": None, "uds_path": "/tmp/x.sock", "api_key_env": "secretpw0123"},
    ],
)
def test_detector_connection_secrets_are_rejected_without_echo(connection: dict[str, Any]) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _parse_config(with_value(run_config_raw(), "detector.connection", connection))

    assert "SECRETPW".lower() not in str(excinfo.value).lower()


def test_bad_request_targets_value_names_the_targets_field() -> None:
    raw = scenario_raw([instruction("a1", 0, "aircraft", "A0100")], [request("r1", 3, ("aircraft", "A0100"))])
    raw["requests"][0]["targets"] = "A0100"

    with pytest.raises(ScenarioError, match=r"requests\[0\]\.targets must be a list"):
        parse_scenario(raw, base=paths.ROOT, limits=LIMITS)
