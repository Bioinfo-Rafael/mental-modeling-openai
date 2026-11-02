import pytest

from decision_experiment.metrics import aircraft_metrics
from decision_experiment.outputs import AIRCRAFT_SETS, parse_output
from decision_experiment.prompts import build_request, load_prompt
from decision_experiment.storage import HERE, read_json


def response(choice):
    # NONE is the largest individual choice, yet the sum of YES choices is 0.6.
    return {"answers": [{"name": "changed_aircraft", "type": "choice", "choice": choice,
                         "confidence": .4, "probabilities": [
                             {"value": "NONE", "probability": .4},
                             {"value": "A1", "probability": .2},
                             {"value": "A2", "probability": .2},
                             {"value": "BOTH", "probability": .2}]}]}


@pytest.mark.parametrize("choice", list(AIRCRAFT_SETS))
def test_four_choices_preserve_identity_and_derive_binary(choice):
    parsed = parse_output(response(choice), "decisions", {}, decisions_task="aircraft_set")
    assert parsed["decision"] == ("NO" if choice == "NONE" else "YES")
    assert parsed["selected_aircraft"] == AIRCRAFT_SETS[choice]
    assert parsed["aircraft_choice"] == choice
    assert parsed["events"] is None  # No invented steps.
    assert parsed["probability_yes"] == pytest.approx(.6)
    assert parsed["aircraft_probabilities"] == {"A0100": .4, "A0200": .4}


def test_aircraft_output_requires_all_four_probabilities_and_right_question():
    raw = response("A1")
    raw["answers"][0]["probabilities"].pop()
    with pytest.raises(ValueError, match="probabilities"):
        parse_output(raw, "decisions", {}, decisions_task="aircraft_set")
    raw = response("A1")
    raw["answers"][0]["name"] = "explanation_needed"
    with pytest.raises(ValueError, match="named"):
        parse_output(raw, "decisions", {}, decisions_task="aircraft_set")


def test_aircraft_refusal_is_not_none():
    with pytest.raises(ValueError, match="refusal"):
        parse_output({"answers": [{"name": "changed_aircraft", "type": "refusal"}]},
                     "decisions", {}, decisions_task="aircraft_set")


def test_one_missing_aircraft_is_not_hidden_by_binary_true_positive():
    row = {"status": "ok", "classification": "TP", "aircraft_selection_supported": True,
           "expected_aircraft": ["A0100", "A0200"], "expected_aircraft_choice": "BOTH",
           "selected_aircraft": ["A0100"], "aircraft_choice": "A1", "aircraft_set_correct": False}
    stats = aircraft_metrics([row])
    assert stats["exact_set_accuracy_valid"] == 0
    assert stats["per_aircraft"]["A0100"]["TP"] == 1
    assert stats["per_aircraft"]["A0200"]["FN"] == 1
    assert stats["confusion_matrix_expected_rows_predicted_columns"]["BOTH"]["A1"] == 1


def test_latest_evaluation_has_two_conditions_and_same_shared_input():
    config = read_json(HERE / "configs/evaluate.json")
    assert config["conditions"] == ["responses_diagnostic", "decisions"]
    prompt = load_prompt(HERE / "prompts/v003")
    decisions = build_request("decisions", prompt, "recorded window", config)
    diagnostic = build_request("responses_diagnostic", prompt, "recorded window", config)
    assert decisions["input"][0] == diagnostic["input"][0]
    assert diagnostic["input"][1]["content"].startswith(decisions["input"][1]["content"])
    assert len(decisions["questions"][0]["choices"]) == 4
    assert "events" in diagnostic["text"]["format"]["schema"]["properties"]
