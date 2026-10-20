"""Small hand-calculable cases; these tests make no network/API calls."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from joint_data import parse_response, parse_component, select_subsets, TASKS, bin_actions
from joint_metrics import nrmse_statistics, pearson, cosine, theta, wrap_to_pi, angular_pair
from joint_aggregate import accuracy_rows


def row(task, metric, response, **gt):
    return {"task": task, "metric": metric, "H": 5, "condition_id": "test", "components": parse_response(response, task, metric), **gt}


def test_mountaincar_action_matching():
    selected = [row(TASKS[0], "next-action", f"Final action choice: [{p}]", gt_action=[g]) for p, g in [(2, 2), (1, 0)]]
    accuracy, _, confusion = accuracy_rows(selected)
    assert accuracy[0]["value"] == 50
    assert sum(c["count"] for c in confusion) == 2


def test_pendulum_raw_and_bins_are_independent():
    parsed = parse_response("predictions = [0.6]\n>>Final action bins: [6]", TASKS[1], "next-action")
    assert parsed["action_value"]["value"] == [.6]
    assert parsed["action_bin"]["value"] == [6]
    assert bin_actions([-2, -1.6, 0, 2], start=-2, stop=2, bins=10) == [0, 0, 4, 9]
    selected = [row(TASKS[1], "last-action", f"predictions=[-2.]\nFinal action bins: [{p}]", gt_action_bin=[g]) for p, g in [(0, 0), (6, 4)]]
    metrics, _, _ = accuracy_rows(selected)
    assert metrics[0]["value"] == 50 and metrics[1]["value"] == 1


def test_state_components_and_direction_accuracy():
    text = 'predictions=["INC", "DEC"]\n- Final state values: [0.2, 0.01]\n5. [Final state deltas]:\n[0.1, -0.02]'
    parsed = parse_response(text, TASKS[0], "last-state")
    assert parsed["state_value"]["value"] == [.2, .01]
    assert parsed["state_delta"]["value"] == [.1, -.02]
    selected = [row(TASKS[0], "last-state", text, gt_direction=g) for g in ([1, 0], [1, 2])]
    accuracy, dimensions, _ = accuracy_rows(selected)
    assert accuracy[0]["value"] == 75 and accuracy[0]["valid_units"] == 4
    assert [d["value"] for d in dimensions] == [100, 50]


@pytest.mark.parametrize("text,error", [
    ("Reasoning says Final state values: [1,2]", "missing_marker"),
    ("Final state values: [v0,v1]", "non_literal_or_placeholder"),
    ("Final state values: [1]", "wrong_dimension"),
    ("Final state values: [True,2]", "non_finite_or_non_numeric"),
    ("Final state values: [1e999,2]", "non_finite_or_non_numeric"),
    ("Final state values: [1,2]\nFinal state values: [2,3]", "conflicting_markers")])
def test_reject_untrustworthy_numeric_output(text, error):
    assert parse_component(text, "Final state values", 2)["error"] == error


def test_component_failure_does_not_remove_other_outputs():
    parsed = parse_response('predictions=["DEC", "UNCH"]\nFinal state values:[v0,v1]\nFinal state deltas:[.1,.2]', TASKS[0], "next-state")
    assert parsed["direction"]["ok"] and parsed["state_delta"]["ok"] and not parsed["state_value"]["ok"]
    assert not parse_component("Final action bins:[10]", "Final action bins", 1, "bin")["ok"]
    assert not parse_component('predictions=["FLAT"]', "predictions", 1, "direction")["ok"]


def test_nrmse_macro_and_bootstrap():
    prediction = np.array([[1, 2], [2, 4]])
    truth = np.zeros((2, 2))
    result = nrmse_statistics(prediction, truth, [2, 2], 1000, 42)
    np.testing.assert_allclose(result["point"], [math.sqrt(2.5)/2, math.sqrt(10)/2])
    assert result["macro"] == pytest.approx(result["point"].mean())
    # Perfectly proportional dimension errors: shared resamples preserve this relation.
    assert result["bootstrap_std"][1] == pytest.approx(2*result["bootstrap_std"][0])
    assert result["macro_bootstrap_std"] == pytest.approx(1.5*result["bootstrap_std"][0])
    repeat = nrmse_statistics(prediction, truth, [2, 2], 1000, 42)
    np.testing.assert_array_equal(result["bootstrap_std"], repeat["bootstrap_std"])


def test_pearson_cosine_undefined():
    assert pearson([1, 2, 3], [3, 2, 1])[0] == pytest.approx(-1)
    assert pearson([1, 1], [2, 3]) == (None, "constant_vector")
    assert pearson([1], [2])[0] is None
    assert cosine([1, 0], [0, 1])[0] == 0
    assert cosine([1, 2], [1, 2])[0] == pytest.approx(1)
    assert cosine([0, 0], [1, 2]) == (None, "zero_norm_vector")


@pytest.mark.parametrize("metric", ["next-state", "last-state"])
@pytest.mark.parametrize("angles", [(0, .3), (3.1, -3.1)])
def test_delta_theta_forward_sign_both_metrics(metric, angles):
    a, b = angles
    early, late = np.array([np.cos(a), np.sin(a), .2]), np.array([np.cos(b), np.sin(b), .4])
    delta = late - early
    record = {"early": early.tolist(), "late": late.tolist(), "metric": metric,
              "components": {"state_value": {"ok": True, "value": (late if metric == "next-state" else early).tolist()},
                             "state_delta": {"ok": True, "value": delta.tolist()}}}
    assert theta([0, 1, 3]) == pytest.approx(np.pi/2)
    assert wrap_to_pi(3*np.pi) == pytest.approx(-np.pi)
    for component in record["components"]:
        predicted, gt, reason = angular_pair(record, component)
        assert not reason and predicted == pytest.approx(gt)
        assert gt == pytest.approx(wrap_to_pi(b-a))
    stats = nrmse_statistics([np.pi-.1], [-np.pi+.1], [np.pi], 1000, 42, angular=True)
    assert stats["point"][0] == pytest.approx(.2/np.pi)


def test_zero_norm_angular_prediction():
    record = {"early": [1, 0, 0], "late": [0, 1, 0], "metric": "next-state",
              "components": {"state_value": {"ok": True, "value": [0, 0, 2]}}}
    assert angular_pair(record, "state_value")[2] == "zero_norm_reconstructed_state"


def test_nested_shared_stable_query_subsets():
    records = [{"task": task, "condition_id": task+metric, "query_id": task+metric+str(i),
                "episode_path": task+".npz", "query_index": i, "ordinal": i}
               for task in TASKS for metric in ("a", "b") for i in range(30)]
    subsets = select_subsets(records, 42)
    assert subsets == select_subsets(list(reversed(records)), 42)
    for task in TASKS:
        a, b = subsets["conditions"][task+"a"], subsets["conditions"][task+"b"]
        assert a["10"] == a["30"][:10] and a["20"] == a["30"][:20]
        assert [r["query_index"] for r in a["10"]] == [r["query_index"] for r in b["10"]]
