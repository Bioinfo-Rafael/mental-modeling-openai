"""Numerical contracts for temporal SUMs and matched-scope comparisons."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('trajectory_comparison_plots', HERE/'plot_results.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def trajectory(state=0., action=(0, 0, 0)):
    return [dict(step=t, state=[state]*10, action=list(action)) for t in range(20, 30)]


def test_ten_points_are_summed_before_window_average():
    truth, prediction = trajectory(), trajectory(2., (1, 2, 0))
    scales = dict(range=[2.]*10, sd=[1.]*10)
    m = plots.window_metrics(prediction, truth, scales)
    assert m['state_sae.alt_m'] == 20  # Not 2 (temporal MAE).
    assert m['state_nsae.alt_m'] == 10
    assert m['state_sum_nmae'] == m['state_sum_nrmse'] == 10
    assert m['state_sum_zmae'] == m['state_sum_zrmse'] == 20
    assert m['action_wrong_steps'] == 10
    assert m['action_sum_bin_error'] == 10  # Mean of 3 ID errors, then sum 10 points.
    assert m['action_sum_bin_error.heading_action'] == 20
    result = plots.estimate([dict(comparison_id='a', metrics=m), dict(comparison_id='b', metrics={'state_sae.alt_m':0})], 'state_sae.alt_m')
    assert result['value'] == 10 and result['n_valid'] == 2


def test_absolute_errors_do_not_cancel_and_angles_wrap():
    truth, pred = trajectory(), trajectory()
    for i in range(10):
        pred[i]['state'][2] = 3 if i % 2 else -3
        truth[i]['state'][5], pred[i]['state'][5] = 359, 1
    m = plots.window_metrics(pred, truth, dict(range=[1.]*10, sd=[1.]*10))
    assert m['state_sae.alt_m'] == 30 and m['state_sae.yaw_deg'] == 20


def test_rmse_is_per_point_then_summed_not_root_after_time_sum():
    truth, pred = trajectory(), trajectory()
    pred[0]['state'][0], pred[1]['state'][0] = 3, 4
    m = plots.window_metrics(pred, truth, dict(range=[1.]*10, sd=[1.]*10))
    assert m['state_sum_nrmse'] == pytest.approx(7/np.sqrt(10))
    assert m['state_sum_nrmse'] != pytest.approx(5/np.sqrt(10))
    assert m['state_sum_nmae'] == pytest.approx(.7)


def test_scope_differences_use_only_common_valid_windows():
    def row(key, value):
        return dict(comparison_id=key, truth=trajectory(), metrics={'error':value})
    left, right = [row('a', 2), row('b', 100)], [row('b', None), row('a', 5)]
    m = plots.paired_estimate(left, right, 'error')
    assert (m['value'], m['n_valid'], m['paired_self'], m['paired_all']) == (3, 1, 2, 5)
    assert m['comparison_ids'] == ['a']
    with pytest.raises(ValueError, match='identical'):
        plots.paired_estimate(left, right[:1], 'error')
    right[1]['truth'][0]['state'][0] = 1
    with pytest.raises(ValueError, match='truth differs'):
        plots.paired_estimate(left, right, 'error')


def test_reference_is_unique_truth_with_short_arc():
    a = dict(episode_id='e', target_id='A', ground_truth=trajectory())
    for i, row in enumerate(a['ground_truth']):
        row['state'] = [float(i%2)*2]*10
        row['state'][5] = 359 if i%2 == 0 else 1
    scales = plots.reference_scales({'self':a,'all':deepcopy(a)})
    assert scales['n_unique_target_states'] == 10
    np.testing.assert_allclose(scales['range'], [2]*10)
    np.testing.assert_allclose(scales['sd'], [1]*10)
    scales['range'][0] = 0
    m = plots.window_metrics(trajectory(2), trajectory(), scales)
    assert m['state_nsae.lon_deg'] is None and m['state_sum_nmae'] is None
    assert m['state_nsae.alt_m'] == 10


def test_failed_prediction_excluded_but_tokens_retained():
    sample = dict(horizon=10, task='trajectory', ground_truth=trajectory())
    data = dict(samples={'s':sample}, rows=[dict(sample_id='s',H=20,scoring={'parse_success':False,'prediction':trajectory(99)},
                                              raw_response={'usage':{'prompt_tokens':100}})])
    rows = plots.prepare(data, dict(range=[1.]*10, sd=[1.]*10))
    assert rows[0]['prediction'] is None
    assert 'state_sae.alt_m' not in rows[0]['metrics']
    assert rows[0]['metrics']['token_prompt_tokens'] == 100
    assert rows[0]['metrics']['token_reasoning'] is None
