"""Numerical contracts for paired plots and shared normalization."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('lag_plot_results', HERE/'plot_results.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def row(key, prediction, truth):
    return dict(comparison_id=key, prediction=prediction, truth=truth)


def test_bin_mae_averages_components_and_queries():
    rows = [row('a', [1, 3, 2], [0, 2, 1]), row('b', [1, 2, 1], [1, 2, 1])]
    assert plots.estimate(rows, 'action_bin_mae', None, {}) == .5
    assert plots.estimate(rows, 'action_accuracy', None, {}) == 50


def test_paired_difference_recomputes_on_common_valid_windows():
    a = [row('a', [0, 0, 0], [0, 0, 0]), row('b', [0, 0, 0], [1, 1, 1])]
    b = [row('a', [1, 1, 1], [0, 0, 0]), row('b', None, [1, 1, 1])]
    delta, n, left, right = plots.paired_estimates(a, b, 'action_accuracy', None, {})
    assert (delta, n, left, right) == (-100, 1, 100, 0)
    with pytest.raises(ValueError):
        plots.paired_estimates(a, b[:1], 'action_accuracy', None, {})


def test_state_scales_and_shortest_angles():
    a = dict(task='next-state', episode_id='e', target_id='A', target_step=1, ground_truth=[0.]*10)
    b = dict(a, target_step=2, ground_truth=[2.]*10)
    a['ground_truth'][5], b['ground_truth'][5] = 359., 1.
    reference = plots.reference_scales({'a':a, 'b':b, 'duplicate':dict(a,task='last-state')})
    assert reference['n_unique_target_states'] == 2
    np.testing.assert_allclose(reference['range'], [2.]*10)
    np.testing.assert_allclose(reference['sd'], [1.]*10)
    rows = [row('a', b['ground_truth'], a['ground_truth'])]
    assert plots.estimate(rows, 'state_nrmse', None, reference) == 1
    assert plots.estimate(rows, 'state_zrmse', None, reference) == 2
    assert plots.estimate(rows, 'state_mae', 5, reference) == 2


def test_zero_reference_scale_is_undefined():
    rows = [row('a', [1.]*10, [0.]*10)]
    reference = dict(range=[0.]+[1.]*9, sd=[0.]+[1.]*9)
    assert plots.estimate(rows, 'state_nrmse', 0, reference) is None
    assert plots.estimate(rows, 'state_zrmse', None, reference) is None
    assert plots.estimate(rows, 'state_nrmse', 1, reference) == 1


def test_nonlinear_difference_is_difference_of_paired_rmse():
    a = [row('a',[0.]*10,[0.]*10),row('b',[2.]*10,[0.]*10)]
    b = [row('a',[1.]*10,[0.]*10),row('b',[1.]*10,[0.]*10)]
    delta,n,_,_ = plots.paired_estimates(a,b,'state_nrmse',0,dict(range=[1.]*10))
    assert n == 2
    assert delta == pytest.approx(1-np.sqrt(2))


def test_overall_state_weights_normalized_dimensions_before_root():
    rows = [row('a', [10.]+[0.]*9, [0.]*10)]
    reference = dict(range=[2.]*10, sd=[1.]*10)
    assert plots.estimate(rows, 'state_joint_nrmse', None, reference) == pytest.approx(np.sqrt(2.5))
    assert plots.estimate(rows, 'state_joint_zrmse', None, reference) == pytest.approx(np.sqrt(10))
    assert plots.estimate(rows, 'state_joint_nmae', None, reference) == .5
    assert plots.estimate(rows, 'state_joint_zmae', None, reference) == 1.
    assert plots.estimate(rows, 'state_nrmse', None, reference) == .5
    reference['range'][0] = 0
    assert plots.estimate(rows, 'state_joint_nrmse', None, reference) is None
