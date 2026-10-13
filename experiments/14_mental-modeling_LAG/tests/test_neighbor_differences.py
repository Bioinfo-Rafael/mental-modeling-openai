from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plot_neighbor_differences as neighbor
import plot_results as plots


def test_target_relative_anchor():
    assert neighbor.anchor_step('next-action',10)==9
    assert neighbor.anchor_step('next-state',10)==9
    assert neighbor.anchor_step('last-action',10)==11
    assert neighbor.anchor_step('last-state',10)==11


def test_copy_truth_opposite_and_zero_denominator():
    actual=[1.]*10;base=[0.]*10;scale=[2.]*10
    assert neighbor.state_ratio(base,actual,base,scale)['ratio']==0
    assert neighbor.state_ratio(actual,actual,base,scale)['ratio']==1
    assert neighbor.state_ratio([-1.]*10,actual,base,scale)['ratio']==1
    for pred in [actual,base]:
        r=neighbor.state_ratio(pred,base,base,scale)
        assert r['ratio'] is None and r['zero_actual_change']


def test_normalize_before_norm_and_wrap_angles():
    truth=[0.]*10;truth[0]=2;truth[5]=1
    base=[0.]*10;base[5]=359
    pred=[0.]*10;pred[0]=1;pred[5]=0
    r=neighbor.state_ratio(pred,truth,base,[2.]+[1.]*9)
    assert r['actual_change']==pytest.approx(np.sqrt(5))
    assert r['predicted_change']==pytest.approx(np.sqrt(1.25))
    assert r['ratio']==pytest.approx(.5)
    with pytest.raises(ValueError):neighbor.state_ratio(pred,truth,base,[0.]*10)


def test_mean_of_ratios_and_matched_scope_difference():
    metric='state_change_ratio_range'
    def row(q,v):return dict(comparison_id=q,truth=[0]*10,neighbor_metrics={metric:v})
    left=[row('a',0.),row('b',10.)]
    right=[row('a',1.),row('b',None)]
    assert plots.estimate(left,metric,None,{})==5
    assert plots.paired_estimates(left,right,metric,None,{})==(1.,1,0.,1.)
    assert not plots.usable(right[1],metric)
