"""Check source masking and prompt-visibility annotations."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plot_question_timelines as timelines


def test_window_missing_invalid_dead_and_unrecorded_values():
    base=dict(alive=1,state_valid=1,action_status='recorded',
              **{f:float(i) for i,f in enumerate(timelines.STATE_FEATURES)},
              **{f'{f}_id':1 for f in timelines.ACTIONS})
    records={(0,'A'):base,(1,'A'):dict(base,state_valid=0),
             (2,'A'):dict(base,action_status='missing'),(3,'A'):dict(base,alive=0)}
    result=timelines.recorded_window(records,'A',1,before=2,after=2)
    assert [r['relative_step'] for r in result]==[-2,-1,0,1,2]
    assert result[0]['state']==[None]*10
    assert result[1]['state']==list(range(10))
    assert result[2]['state']==[None]*10 and result[2]['action']==[1.]*4
    assert result[3]['state']==list(range(10)) and result[3]['action']==[None]*4
    assert result[4]['state']==[None]*10 and result[4]['action']==[None]*4


def test_angle_wrap_breaks_line_without_mutating_source():
    values=[358.,359.,1.,2.,None,3.]
    result=timelines.line_values(values,circular=True)
    assert values==[358.,359.,1.,2.,None,3.]
    assert np.isnan(result[2]) and np.isnan(result[4])
    assert result[3]==2.
    assert timelines.line_values(values)[2]==1.


def test_context_steps_only_include_target_values():
    visible=dict(target_id='A',query_context=[
        dict(step=10,actions=[dict(object_id='A')],states=[dict(object_id='B')]),
        dict(step=11,states=[dict(object_id='A'),dict(object_id='B')])])
    assert timelines.visible_steps(visible)==dict(state=[11],action=[10])
