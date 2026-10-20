import json
from pathlib import Path
import sys
import pytest
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'13_LAG')]
from commentary_experiment import runner, execution, evaluation, batch
from commentary_experiment.dataset import build_window
from commentary_experiment.config import CENTERS, EPISODE, REFERENCE_URL
from lag_data.dataset import load_episode, PROCESSED

@pytest.fixture(scope='module')
def frame():
    return load_episode(PROCESSED/f'{EPISODE}.csv')

@pytest.mark.parametrize('center', CENTERS)
def test_exact_input(frame, center):
    window = build_window(frame, center)
    assert [b['step'] for b in window['blocks']] == list(range(center-30, center+31))
    for block in window['blocks']:
        source = frame[frame.step.eq(block['step']) & frame.entity_type.isin(('aircraft','missile'))]
        assert {e['object_id'] for e in block['entities']} == set(source.object_id)
        for obj in block['entities']:
            row = source[source.object_id.eq(obj['object_id'])].iloc[0]
            assert 'shoot_action' not in json.dumps(obj)
            if obj['state_valid'] == 0:
                assert obj['state'] == [None]*10
            if obj['entity_type'] == 'aircraft':
                assert len(obj['action']) == 3
                assert obj['missiles_remaining'] == (None if __import__('pandas').isna(row.missiles_remaining) else row.missiles_remaining)
                if obj['team'] == 'B':
                    assert obj['action'] == [None]*3
            else:
                assert 'action' not in obj
                assert obj['state'][6:] == [None]*4


def test_boundary_and_missing_row(frame):
    for center in (29, 311):
        with pytest.raises(ValueError):
            build_window(frame, center)
    with pytest.raises(ValueError):
        build_window(frame[~(frame.step.eq(45) & frame.object_id.eq('A0100'))], 45)


def test_plan_and_exports(tmp_path):
    plan = runner.make_plan()
    assert plan['counts']['requests'] == 15
    for sample in plan['samples'].values():
        content = json.dumps(sample['messages'])
        assert 'aircraft combat simulation game' in content
        assert 'shoot_action' not in content
        assert 'missiles_remaining' in content
        assert EPISODE not in content and REFERENCE_URL not in content
        assert '{window_data}' not in content
    execution.save_plan(plan, tmp_path)
    batch.prepare(plan, tmp_path)
    evaluation.evaluate_saved(plan, tmp_path)
    evaluation_path = tmp_path/'evaluation/rubric_template.json'
    evaluation_path.write_text('[{"reviewer":"human"}]')
    evaluation.evaluate_saved(plan, tmp_path)
    assert json.loads(evaluation_path.read_text()) == [{'reviewer':'human'}]
    with pytest.raises(FileExistsError):
        execution.save_plan(runner.make_plan(models=('sol',)), tmp_path)
