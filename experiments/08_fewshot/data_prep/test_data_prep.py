"""Offline regression checks for example identity, truth, and exported sampling."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('prepare_prompt',HERE/'prepare_prompt.py')
prompt=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompt)


@pytest.fixture(scope='module')
def examples():
    return json.loads((HERE/'examples.json').read_text())


def test_example_coverage_and_input_boundaries(examples):
    assert len(examples)==len({e['query_id'] for e in examples})==1920
    assert sum(not e['response_available'] for e in examples)==6
    assert sum(not e['reasoning_available'] for e in examples)==126
    sources={}
    for e in examples:
        assert 'system_prompt' not in e
        path=ROOT/e['source_records']
        if path not in sources:
            sources[path]=[json.loads(line) for line in path.read_text().splitlines()]
        raw=sources[path][e['source_line']-1]
        assert e['query_id']==raw['query_id']
        assert e['question']==raw['user_prompt']
        assert e['answer']==raw.get('assistant_text')
        assert e['reasoning_score'] in (1,2,3,4)


def test_state_truth_uses_target_and_forward_time(examples):
    import numpy as np
    episodes={}
    for e in examples:
        if not e['metric'].endswith('state'):
            continue
        path=ROOT/e['episode_path']
        if path not in episodes:
            with np.load(path,allow_pickle=False) as data:
                episodes[path]=data['states'].copy()
        states=episodes[path]
        early=states[e['query_index']].reshape(-1).astype(float)
        late=states[e['query_index']+1].reshape(-1).astype(float)
        assert e['ground_truth']['state_value']==(late if e['metric']=='next-state' else early).tolist()
        assert e['ground_truth']['state_delta']==(late-early).tolist()


def test_errors_and_parse_failures(examples):
    for e in examples:
        for c in e['comparison'].values():
            if not c['parsed']:
                assert c['prediction'] is None and c['exact_match'] is None and c['absolute_error'] is None
            elif c['element_matches'] is not None:
                assert c['element_matches']==[p==t for p,t in zip(c['prediction'],c['ground_truth'])]
                assert c['exact_match']==all(c['element_matches'])
            if c['absolute_error'] is not None:
                assert c['absolute_error']==[abs(p-t) for p,t in zip(c['prediction'],c['ground_truth'])]
            if c['label'] in ('トルク','状態値','状態変化量（後 − 前）'):
                assert c['exact_match'] is None


def test_selection_and_prompt(examples):
    pool=[{**e,'example_type':'correct' if i%2==0 else 'incorrect'} for i,e in enumerate(examples) if e['response_available']]
    options=dict(task='Pendulum-v1',metric='next-action',scores=[1,2],correct=3,incorrect=2,sampling='random',seed=42,order='alternate')
    chosen=prompt.select(pool,**options)
    assert [e['example_type'] for e in chosen]==['correct','incorrect','correct','incorrect','correct']
    assert chosen==prompt.select(pool,**options)
    assert all(e['task']=='Pendulum-v1' and e['metric']=='next-action' and e['reasoning_score'] in (1,2) for e in chosen)
    text=prompt.prompt_text(chosen)
    assert text.count('\nQuestion:\n')==5
    for e in chosen:
        assert e['question'] in text and e['answer'] in text
    assert 'system_prompt' not in text
    with pytest.raises(ValueError,match='available'):
        prompt.select(pool,task='Pendulum-v1',correct=99999)
    with pytest.raises(ValueError,match='unlabelled'):
        prompt.select(examples,correct=1)
