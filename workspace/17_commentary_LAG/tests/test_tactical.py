import json
from pathlib import Path
import sys
import pytest
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'13_LAG')]
from commentary_experiment import runner, evaluation


def test_tactical_windows_and_prompts():
    plan = runner.make_plan(variant='tactical')
    baseline = json.loads((HERE/'results/commentary_batch_v1/manifest.json').read_text())
    assert plan['counts']['requests'] == 15
    assert plan['prompt_version'] == 'lag-tactical-v1'
    for sid, sample in plan['samples'].items():
        assert sample['window'] == baseline['samples'][sid]['window']
        system, user = [m['content'] for m in sample['messages']]
        assert system == (HERE/'commentary_experiment/tactical_system.txt').read_text()
        assert 'ENTIRE 61-step window' in user
        assert '[INSERT' not in user
        data = user.split('BEGIN WINDOW DATA')[1].split('END WINDOW DATA')[0].strip()
        assert [json.loads(line) for line in data.splitlines()] == sample['window']['blocks']
        if sample['target_step'] == 85:
            assert 'Window: steps 55 through 115, inclusive' in user
            assert 'Center time: 17.0 seconds' in user
    assert plan['request_options'] == baseline['request_options']
    assert [q['model'] for q in plan['queries']] == [q['model'] for q in baseline['queries']]


def raw(text, finish='stop'):
    return {'choices':[{'message':{'content':text},'finish_reason':finish}]}


def test_json_contract():
    text = json.dumps({a:'Narrative.' for a in ('A0100','A0200','B0100','B0200')})
    assert evaluation.parse_intentions(raw(text))['valid']
    for bad in ('```json\n'+text+'\n```', '{}', text[:-1]+',"extra":"x"}', text.replace('"Narrative."','[]',1), text[:-1]+',"A0100":"duplicate"}'):
        assert not evaluation.parse_intentions(raw(bad))['valid']
    assert not evaluation.parse_intentions(raw(text, 'length'))['valid']
