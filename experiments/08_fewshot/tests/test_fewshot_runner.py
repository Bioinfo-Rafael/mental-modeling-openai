"""Dry-run regression tests. Test selections are confined to pytest's temporary directory."""
from dataclasses import replace
from importlib import import_module
import json
from pathlib import Path
import pytest

entry=import_module('experiments.08_fewshot.run')
r=entry.runner


@pytest.fixture(scope='module')
def catalog():
    return r.read_json(r.HERE/'data_prep/examples.json')


@pytest.fixture(scope='module')
def examples(catalog):
    # Rebuild the authorized ID-based candidate pool in memory; never overwrite user files.
    selector=import_module('experiments.08_fewshot.data_prep.select_by_id')
    selected,_,_=selector.select(catalog)
    assert len(selected)==74
    return selected


@pytest.fixture
def exported(tmp_path,examples):
    path=tmp_path/'selected_examples_test.json'
    r.write_json(path,dict(schema_version=1,kind='candidate_pool',examples=examples))
    return path


@pytest.fixture(scope='module')
def plan(examples):
    return r.make_plan(entry.EXPERIMENT,examples,{})


def test_default_grid_and_paired_targets(plan,examples):
    assert len(r.conditions(entry.EXPERIMENT))==48
    assert len(plan)==len({q['query_id'] for q in plan})==144
    assert {q['metric'] for q in plan}=={'next-action'}
    assert {q['model_alias'] for q in plan}=={'terra','luna'}
    assert {q['H'] for q in plan}=={5,20}
    for task in entry.EXPERIMENT.tasks:
        keys={(q['episode_path'],q['query_index']) for q in plan if q['task']==task}
        assert len(keys)==3
    example_episodes={e['episode_path'] for e in examples}
    assert all(q['episode_path'] not in example_episodes for q in plan)
    assert all(q['history_end_exclusive']-q['history_start']==q['H'] for q in plan)


def test_nested_shots_roles_and_cross_axes(plan,examples):
    lookup={e['query_id']:e for e in examples}
    for q in plan:
        chosen=[lookup[key] for key in q['example_ids']]
        assert len(chosen)==q['shots']
        assert all(e['task']==q['task'] for e in chosen)
        assert [e['example_type'] for e in chosen]==['correct','incorrect']*(q['shots']//2)
        assert all(e['reasoning_score'] in q[e['example_type']+'_scores'] for e in chosen)
    assert any(lookup[e]['metric']!='next-action' for q in plan for e in q['example_ids'])
    for task in entry.EXPERIMENT.tasks:
        for pattern in entry.EXPERIMENT.score_patterns:
            pools={n:{tuple(q['example_ids']) for q in plan if q['task']==task and q['score_pattern']==pattern.name and q['shots']==n} for n in (4,8,12)}
            assert all(len(p)==1 for p in pools.values())
            assert next(iter(pools[12]))[:8]==next(iter(pools[8]))
            assert next(iter(pools[8]))[:4]==next(iter(pools[4]))


def test_prompt_exact_request_and_boundary(plan):
    for q in plan:
        assert q['user_prompt'][q['target_offset']:]==q['target_user_prompt']
        assert q['target_user_prompt'].startswith(r.MARKER)
        assert q['user_prompt'].count('\nQuestion:\n')==q['shots']
        assert q['user_prompt'].count('\nAnswer:\n')==q['shots']
        assert q['user_prompt'].count('\nLabel:\n')==q['shots']
        assert q['system_prompt'] not in q['user_prompt']
        assert q['request']==r.common.expected_api_request(q['model'],q['system_prompt'],q['user_prompt'])
        assert q['request']['reasoning_effort']=='medium'
        assert 'temperature' not in q['request']
        assert q['input_content_tokens']>0
        assert 'ground_truth' not in q['target_user_prompt']


def test_input_validation_duplicates_and_discovery(exported,tmp_path,examples):
    result,hashes,duplicates=r.load_candidates([exported],r.HERE/'data_prep/examples.json')
    assert len(result)==74 and duplicates==0 and str(exported) in hashes
    assert r.discover_inputs((),tmp_path)==[exported]
    second=tmp_path/'another.json';second.write_bytes(exported.read_bytes())
    with pytest.raises(ValueError,match='複数'):
        r.discover_inputs((),tmp_path)
    result,_,duplicates=r.load_candidates([exported,second],r.HERE/'data_prep/examples.json')
    assert len(result)==74 and duplicates==74
    data=r.read_json(second);data['examples'][0]['answer']='MODIFIED'
    r.write_json(second,data)
    with pytest.raises(ValueError,match='不一致'):
        r.load_candidates([second],r.HERE/'data_prep/examples.json')
    with pytest.raises(ValueError,match='不足'):
        r.choose_demonstrations(entry.EXPERIMENT,examples[:2])


def test_run_missing_and_ready_without_clients(tmp_path,monkeypatch,exported):
    import openai
    def forbidden(*a,**kw):
        raise AssertionError('An API client must not be constructed during dry-run')
    monkeypatch.setattr(openai,'OpenAI',forbidden)
    monkeypatch.setattr(r.common,'OpenAIChatBackend',forbidden)
    monkeypatch.setattr(r.common,'output_path',lambda p:tmp_path/Path(p).name)
    original_discovery=r.discover_inputs
    monkeypatch.setattr(r,'discover_inputs',lambda explicit,data_dir:[])
    assert r.run(entry.EXPERIMENT,[])==2
    pending=next(tmp_path.glob('*/manifest.json'))
    assert r.read_json(pending)['status']=='input_required'
    assert r.read_json(pending)['queries']==[]
    monkeypatch.setattr(r,'discover_inputs',original_discovery)
    assert r.run(entry.EXPERIMENT,['--input',str(exported)])==0
    manifests=[r.read_json(p) for p in tmp_path.glob('*/manifest.json')]
    ready=next(m for m in manifests if m['status']=='ready')
    assert ready['api_requests_made']==0 and ready['generated_queries']==144
    output=next(p.parent for p in tmp_path.glob('*/manifest.json') if r.read_json(p)['status']=='ready')
    assert len(list((output/'prompts').rglob('*.txt')))==144
    template_bytes=(r.HERE/'results/dry_run/preview_template.html').read_bytes()
    assert (output/'preview_template.html').read_bytes()==template_bytes
    first=ready['queries'][0]
    text=(output/first['prompt_txt']).read_text()
    assert first['user_prompt'] in text and first['system_prompt'] in text
    assert '改行' in (output/'preview.html').read_text() or 'pre-wrap' in (output/'preview.html').read_text()
    incomplete=tmp_path/'selected_incomplete.json'
    payload=r.read_json(exported)
    payload['examples']=payload['examples'][:2]
    r.write_json(incomplete,payload)
    assert r.run(entry.EXPERIMENT,['--input',str(incomplete)])==2
    manifests=[r.read_json(p) for p in tmp_path.glob('*/manifest.json')]
    shortage=next(m for m in manifests if m['status']=='insufficient_candidates')
    assert shortage['queries']==[] and any(row['missing'] for row in shortage['candidate_inventory'])
    assert (output/'preview_template.html').read_bytes()==template_bytes


def test_execution_flags_and_counts():
    with pytest.raises(SystemExit):
        r.run(entry.EXPERIMENT,['--execute'])
    with pytest.raises(ValueError):
        r.validate_spec(replace(entry.EXPERIMENT,shot_counts=(3,)))
    with pytest.raises(ValueError):
        r.validate_spec(replace(entry.EXPERIMENT,metric='last-action'))


def test_default_score_ranges():
    p1,p2=entry.EXPERIMENT.score_patterns
    assert (p1.correct_scores,p1.incorrect_scores)==((1,),(3,4))
    assert (p2.correct_scores,p2.incorrect_scores)==((1,2),(1,2,3))


@pytest.mark.parametrize('pattern_index,kind,allowed',[
    (0,'correct',{1}),(0,'incorrect',{3,4}),
    (1,'correct',{1,2}),(1,'incorrect',{1,2,3}),
])
@pytest.mark.parametrize('score',[1,2,3,4])
def test_score_range_boundaries(pattern_index,kind,allowed,score):
    pattern=entry.EXPERIMENT.score_patterns[pattern_index]
    spec=replace(entry.EXPERIMENT,tasks=('Pendulum-v1',),shot_counts=(2,),score_patterns=(pattern,),selection_policy='seeded')
    opposite='incorrect' if kind=='correct' else 'correct'
    examples=[dict(query_id='under-test',task='Pendulum-v1',example_type=kind,reasoning_score=score),
              dict(query_id='other-role',task='Pendulum-v1',example_type=opposite,
                   reasoning_score=getattr(pattern,opposite+'_scores')[0])]
    row=next(row for row in r.candidate_inventory(spec,examples) if row['example_type']==kind)
    assert row['available']==int(score in allowed)
    assert row['missing']==int(score not in allowed)
    if score in allowed:
        assert len(r.choose_demonstrations(spec,examples)[('Pendulum-v1',pattern.name,2)])==2
    else:
        with pytest.raises(ValueError,match='不足'):
            r.choose_demonstrations(spec,examples)


def test_pattern_overlap_is_allowed():
    spec=replace(entry.EXPERIMENT,tasks=('Pendulum-v1',),selection_policy='seeded')
    examples=[dict(query_id=f'{kind}-{i}',task='Pendulum-v1',example_type=kind,reasoning_score=score)
              for kind,score in [('correct',1),('incorrect',3)] for i in range(6)]
    assert not any(row['missing'] for row in r.candidate_inventory(spec,examples))
    bundles=r.choose_demonstrations(spec,examples)
    assert bundles[('Pendulum-v1','pattern1',12)]==bundles[('Pendulum-v1','pattern2',12)]


def test_balanced_model_quotas_and_minimum_last(examples):
    from collections import Counter
    from itertools import combinations, product
    spec=entry.EXPERIMENT
    for task in spec.tasks:
        for pattern in spec.score_patterns:
            for kind in ('correct','incorrect'):
                selected,_=r.role_pool(spec,examples,task,pattern,kind)
                assert len(selected)==6
                quotas={'sol':2,'terra':2,'luna':2,'3.5':0} if pattern.name=='pattern1' and kind=='correct' else {'sol':1,'terra':1,'luna':2,'3.5':2}
                counts=Counter(e['model'] for e in selected)
                assert all(counts[m]==n for m,n in quotas.items())
                needed=4 if pattern.name=='pattern1' and kind=='incorrect' else 2 if pattern.name=='pattern2' and kind=='correct' else None
                pools={m:[e for e in examples if e['task']==task and e['model']==m and e['example_type']==kind and e['reasoning_score'] in getattr(pattern,kind+'_scores')] for m in quotas}
                last_counts=[]
                for groups in product(*(combinations(pools[m],n) for m,n in quotas.items())):
                    rows=[e for g in groups for e in g]
                    if needed is not None and not any(e['reasoning_score']==needed for e in rows):
                        continue
                    last_counts.append(sum(e['metric']=='last-action' for e in rows))
                assert sum(e['metric']=='last-action' for e in selected)==min(last_counts)


def test_api_execution_persistence_resume_and_no_resend(tmp_path, monkeypatch, plan):
    import openai
    from types import SimpleNamespace
    executor = import_module('experiments.08_fewshot.runner.api_execution')
    monkeypatch.setattr(r.common, 'output_path', lambda p: Path(p))
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    rows = [plan[0], next(q for q in plan if q['task']=='Pendulum-v1')]
    manifest = dict(status='ready', queries=rows, input_sha256={})
    (tmp_path/'dry_run').mkdir()
    r.write_json(tmp_path/'dry_run/manifest.json',manifest)
    calls=[]
    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == dict(max_retries=0,timeout=180.0)
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            row=rows[len(calls)-1]
            episode=executor.Episode.load(r.ROOT/row['episode_path'])
            expected=executor.score(row,'',episode)['ground_truth']
            value=expected[0] if isinstance(expected,list) else expected
            text=f'predictions = [2.00]\nFinal action bins: [{value}]' if isinstance(expected,list) else f'Final action choice: {value}'
            raw={'choices':[{'message':{'content':text}}],
                 'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}}
            return SimpleNamespace(model_dump=lambda **kw:raw)
        def close(self):
            pass
    monkeypatch.setattr(openai,'OpenAI',FakeClient)
    assert executor._execute(manifest,tmp_path)==0
    assert calls==[q['request'] for q in rows]
    records=r.common.read_jsonl(tmp_path/'api/records.jsonl')
    assert len(records)==2 and all(row['status']=='match' for row in records)
    assert all(row['total_tokens']==13 for row in records)
    with pytest.raises(FileExistsError):
        executor._execute(manifest,tmp_path)
    # Simulate an interruption after persisting the response but before scoring it.
    (tmp_path/'api/records.jsonl').write_text('')
    assert executor._execute(manifest,tmp_path,resume=True)==0
    assert len(calls)==2 and len(r.common.read_jsonl(tmp_path/'api/records.jsonl'))==2


def test_api_failure_is_not_retried_and_resume_sends_unstarted(tmp_path, monkeypatch, plan):
    import openai
    from types import SimpleNamespace
    executor=import_module('experiments.08_fewshot.runner.api_execution')
    monkeypatch.setattr(r.common,'output_path',lambda p:Path(p))
    monkeypatch.setenv('OPENAI_API_KEY','offline-test-key')
    manifest=dict(status='ready',queries=plan[:2],input_sha256={})
    (tmp_path/'dry_run').mkdir()
    r.write_json(tmp_path/'dry_run/manifest.json',manifest)
    calls=[]
    class FakeClient:
        def __init__(self,**kwargs):
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                raise KeyboardInterrupt()
            return SimpleNamespace(model_dump=lambda **kw:{'choices':[{'message':{'content':'Prediction: 0'}}]})
        def close(self):
            pass
    monkeypatch.setattr(openai,'OpenAI',FakeClient)
    with pytest.raises(KeyboardInterrupt):
        executor._execute(manifest,tmp_path)
    assert r.read_json(tmp_path/'api/summary.json')['unstarted_queries']==1
    assert executor._execute(manifest,tmp_path,resume=True)==2
    assert calls==[q['request'] for q in plan[:2]]
    assert r.read_json(tmp_path/'api/summary.json')['unresolved_queries']==1
    assert executor._execute(manifest,tmp_path,resume=True)==2
    assert len(calls)==2


def test_api_preflight_rejects_changed_preview_and_missing_key(tmp_path, monkeypatch, plan):
    import openai
    executor=import_module('experiments.08_fewshot.runner.api_execution')
    monkeypatch.setattr(r.common,'output_path',lambda p:Path(p))
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    def forbidden(**kw):
        raise AssertionError('No API client may be created')
    monkeypatch.setattr(openai,'OpenAI',forbidden)
    manifest=dict(status='ready',queries=plan[:1],input_sha256={})
    (tmp_path/'dry_run').mkdir()
    r.write_json(tmp_path/'dry_run/manifest.json',{**manifest,'queries':plan[1:2]})
    with pytest.raises(ValueError,match='differ'):
        executor._execute(manifest,tmp_path)
    r.write_json(tmp_path/'dry_run/manifest.json',manifest)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):
        executor._execute(manifest,tmp_path)
    assert not (tmp_path/'api').exists()


@pytest.mark.parametrize('text,status,prediction', [
    ('predictions = [2.00]\nFinal action bins: [9]', 'match', [9]),
    ('predictions = [1.679]\nFinal action bins: [9]', 'match', [9]),
    ('predictions = [2.00]\nFinal action bins: [2]', 'mismatch', [2]),
    ('predictions = [2.00]', 'ignored', None),
    ('predictions = [2.00]\nFinal action bins: [9]\nFinal action bins: [8]', 'ignored', None),
    ('predictions = [2.00]\nFinal action bins: [10]', 'ignored', None),
    ('', 'empty_response', None),
])
def test_joint_bin_scoring_uses_final_bin_not_torque(plan, text, status, prediction):
    scorer=import_module('experiments.08_fewshot.runner.scoring')
    executor=import_module('experiments.08_fewshot.runner.api_execution')
    row=next(q for q in plan if q['task']=='Pendulum-v1')
    episode=executor.Episode.load(r.ROOT/row['episode_path'])
    result=scorer.score(row,text,episode)
    assert result['ground_truth']==[9]
    assert result['status']==status
    assert result['prediction']==prediction
    if text.startswith('predictions = [2.00]'):
        assert result['action_value_prediction']==[2.0]
        assert result['absolute_error']==0


def test_rescore_offline_preserves_response_and_is_idempotent(tmp_path, plan, monkeypatch):
    import openai
    monkeypatch.setattr(r.common, 'output_path', lambda p: Path(p))
    def forbidden(**kwargs):
        raise AssertionError('Offline rescoring must not create an API client')
    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    migration=import_module('experiments.08_fewshot.analysis.rescore_results')
    row=next(q for q in plan if q['task']=='Pendulum-v1')
    text='predictions = [2.00]\nFinal action bins: [9]'
    raw={'choices':[{'message':{'content':text}}]}
    old={**row,'raw_response':raw,'assistant_text':text,'status':'mismatch',
         'prediction':[2],'ground_truth':[9],'index':row['query_index'],'element_accuracy':0.0}
    directory=tmp_path/'api';directory.mkdir()
    r.write_json(directory/'manifest.json',dict(queries=[row]))
    r.write_json(directory/'summary.json',dict(status='complete',matches=0,mismatches=1,ignored=0))
    (directory/'records.jsonl').write_text(json.dumps(old)+'\n')
    (directory/'responses.jsonl').write_text(json.dumps(dict(query_id=row['query_id'],raw_response=raw))+'\n')
    (directory/'requests.jsonl').write_text(json.dumps(dict(query_id=row['query_id']))+'\n')
    (directory/'records.csv').write_text('old csv\n')
    assert migration.rescore(tmp_path)
    new=r.common.read_jsonl(directory/'records.jsonl')[0]
    assert new['status']=='match' and new['prediction']==[9]
    assert new['assistant_text']==text and new['raw_response']==raw
    assert new['legacy_evaluation']['prediction']==[2]
    saved=(directory/'records.jsonl').read_bytes()
    assert not migration.rescore(tmp_path)
    assert (directory/'records.jsonl').read_bytes()==saved
    assert len(list((tmp_path/'scoring_backups').iterdir()))==1
