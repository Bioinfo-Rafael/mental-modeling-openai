import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from dataset_adapters.base import ROOT
from dataset_adapters.registry import dataset_ids,get_adapter
from preprocessing.prompting import load_config,config_identity,eligible_sequences,build_external_query,transform
from preprocessing.llmx_original import Episode,build_prompt_queries,default_metric_question
from tools.output_safety import derived_path

def test_registry_and_all_official_raw():
    assert len(dataset_ids())==25
    for name in dataset_ids()[:20]:
        a=get_adapter(name)
        for s in a.sequences():
            r=a.record(s.sequence_id,0)
            with np.load(ROOT/s.source_file,allow_pickle=False) as z:
                assert set(r['fields'])==set(z.files)
                for key in z.files:
                    value=z[key] if key=='episodic_return' else z[key][0]
                    assert r['fields'][key]['shape']==list(value.shape)
                    assert r['fields'][key]['dtype']==str(value.dtype)

@pytest.mark.parametrize('name',['f16capstone','trajair','aircombat_wez','calculated_moves'])
def test_external_raw_and_processed_separation(name):
    a=get_adapter(name);c=load_config(name);s=next(s for s in eligible_sequences(a,c) if s.length>2)
    before=a.record(s.sequence_id,1)
    assert before['representation']=='raw'
    assert 'state' not in before and 'action' not in before
    h=0 if c['prompt_mode']=='external_static' else 1
    q,processed=build_external_query(a,s,1,c,h)
    assert 'reward' not in processed
    if name!='f16capstone': assert 'action' not in processed
    assert a.record(s.sequence_id,1)==before
    assert q.history_text in q.user_prompt
    assert len(q.user_prompt)>0
    if name=='f16capstone': assert len(before['fields'])==55 and len(processed['state'])<55

def test_feature_selection_and_hash():
    c=load_config('f16capstone'); a=get_adapter('f16capstone');s=a.sequence(0)
    q,x=build_external_query(a,s,10,c,1)
    modified={**c,'state_columns':['m-airspeed-MPS'],'action_columns':[]}
    q2,x2=build_external_query(a,s,10,modified,1)
    assert list(x2['state'])==['m-airspeed-MPS'] and 'action' not in x2
    assert q.user_prompt!=q2.user_prompt
    assert config_identity(c,1)[1]!=config_identity(modified,1)[1]
    assert config_identity(c,1)[1]!=config_identity(c,5)[1]
    with pytest.raises(ValueError): transform({},c)

def test_original_prompt_identical():
    from llm_x.prompting import system_prompt,user_prompt
    from llm_x.questions import resolve_task,resolve_question,render_question
    a=get_adapter('MountainCar-v0');s=a._sources[0];ep=Episode.load(s.path);m,n=default_metric_question(s.task,ep)
    q=next(build_prompt_queries(s,metric=m,question_name=n,history_size=1))
    question=render_question(m,resolve_question(m,n),ep,index=q.query_index,drop_last_feature=False,presented_action=None)
    assert q.system_prompt==system_prompt(resolve_task(s.task))
    assert q.user_prompt==user_prompt(ep,history_start=q.history_start,history_end=q.history_end,indexed_history=True,drop_last_state_feature=False,question=question)

def test_protected_output_and_symlink(tmp_path):
    for path in [ROOT/'data/generated.csv',ROOT/'upstream/derived.json']:
        with pytest.raises(ValueError): derived_path(path)
    (tmp_path/'link').symlink_to(ROOT/'data',target_is_directory=True)
    with pytest.raises(ValueError): derived_path(tmp_path/'link/new.csv')

def test_local_counter_provenance(tmp_path,monkeypatch):
    import socket
    from tools.count_input_tokens import main
    def denied(*a,**k): raise AssertionError('network prohibited in local count')
    monkeypatch.setattr(socket.socket,'connect',denied)
    main(['--dataset','MountainCar-v0','--history-sizes','0','1','--model','gpt-4o','--output-dir',str(tmp_path)])
    rows=list(csv.DictReader((tmp_path/'queries.csv').open()))
    assert len(rows)==1031+1021
    assert {r['history_size'] for r in rows}=={'0','1'}
    assert all(r['preprocessing_config_sha256'] and r['source_file'] for r in rows)
    for r in rows[:10]:
        import hashlib
        assert hashlib.sha256(r['preprocessing_config'].encode()).hexdigest()==r['preprocessing_config_sha256']
        assert int(r['estimated_api_input_tokens'])==int(r['system_tokens'])+int(r['user_tokens'])+9
    assert len(list(csv.DictReader((tmp_path/'token_inventory.csv').open())))==2

def test_static_row_mapping_and_no_history():
    a=get_adapter('aircombat_wez');c=load_config(a.dataset_id)
    r=a.global_row(100);assert r['row_id']==100 and r['index']==100
    with pytest.raises(ValueError):build_external_query(a,a.sequence(0),1,c,1)

def test_f16_mat_and_ulog_readers():
    a=get_adapter('f16capstone')
    for fmt in ('.mat','.ulg'):
        s=next(s for s in a.sequences() if s.format==fmt and s.length)
        r=a.record(s.sequence_id,0)
        assert r['fields'] and r['source_file_sha256']

def test_original_config_cannot_change_semantics(tmp_path):
    import yaml
    c=load_config('MountainCar-v0');c['numeric_precision']=2
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):load_config('MountainCar-v0',path)
