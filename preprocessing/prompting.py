"""Configurable example preprocessing. Never writes data or invents missing values."""
from pathlib import Path
from hashlib import sha256
import fnmatch
import json
import yaml
from dataset_adapters.base import ROOT, json_safe, file_sha256
from .llmx_original import PromptQuery

def load_config(dataset_id, path=None):
    name = dataset_id + '_default' if dataset_id in ('f16capstone','trajair','aircombat_wez','calculated_moves') else 'llmx_original'
    path = Path(path) if path else ROOT / 'configs/preprocessing' / (name + '.yaml')
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict): raise ValueError('configuration must be a mapping')
    if config.get('dataset_id') not in (None, dataset_id): raise ValueError('configuration dataset mismatch')
    external = dataset_id in ('f16capstone','trajair','aircombat_wez','calculated_moves')
    if external == (config.get('prompt_mode') == 'original_llmx'):
        raise ValueError('official datasets require original_llmx; external datasets require external preprocessing')
    if config.get('prompt_mode') == 'original_llmx':
        expected = dict(name='llmx_original', prompt_mode='original_llmx', implementation='upstream/LLM-Xavier', indexed_history=True, numeric_precision=4, fetch_drop_last_state_feature=True, semantics='upstream Episode/history_text/render_question/system_prompt/user_prompt unchanged')
        if config != expected:
            raise ValueError('original LLM-X semantics/configuration must remain unchanged')
    else:
        allowed = {'name','dataset_id','prompt_mode','source_globs','selector_scope','state_columns','action_columns','reward_column','fields','timestamp_column','system_prompt','question','gap_policy','value_format'}
        if set(config)-allowed: raise ValueError(f'unknown configuration keys: {set(config)-allowed}')
        if config.get('prompt_mode') not in ('external_generic','external_static'): raise ValueError('invalid external prompt mode')
        if config.get('value_format') != 'lexical': raise ValueError('only explicit lexical formatting supported')
        if config.get('gap_policy') not in (None,'preserve'): raise ValueError('only preserve gaps supported; no filling/interpolation')
    return config

def config_identity(config, history_size):
    effective = {**config, 'history_size': history_size}
    text = json.dumps(effective, ensure_ascii=False, sort_keys=True, separators=(',',':'))
    return text, sha256(text.encode()).hexdigest()

def eligible_sequences(adapter, config):
    for s in adapter.sequences():
        if not any(fnmatch.fnmatchcase(s.source_file, g) for g in config['source_globs']): continue
        if config.get('selector_scope') and s.selector.get('scope') != config['selector_scope']: continue
        if config['prompt_mode'] == 'external_generic' and not s.sequential: continue
        yield s

def transform(fields, config):
    def select(columns):
        missing = set(columns)-set(fields)
        if missing: raise ValueError(f'selected columns missing: {sorted(missing)}')
        return {c: json_safe(fields[c]) for c in columns}
    result = {}
    for key in ('state','action'):
        if config.get(key+'_columns'): result[key] = select(config[key+'_columns'])
    if config.get('reward_column'): result['reward'] = select([config['reward_column']])
    if config.get('timestamp_column'): result['timestamp'] = select([config['timestamp_column']])
    if config.get('fields'): result['fields'] = select(list(fields) if config['fields'] == '*' else config['fields'])
    if not result: raise ValueError('no selected input fields')
    return result

def build_external_query(adapter, seq, index, config, history_size):
    static = config['prompt_mode'] == 'external_static'
    if history_size < 0 or (static and history_size): raise ValueError('static preprocessing has history_size=0 only')
    start = index-history_size
    if start < 0 or index >= seq.length: raise IndexError('insufficient history or query index outside sequence')
    steps = []
    current = None
    for i in range(start,index+1):
        fields, _ = adapter.raw_fields(seq,i)
        current = transform(fields,config)
        steps.append(('Record' if static else f'Step {i}') + ':\n' + json.dumps(current,ensure_ascii=False,sort_keys=True,indent=2))
    history = '\n\n'.join(steps)
    question = config['question']
    query = PromptQuery(adapter.dataset_id,adapter.dataset_id,seq.key,ROOT/seq.source_file,file_sha256(ROOT/seq.source_file),'representation','baseline_description',history_size,index,start,index+1,config['system_prompt'],history+'\n\n'+question,history,question)
    return query, current
