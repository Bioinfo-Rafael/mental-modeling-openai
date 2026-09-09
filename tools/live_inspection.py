"""Presentation of explicitly supplied live scans; no disk cache is read here."""
import json
from html import escape

import pandas as pd
import numpy as np
from numbers import Real

from dataset_adapters.base import inventory_context
from dataset_adapters.registry import dataset_ids, get_adapter, EXTERNAL


def official_frames(inventory):
    tasks, files, arrays = [], [], []
    for task in inventory['tasks']:
        tasks.append(dict(task=task['task'], number_of_episodes=task['number_of_episodes'],
                          total_timesteps=task['total_timesteps'],
                          **{'timesteps_'+k: v for k, v in task['timesteps_per_episode'].items()},
                          state_dimensions=task['states']['dimensions'],
                          action_dimensions=task['actions']['dimensions'],
                          action_kind=task['actions']['kinds'], npz_keys=task['npz_keys'],
                          schema_consistent_except_T=task['schema_consistent_except_T']))
        for eid, f in enumerate(sorted(task['files'], key=lambda f: f['relative_path'])):
            identity = dict(task=task['task'], episode_id=eid, filename=f['filename'], relative_path=f['relative_path'])
            files.append(dict(**identity, timesteps=f['timesteps'], size_bytes=f['raw_size_bytes'], schema_group=f['schema_group']))
            for key, spec in f['arrays'].items():
                arrays.append(dict(**identity, key=key, **{k: spec[k] for k in ['shape', 'dtype', 'ndim', 'num_elements']}))
    return pd.DataFrame(tasks), pd.DataFrame(files), pd.DataFrame(arrays)


def external_frames(inventory):
    datasets = pd.DataFrame([dict(dataset=k, **v) for k, v in inventory['datasets'].items()])
    files = pd.DataFrame(inventory['files']).reindex(columns=[
        'dataset', 'relative_path', 'format', 'size_bytes', 'rows', 'columns', 'schema_group', 'status'])
    groups, columns, arrays = [], [], []
    for gid, g in inventory['groups'].items():
        groups.append(dict(dataset=g['dataset'], schema_group=gid, file_count=g['file_count'],
                           rows=g['rows'], schema_kind=g['schema']['kind'], example_file=g['example_file']))
        for f in g['schema'].get('fields', []):
            columns.append(dict(dataset=g['dataset'], schema_group=gid, column=f['name'], dtype=f.get('dtype'),
                                rows=g['rows'], missing=g.get('missing', {}).get(f['name']),
                                schema_kind=g['schema']['kind']))
    for f in inventory['files']:
        schema = inventory['groups'].get(f['schema_group'], {}).get('schema', {})
        variables = f.get('variables', schema.get('variables', {}))
        for key, spec in variables.items():
            shape = spec.get('shape')
            arrays.append(dict(dataset=f['dataset'], relative_path=f['relative_path'], format=f['format'],
                               key=key, shape=shape, dtype=spec.get('dtype'),
                               ndim=spec.get('ndim', len(shape) if shape is not None else None),
                               opaque=spec.get('opaque', False)))
        for stream in f.get('streams', []):
            for key, spec in stream['arrays'].items():
                arrays.append(dict(dataset=f['dataset'], relative_path=f['relative_path'], format=f['format'],
                                   key=f"{stream['name']}/{stream['multi_id']}/{key}", shape=spec['shape'],
                                   dtype=spec['dtype'], ndim=spec['ndim'], opaque=False))
    return datasets, files, pd.DataFrame(groups), pd.DataFrame(columns), pd.DataFrame(arrays)


def unified_overview(llmx_inventory, candidate_inventory):
    """Counts/reader units from live scans; capabilities remain explicit reader metadata."""
    rows = []
    with inventory_context(llmx_inventory, candidate_inventory):
        for name in dataset_ids():
            d = get_adapter(name).describe()
            rows.append(dict(dataset=name, category=d['category'], files=d['number_of_files'],
                             sequences=d['number_of_sequences'], records=d['number_of_records'],
                             format=d['format'], sequential=d['sequential'],
                             state_available=d['state_available'], action_available=d['action_available'],
                             reward_available=d['reward_available'],
                             status=d.get('status', 'acquired'),
                             record_scope='episode timesteps' if name not in EXTERNAL else
                             'reader units; duplicate exports/topics may overlap'))
    return pd.DataFrame(rows)


def compare_frames(live, saved, keys, values, *, float_columns=()):
    """Full-key outer comparison; missing rows are not silently discarded or zero-filled."""
    left, right = live[keys+values].copy(), saved[keys+values].copy()
    def canonical(value):
        if isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value):
            if float(value).is_integer():
                value = int(value)
        return json.dumps(value, sort_keys=True, default=str)
    for value in values:
        for frame in [left, right]:
            frame[value] = frame[value].map(canonical)
    result = left.merge(right, on=keys, how='outer', suffixes=('_live', '_saved'), indicator=True, validate='one_to_one')
    result['matches'] = result['_merge'].eq('both')
    for value in values:
        same = result[value+'_live'].eq(result[value+'_saved'])
        if value in float_columns:
            a = pd.to_numeric(result[value+'_live'], errors='coerce')
            b = pd.to_numeric(result[value+'_saved'], errors='coerce')
            same |= np.isfinite(a) & np.isfinite(b) & np.isclose(a, b, rtol=1e-12, atol=1e-9)
        result['matches'] &= same
    return result.rename(columns={'_merge': 'presence'})


def show_frame(frame, *, title='', start=0, page_size=None, height=440):
    """Scrollable HTML, all official rows or an explicitly labelled external page."""
    from IPython.display import HTML, display
    view = frame if page_size is None else frame.iloc[start:start+page_size]
    label = f'{title}: {len(frame):,} total rows'
    if page_size is not None:
        label += f'; displaying [{start}:{min(start+page_size, len(frame))})'
    display(HTML(f'<p>{escape(label)}</p><div style="max-height:{height}px;overflow:auto">'
                 + view.to_html(index=True, max_rows=None, max_cols=None, escape=True) + '</div>'))


def token_summary_frame(result):
    return pd.DataFrame(result.summary).rename(columns={
        'n_units': 'counted_queries', 'mean_input_tokens': 'mean', 'std_input_tokens': 'std',
        'min_input_tokens': 'min', 'median_input_tokens': 'median', 'p95_input_tokens': 'p95',
        'max_input_tokens': 'max', 'total_input_tokens': 'total'})


def compare_saved_tokens(live_result, saved_summary, saved_metadata):
    """Compare only like-for-like saved configuration, model, sample policy and metric.

    Old batch summaries omit metric/question/seed on aggregate rows: use metadata
    scopes and label what cannot be checked instead of declaring a strict match.
    """
    live = pd.DataFrame(live_result.summary)
    keys = ['dataset_id', 'prompt_mode', 'history_size', 'preprocessing_config_sha256']
    values = ['n_units', 'eligible_query_pool', 'sample_size', 'seed', 'count_scope',
              'mean_input_tokens', 'std_input_tokens', 'min_input_tokens', 'median_input_tokens',
              'p95_input_tokens', 'max_input_tokens', 'total_input_tokens']
    values = [k for k in values if k in live and k in saved_summary]
    result = compare_frames(live, saved_summary, keys, values,
                            float_columns=['mean_input_tokens', 'std_input_tokens'])
    same_model = (live_result.metadata['model'] == saved_metadata.get('model') and
                  live_result.metadata['tokenizer'] == saved_metadata.get('tokenizer'))
    result['model_tokenizer_match'] = same_model
    result['matches'] &= same_model
    result['comparison_note'] = 'counts exact; mean/std rtol=1e-12 atol=1e-9 for CSV roundtrip; prompt hashes/metric require query CSV comparison'
    return result
