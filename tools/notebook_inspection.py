"""Read-only presentation helpers, using the same raw readers and prompt/token code as CLI.

No downloads, output writers, API clients, normalization, or evaluation here.
"""
from pathlib import Path

import pandas as pd

from dataset_adapters.base import ROOT
from dataset_adapters.registry import EXTERNAL
from preprocessing.llmx_original import (
    Episode, REGISTERED_TASKS, build_prompt_queries, default_metric_question,
)
from preprocessing.prompting import (
    build_external_query, config_identity, eligible_sequences, load_config,
)
from tools.count_input_tokens import count_query, resolve_encoding, token_statistics
from tools.inventory_llmx_data import inspect_file
from tools.inventory_candidates import inspect_one
from tools.list_datasets import inventory_rows


def dataset_table():
    table = pd.DataFrame(inventory_rows())
    table['status'] = table['prompt_mode'].map(
        lambda mode: 'NOT_ACQUIRED' if mode == 'unavailable' else 'acquired')
    return table


def select_record(adapter, *, episode=0, sequence_id=None, row_id=None, index=10):
    if not adapter.sequences():
        return None, None
    if row_id is not None:
        if adapter.describe()['sequential']:
            raise ValueError('Use episode/sequence_id and index for sequential datasets.')
        raw = adapter.global_row(row_id)
    else:
        sid = (episode if adapter.dataset_id not in EXTERNAL else sequence_id)
        raw = adapter.record(0 if sid is None else sid, index)
    return adapter.sequence(raw['sequence_id']), raw


def _schema_rows(result):
    """Tabulate existing inspector metadata; no shape/dtype inference is duplicated."""
    rows = []
    arrays = result.get('arrays', result.get('variables', {}))
    for name, spec in arrays.items():
        shape = spec.get('shape')
        rows.append(dict(name=name, shape=shape, dtype=spec.get('dtype'),
                         ndim=spec.get('ndim', len(shape) if shape is not None else None)))
    for field in result.get('schema', {}).get('fields', []):
        n = result.get('logical_records', result.get('rows'))
        rows.append(dict(name=field['name'], shape=[n] if n is not None else None,
                         dtype=field.get('dtype'), ndim=1 if n is not None else None))
    for stream in result.get('streams', []):
        for name, spec in stream['arrays'].items():
            rows.append(dict(name=f"{stream['name']}/{stream['multi_id']}/{name}",
                             shape=spec['shape'], dtype=spec['dtype'], ndim=spec['ndim']))
    return pd.DataFrame(rows, columns=['name', 'shape', 'dtype', 'ndim'])


def inspect_selected_file(adapter, sequence):
    """Re-read the selected file with inventory code; compare to saved inventory."""
    path = ROOT / sequence.source_file
    if adapter.dataset_id not in EXTERNAL:
        actual = inspect_file(path)
        saved = next(f for f in adapter.inventory['files'] if f['relative_path'] == sequence.key)
    else:
        actual = inspect_one(path, adapter.dataset_id)
        saved = next(f for f in adapter.files if adapter.path(f) == sequence.source_file)
        saved = {**saved, 'schema': adapter.inventory['groups'][saved['schema_group']]['schema']}
    current = _schema_rows(actual)
    old = _schema_rows(saved)
    comparison = current.merge(old, on='name', how='outer', suffixes=('', '_inventory'), indicator=True)
    if not comparison.empty:
        comparison['matches_inventory'] = comparison.apply(
            lambda r: r['_merge'] == 'both' and all(
                str(r[k]) == str(r[k+'_inventory']) for k in ('shape', 'dtype', 'ndim')), axis=1)
    measures = []
    for key in ('rows', 'timesteps', 'columns', 'logical_records'):
        if key in actual or key in saved:
            measures.append(dict(measure=key, actual=actual.get(key), inventory=saved.get(key),
                                 matches=actual.get(key) == saved.get(key)))
    return comparison.drop(columns='_merge'), pd.DataFrame(measures), actual


def preview_raw(adapter, sequence, count=5):
    if adapter.dataset_id not in EXTERNAL:
        return {name: value if name == 'episodic_return' else value[:count]
                for name, value in adapter.arrays(sequence.sequence_id).items()}
    return pd.DataFrame([
        {'index': i, **adapter.raw_fields(sequence, i)[0]}
        for i in range(min(count, sequence.length))
    ])


def preprocessing_for_inspection(dataset_id, explicit_path=None):
    if dataset_id in EXTERNAL and explicit_path is None:
        return None, 'Model-input preprocessing: not defined (PREPROCESSING_CONFIG is not selected)'
    if dataset_id not in EXTERNAL and dataset_id not in REGISTERED_TASKS:
        return None, 'Original LLM-X task class/description is unavailable; raw inspection only.'
    path = Path(explicit_path) if explicit_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    return load_config(dataset_id, path), 'Model-input preprocessing: defined'


def queries_for_sequence(adapter, sequence, config, history_size, metric=None, question_name=None):
    if (metric is None) != (question_name is None):
        raise ValueError('Specify both metric and question_name, or set both to None for upstream defaults.')
    if history_size < 0:
        raise ValueError('history_size must be non-negative')
    if config['prompt_mode'] != 'original_llmx':
        if sequence not in list(eligible_sequences(adapter, config)):
            raise ValueError('Selected raw sequence is outside the explicit preprocessing scope.')
        h = 0 if config['prompt_mode'] == 'external_static' else history_size
        for i in range(h, sequence.length):
            yield build_external_query(adapter, sequence, i, config, h)[0]
    else:
        source = adapter._sources[sequence.sequence_id]
        chosen_metric, chosen_question = (metric, question_name) if metric else default_metric_question(
            adapter.dataset_id, Episode.load(source.path))
        yield from build_prompt_queries(source, metric=chosen_metric, question_name=chosen_question,
                                        history_size=history_size, seed=0)


def selected_query(adapter, sequence, index, config, history_size, metric=None, question_name=None):
    if config is None or sequence is None:
        return None
    if config['prompt_mode'] != 'original_llmx':
        if sequence not in list(eligible_sequences(adapter, config)):
            raise ValueError('Selected raw sequence is outside the explicit preprocessing scope.')
        h = 0 if config['prompt_mode'] == 'external_static' else history_size
        return build_external_query(adapter, sequence, index, config, h)[0]
    query = next((q for q in queries_for_sequence(adapter, sequence, config, history_size,
                                                 metric, question_name) if q.query_index == index), None)
    if query is None:
        raise ValueError('No valid query at this index/history. Raw record remains available; choose another QUERY_INDEX.')
    return query


def token_record(query, sequence, config, encoding, fallback_used, model):
    """Same count_query as CLI, with stable IDs and preprocessing provenance."""
    result = count_query(query, encoding, model=model, encoding_name=encoding.name,
                         fallback_used=fallback_used)
    text, digest = config_identity(config, query.history_size)
    result.update(dataset_id=query.task, episode_id=sequence.episode_id,
                  sequence_id=sequence.sequence_id, source_file=sequence.source_file,
                  input_tokens=result['estimated_api_input_tokens'], prompt_mode=config['prompt_mode'],
                  preprocessing_name=config['name'], preprocessing_config=text,
                  preprocessing_config_sha256=digest)
    return result


def dataset_token_table(adapter, config, history_size, model, metric=None, question_name=None,
                        *, external_sequence=None, external_full_dataset=False, max_external_queries=100000):
    """Official: all episodes. External: selected unit by default; explicit full scope opt-in.

    Guard prevents an inspection notebook from starting millions of external queries silently.
    Token values and statistics are delegated to the existing CLI functions.
    """
    if config is None:
        return pd.DataFrame(), 'Token count: N/A'
    external = config['prompt_mode'] != 'original_llmx'
    seqs = list(eligible_sequences(adapter, config)) if external else adapter.sequences()
    if external and not external_full_dataset:
        seqs = [s for s in seqs if s.sequence_id == external_sequence]
    h = 0 if config['prompt_mode'] == 'external_static' else history_size
    if external and sum(max(0, s.length-h) for s in seqs) > max_external_queries:
        return pd.DataFrame(), f'Not executed: external scope exceeds {max_external_queries:,} queries. Select a smaller unit or explicitly raise MAX_EXTERNAL_QUERIES.'
    if not seqs:
        return pd.DataFrame(), 'Token count: N/A (no sequence in preprocessing scope)'
    encoding, fallback = resolve_encoding(model)
    records = [token_record(q, s, config, encoding, fallback, model)
               for s in seqs for q in queries_for_sequence(adapter, s, config, h, metric, question_name)]
    scope = 'all dataset episodes / full valid queries' if not external else (
        'full explicitly configured external scope' if external_full_dataset else 'selected external reader unit only (not whole dataset)')
    return pd.DataFrame(records), scope


def table_statistics(table):
    return token_statistics(table.to_dict('records'))


def read_batch_summary(path=None):
    path = Path(path) if path else ROOT / 'outputs/token_counts/token_inventory.csv'
    return pd.read_csv(path) if path.is_file() else None
