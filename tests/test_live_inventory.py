import json

import pandas as pd
import pytest

from dataset_adapters import base
from dataset_adapters.base import ROOT, inventories, inventory_context
from tools.inventory_llmx_data import create_inventory
from tools.live_inspection import official_frames, external_frames, unified_overview, compare_frames
from tools.token_inventory import build_token_inventory, save_token_inventory


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('No network allowed')
    monkeypatch.setattr(socket.socket, 'connect', blocked)


def test_live_context_never_reads_saved_inventory(monkeypatch):
    official = create_inventory(ROOT / 'data/llmx_data')
    external = inventories()[1]  # fixture only; unavailable within the tested live context
    def forbidden():
        raise AssertionError('Saved inventory must not be consulted in live context')
    monkeypatch.setattr(base, '_saved_inventories', forbidden)
    with inventory_context(official, external):
        assert inventories()[0] is official
        overview = unified_overview(official, external)
        result = build_token_inventory('MountainCar-v0')
    assert len(overview) == 25
    assert len(result.queries) == 1021


def test_library_has_no_writers_and_preserves_stats(monkeypatch):
    import tools.token_inventory as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Library must not write files')
    monkeypatch.setattr(module, '_atomic_text', forbidden)
    monkeypatch.setattr(module, '_write_csv', forbidden)
    result = build_token_inventory(['MountainCar-v0', 'Pendulum-v1'], history_sizes=[0, 1])
    assert len(result.summary) == 4
    mountain_h1 = next(r for r in result.summary if r['dataset_id'] == 'MountainCar-v0' and r['history_size'] == 1)
    assert mountain_h1['total_input_tokens'] == 656880
    assert len(result.by_sequence) == 40
    assert len(result.preprocessing_configs) == 2


def test_sampling_is_repeatable_and_streaming_equivalent():
    kwargs = dict(datasets='f16capstone', history_sizes=[1], external_sample_threshold=1,
                  external_sample_size=7, seed=0)
    first = build_token_inventory(**kwargs)
    streamed = []
    second = build_token_inventory(**kwargs, collect_queries=False, query_sink=streamed.append)
    assert first.queries == streamed and second.queries == []
    assert first.summary == second.summary
    assert first.scopes[0]['eligible_query_pool'] == 43041
    assert first.scopes[0]['count_scope'] == 'deterministic_sample'
    assert first.scopes[0]['sample_size'] == 7
    assert {r['seed'] for r in first.queries} == {0}


def test_static_mode_and_skip_reasons():
    result = build_token_inventory(['aircombat_wez', 'HalfCheetah-v4', 'baidu_fighter_jet'],
                                  history_sizes=[1, 5], external_sample_threshold=1, external_sample_size=3)
    assert len(result.queries) == 3
    assert {r['history_size'] for r in result.queries} == {0}
    assert {r['prompt_mode'] for r in result.queries} == {'external_static'}
    assert all(r['row_id'] is not None for r in result.queries)
    assert len(result.skipped) == 2


def test_live_frame_shapes_and_validation_detects_changes():
    official, external = inventories()
    tasks, files, arrays = official_frames(official)
    assert len(tasks) == 20 and len(files) == 172
    assert len(arrays) == sum(len(f['arrays']) for t in official['tasks'] for f in t['files'])
    assert not arrays.duplicated(['task', 'episode_id', 'key']).any()
    changed = arrays.copy()
    changed.loc[0, 'dtype'] = 'changed-dtype'
    comparison = compare_frames(arrays, changed, ['task', 'relative_path', 'key'], ['shape', 'dtype'])
    assert (~comparison.matches).sum() == 1
    ds, ext_files, groups, columns, ext_arrays = external_frames(external)
    assert len(ext_files) == 34036 and len(groups) == 82
    assert set(ext_arrays['format']) >= {'.mat', '.ulg'}
    assert {'shape', 'dtype', 'ndim'} <= set(ext_arrays)


def test_explicit_writer_only_uses_output_path(tmp_path):
    result = build_token_inventory('MountainCar-v0', history_sizes=[1])
    save_token_inventory(result, tmp_path)
    csv = pd.read_csv(tmp_path / 'queries.csv')
    assert len(csv) == 1021
    assert json.loads((tmp_path / 'summary.json').read_text())['number_of_queries'] == 1021
    with pytest.raises(ValueError):
        save_token_inventory(result, ROOT / 'data/do_not_write')


def test_csv_numeric_comparison_does_not_confuse_integer_and_float():
    a = pd.DataFrame([dict(key='x', median=643, std=1.123456789012345)])
    b = pd.DataFrame([dict(key='x', median=643.0, std=1.1234567890123448)])
    assert compare_frames(a, b, ['key'], ['median', 'std'], float_columns=['std']).matches.all()
    b.loc[0, 'median'] = 644
    assert not compare_frames(a, b, ['key'], ['median', 'std'], float_columns=['std']).matches.all()
