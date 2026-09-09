import hashlib
import json
from pathlib import Path

import pytest

from dataset_adapters.base import ROOT
from dataset_adapters.registry import get_adapter
from tools.notebook_inspection import (
    dataset_table, select_record, inspect_selected_file, preprocessing_for_inspection,
    selected_query, dataset_token_table, table_statistics, read_batch_summary,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('Network forbidden in notebook inspection tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)


def test_all_datasets_and_unacquired():
    table = dataset_table()
    assert len(table) == 25
    assert (table.category == 'official').sum() == 20
    assert table.set_index('dataset_id').loc['baidu_fighter_jet', 'status'] == 'NOT_ACQUIRED'
    assert select_record(get_adapter('baidu_fighter_jet')) == (None, None)


@pytest.mark.parametrize('dataset', ['MountainCar-v0', 'f16capstone', 'trajair', 'aircombat_wez'])
def test_raw_schema_actual_matches_inventory(dataset):
    adapter = get_adapter(dataset)
    seq, record = select_record(adapter, index=10)
    before = hashlib.sha256((ROOT / seq.source_file).read_bytes()).hexdigest()
    table, measures, actual = inspect_selected_file(adapter, seq)
    assert not table.empty
    assert table.matches_inventory.all(), table.to_string()
    assert measures.matches.all()
    assert record['representation'] == 'raw'
    assert hashlib.sha256((ROOT / seq.source_file).read_bytes()).hexdigest() == before


def test_external_requires_explicit_configuration_and_scope_guard():
    assert preprocessing_for_inspection('f16capstone')[0] is None
    config, _ = preprocessing_for_inspection('f16capstone', 'configs/preprocessing/f16capstone_default.yaml')
    adapter = get_adapter('f16capstone')
    table, scope = dataset_token_table(adapter, config, 1, 'gpt-4o', external_sequence=0,
                                     max_external_queries=1)
    assert table.empty and 'Not executed' in scope
    assert preprocessing_for_inspection('HalfCheetah-v4')[0] is None


def test_full_mountaincar_counts_and_invalid_position():
    adapter = get_adapter('MountainCar-v0')
    seq, _ = select_record(adapter)
    config, _ = preprocessing_for_inspection(adapter.dataset_id)
    query = selected_query(adapter, seq, 10, config, 1)
    assert query.query_index == 10 and query.history_start == 8 and query.history_end == 10
    with pytest.raises(ValueError, match='No valid query'):
        selected_query(adapter, seq, 0, config, 1)
    table, scope = dataset_token_table(adapter, config, 1, 'gpt-4o')
    assert len(table) == 1021 and table.episode_id.nunique() == 10
    batch = read_batch_summary()
    expected = batch[(batch.dataset_id == 'MountainCar-v0') & (batch.history_size == 1)]
    assert len(expected) == 1
    assert table_statistics(table)['total_input_tokens'] == int(expected.iloc[0].total_input_tokens)
    assert table.preprocessing_config_sha256.nunique() == 1
    assert 'all dataset episodes' in scope


def test_missing_batch_does_not_execute(tmp_path):
    assert read_batch_summary(tmp_path / 'missing.csv') is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('external', [False, True])
def test_notebook_source_has_no_writer_or_api(external):
    filename = '02_external_data_and_token_inspection.ipynb' if external else '01_data_and_token_inspection.ipynb'
    nb = json.loads((ROOT / 'notebooks' / filename).read_text())
    source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    for forbidden in ['to_csv(', 'write_text(', 'savefig(', 'OpenAI(', 'responses.create', 'execute-paid-api']:
        assert forbidden not in source
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert 'DATASET =' not in ''.join(code_cells[0]['source'])
    tags = [c['metadata'].get('tags', []) for c in code_cells]
    selection = next(i for i, t in enumerate(tags) if 'selection' in t)
    scan_tag = 'live-external-scan' if external else 'live-official-scan'
    scan_index = next(i for i, t in enumerate(tags) if scan_tag in t)
    assert selection > scan_index
    assert 'build_token_inventory(' in source
    assert 'datasets="everything"' not in source
    if external:
        assert 'llmx_inventory' not in source
        assert 'create_inventory(' not in source
        assert 'datasets="external-all"' in source
    else:
        assert 'candidate_inventory' not in source
        assert 'inventory_candidates' not in source
        assert 'external_frames(' not in source
        assert 'datasets="all"' in source
    for cell in code_cells:
        compile(''.join(cell['source']), filename, 'exec')
