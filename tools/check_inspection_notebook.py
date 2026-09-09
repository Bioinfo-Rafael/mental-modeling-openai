#!/usr/bin/env python3
"""Execute inspection notebooks in fresh kernels, blocking network and raw writes.

Saves test execution artifacts only under outputs/notebook_checks/.
The shipped notebook and existing batch inventory are never overwritten.
"""
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.output_safety import derived_path

GUARD = '''import sys, socket, os
from pathlib import Path
protected = [Path.cwd() / 'data', Path.cwd() / 'upstream']
def audit_raw_writes(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if write and any(path == p or path.is_relative_to(p) for p in protected):
            raise RuntimeError('Protected raw write attempted: ' + str(path))
def no_network(*args, **kwargs):
    raise RuntimeError('Network disabled for inspection verification')
sys.dont_write_bytecode = True
sys.addaudithook(audit_raw_writes)
socket.socket.connect = no_network
socket.create_connection = no_network
'''


def run_case(name, overrides, assertions):
    nb = nbformat.read(ROOT / 'notebooks/01_data_and_token_inspection.ipynb', as_version=4)
    config = next(c for c in nb.cells if 'configuration' in c.metadata.get('tags', []))
    config.source += '\n' + '\n'.join(f'{key} = {value!r}' for key, value in overrides.items())
    nb.cells.insert(0, nbformat.v4.new_code_cell(GUARD))
    nb.cells.append(nbformat.v4.new_code_cell(assertions))
    NotebookClient(nb, timeout=600, kernel_name='mental-modeling',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    output = derived_path(ROOT / 'outputs/notebook_checks' / (name + '.ipynb'))
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    print(f'{name}: all cells executed; assertions passed; network/raw writes blocked', flush=True)
    return {'case': name, 'code_cells': sum(c.cell_type == 'code' for c in nb.cells),
            'status': 'passed', 'api_requests': 0, 'output': str(output.relative_to(ROOT))}


def main():
    results = [
        run_case('MountainCar-v0', {}, '''assert len(datasets) == 25
assert schema_table.matches_inventory.all()
assert len(query_tokens) == 1021
assert set(history_tables) == {0, 1, 2, 3, 5}
assert query.query_index == 10
assert query.system_prompt and query.user_prompt
assert selected_tokens['tokenizer'] == 'o200k_base'
assert not selected_tokens['fallback_encoding_used']
assert len(statistics_table) == 1
assert batch_summary is not None
print('MountainCar assertions passed')'''),
        run_case('f16capstone_raw', {'DATASET': 'f16capstone', 'SEQUENCE_ID': 0}, '''assert len(raw_record['fields']) == 55
assert len(raw_preview) == 5
assert schema_table.matches_inventory.all()
assert config is None and query is None
assert query_tokens.empty
print('F16 raw-only assertions passed')'''),
        run_case('baidu_not_acquired', {'DATASET': 'baidu_fighter_jet'}, '''assert sequence is None
assert query is None and query_tokens.empty
print('Not-acquired case passed')'''),
        run_case('aircombat_static', {'DATASET': 'aircombat_wez', 'ROW_ID': 100,
                 'PREPROCESSING_CONFIG': 'configs/preprocessing/aircombat_wez_default.yaml'}, '''assert raw_record['row_id'] == 100
assert query.history_size == 0
assert selected_tokens['prompt_mode'] == 'external_static'
assert len(history_comparison) == 1
assert set(history_tables) == {0}
assert len(query_tokens) == sequence.length
assert 'selected external reader unit only' in count_scope
print('Explicit external preprocessing assertions passed')'''),
    ]
    path = derived_path(ROOT / 'outputs/notebook_checks/verification.json')
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
