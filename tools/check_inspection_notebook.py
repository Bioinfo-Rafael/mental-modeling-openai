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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['official', 'external', 'both'], default='both')
    args = parser.parse_args()
    reports = []
    for scope in ('official', 'external'):
        if args.scope not in (scope, 'both'):
            continue
        external = scope == 'external'
        name = '02_external_data_and_token_inspection.ipynb' if external else '01_data_and_token_inspection.ipynb'
        nb = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        # Fail if a notebook even reads the other scope's raw data or saved raw inventory.
        other_raw = 'llmx_data' if external else 'candidate_datasets'
        other_schema = 'llmx_schema.json' if external else 'candidate_schema.json'
        isolation = f"""
blocked_raw = Path.cwd() / 'data' / {other_raw!r}
blocked_schema = Path.cwd() / 'outputs/dataset_inventory' / {other_schema!r}
def audit_scope(event, args):
    if event in ('open', 'os.scandir') and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path == blocked_schema or path == blocked_raw or path.is_relative_to(blocked_raw):
            raise RuntimeError('Other dataset scope accessed: ' + str(path))
sys.addaudithook(audit_scope)
"""
        nb.cells.insert(0, nbformat.v4.new_code_cell(GUARD + isolation))
        assertions = """
assert result.metadata['api_requests'] == 0
assert len(largest_token_queries_df) == 20
assert selected_tokens is not None
for name, table in raw_validation.items():
    assert {'matches', 'presence'} <= set(table.columns)
    print('Saved raw comparison:', name, 'differences:', int((~table.matches).sum()))
if 'saved_token_validation_df' in globals():
    assert saved_token_validation_df.query("presence == 'both'").matches.all()
"""
        if external:
            assertions += """
assert len(external_files_df) == len(candidate_inventory['files'])
assert candidate_inventory['source_size_mtime_unchanged']
assert not external_files_df.query("status == 'error'").shape[0]
assert len(all_token_queries_df) == 50905
assert all_token_queries_df.dataset_id.nunique() == 4
assert set(all_token_queries_df.dataset_id) <= set(EXTERNAL)
assert len(token_skipped_df) == 1
assert set(token_scopes_df.query("count_scope == 'deterministic_sample'").dataset_id) == {'trajair', 'calculated_moves'}
import hashlib
r = all_token_queries_df.query("dataset_id == 'f16capstone'").iloc[10]
with inventory_context({'tasks': []}, candidate_inventory):
    ext = get_adapter('f16capstone')
seq, raw = select_record(ext, sequence_id=int(r.sequence_id), index=int(r.query_index))
cfg, _ = preprocessing_for_inspection('f16capstone', 'configs/preprocessing/f16capstone_default.yaml')
q = selected_query(ext, seq, int(r.query_index), cfg, int(r.history_size))
assert hashlib.sha256(q.user_prompt.encode()).hexdigest() == r.user_prompt_sha256
"""
        else:
            assertions += """
assert len(official_tasks_df) == 20
assert len(official_files_df) == 172
assert len(official_arrays_df) == sum(len(f['arrays']) for t in llmx_inventory['tasks'] for f in t['files'])
assert len(all_token_queries_df) == 10095
assert all_token_queries_df.dataset_id.nunique() == 11
assert len(token_skipped_df) == 9
assert set(all_token_queries_df.prompt_mode) == {'original_llmx'}
"""
        nb.cells.append(nbformat.v4.new_code_cell(assertions))
        def progress(cell, cell_index, **kwargs):
            print(f"{scope}: cell {cell_index}: {cell.source.splitlines()[0][:100]}", flush=True)
        output = derived_path(ROOT / f'outputs/notebook_checks/live_{scope}_split.ipynb')
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            NotebookClient(nb, timeout=1800, kernel_name='mental-modeling',
                           resources={'metadata': {'path': str(ROOT)}},
                           on_cell_execute=progress).execute()
        finally:
            nbformat.write(nb, output)
        reports.append(dict(scope=scope, status='passed', api_requests=0,
                            other_scope_reads=0, output=str(output.relative_to(ROOT))))
    path = derived_path(ROOT / f'outputs/notebook_checks/split_verification_{args.scope}.json')
    path.write_text(json.dumps(reports, indent=2) + '\n')
    print(reports)


if __name__ == '__main__':
    main()
