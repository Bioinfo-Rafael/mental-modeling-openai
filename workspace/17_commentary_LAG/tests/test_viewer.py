import importlib.util
import json
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('commentary_viewer', HERE/'build_results_viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_snapshot_and_escape(tmp_path):
    path = tmp_path/'journal.jsonl'
    path.write_bytes(b'{"query_id":"a"}\n{"partial":')
    rows, info = viewer.journal_snapshot(path)
    assert rows == [{'query_id':'a'}]
    assert info['trailing_bytes_excluded'] > 0
    payload = {'text':'</script><script>alert(1)</script>'}
    html = viewer.render_html(payload, '__DATASET__')
    assert '<script>' not in html and json.loads(html) == payload
    path.write_text('{bad}\n')
    with pytest.raises(json.JSONDecodeError):
        viewer.journal_snapshot(path)


def test_real_data():
    data = viewer.load_data(HERE/'results/commentary_batch_v1')
    assert len(data['rows']) == 15
    assert set(r['target_step'] for r in data['rows']) == {45,85,145,245,310}
    for row in data['rows']:
        assert row['comparison_id'] == row['sample_id']
        assert row['end']-row['start'] == 60
    html = viewer.render_html(data, (HERE/'templates/results_viewer_template.html').read_text())
    assert '__DATASET__' not in html


def test_runs_keep_overlapping_sample_ids_separate():
    bundle = viewer.load_runs(['commentary_batch_v1', 'tactical_batch_v1'])
    assert bundle['default_run'] == 'tactical_batch_v1'
    old, new = bundle['runs'].values()
    assert len(old['rows']) == len(new['rows']) == 15
    assert old['samples'].keys() == new['samples'].keys()
    for sid in old['samples']:
        assert old['samples'][sid]['window'] == new['samples'][sid]['window']
        assert old['samples'][sid]['messages'] != new['samples'][sid]['messages']
    assert not ({r['query_id'] for r in old['rows']} & {r['query_id'] for r in new['rows']})
    assert viewer.load_runs(['tactical_batch_v1'])['default_run'] == 'tactical_batch_v1'
    with pytest.raises(ValueError):
        viewer.load_runs(['../escape'])
    with pytest.raises(ValueError):
        viewer.load_runs(['commentary_batch_v1', 'commentary_batch_v1'])
