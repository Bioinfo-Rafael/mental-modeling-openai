"""Check replay exports against saved experimental evidence."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('foresight', Path(__file__).with_name('foresight.py'))
foresight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foresight)


def test_saved_results_and_translations():
    root = Path(__file__).resolve().parents[2]
    runs = foresight.build_foresight(root)
    assert [r['offsets'] for r in runs] == [[-1, 1, 30], [1, 30], [1]]
    assert [len(r['events']) for r in runs] == [18, 12, 21]
    for run, (_, experiment, name) in zip(runs, foresight.EXPERIMENTS):
        responses = json.loads((root / experiment / 'results' / name / 'report/responses.json').read_text())
        originals = {r['query_id']: r for r in responses}
        for event in run['events']:
            original = originals[event['query_id']]
            assert original['model'] == 'gpt-5.6-sol'
            assert event['en'] == original['narratives']
            assert set(event['ja']) == {'A0100', 'A0200', 'B0100', 'B0200'}
            assert all(event['ja'].values())
    assert [e['step'] for e in runs[0]['events'] if e['offset'] == 30] == [46, 49, 83, 95, 143, 243]
    assert [e['step'] for e in runs[2]['events']] == list(range(30, 331, 15))


def test_missing_files_are_empty_and_stale_translation_is_not_used(tmp_path):
    assert all(not r['events'] for r in foresight.build_foresight(tmp_path))
    report = tmp_path / '20_foresight_LAG/results/foresight_short_v1/report'
    report.mkdir(parents=True)
    response = {'model': 'gpt-5.6-sol', 'event': {'step': 46}, 'offset': 30,
                'query_id': 'q', 'status': 'valid', 'narratives': {'A0100': 'updated'},
                'observation_range': [16, 76], 'target_interval': [46, 76]}
    (report / 'responses.json').write_text(json.dumps([response]))
    (report / 'translations_ja.json').write_text(json.dumps({'records': [
        {'query_id': 'q', 'aircraft_id': 'A0100', 'english': 'old', 'japanese': '古い訳'}]}))
    event = foresight.build_foresight(tmp_path)[0]['events'][0]
    assert event['en']['A0100'] == 'updated'
    assert event['ja'] == {}
