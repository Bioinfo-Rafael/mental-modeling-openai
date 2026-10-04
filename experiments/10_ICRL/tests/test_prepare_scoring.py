import csv
from importlib import import_module
import json
from pathlib import Path
import shutil

import pytest

prepare = import_module('experiments.10_ICRL.reasoning_score_analysis.prepare_scoring')


@pytest.fixture
def source(tmp_path):
    api = tmp_path / 'results/api'
    api.mkdir(parents=True)
    for name in ('manifest.json', 'records.jsonl'):
        shutil.copyfile(prepare.ROOT / 'results/api' / name, api / name)
    return tmp_path


def test_roundtrip_empty_scores_and_preserved_sources(source, monkeypatch):
    import openai
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: pytest.fail('No API client allowed'))
    inputs = list((source / 'results/api').iterdir())
    hashes = {p: prepare.sha256(p) for p in inputs}
    report = prepare.prepare(source)
    destination = source / 'reasoning_for_scoring.csv'
    assert destination.read_bytes().startswith(b'\xef\xbb\xbf')
    with destination.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(prepare.FIELDS)
    records = [json.loads(line) for line in (source / 'results/api/records.jsonl').read_text().splitlines()]
    assert report['rows'] == report['empty_scores'] == len(rows) == 72
    assert [r['query_id'] for r in rows] == [r['query_id'] for r in records]
    assert all(r['score'] == '' and r['Reasoning'] == prepare.extract(raw['assistant_text'])
               for r, raw in zip(rows, records))
    assert all(r['K'] == '' for r in rows if r['method'] in ('direct', 'ppo_prior'))
    assert all(prepare.sha256(p) == digest for p, digest in hashes.items())
    assert not (source / 'reasoning_for_scoring_scored_astra_high.csv').exists()
    assert prepare.extract is prepare.previous.extract
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        prepare.prepare(source)
    assert destination.read_bytes() == before


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'metadata', 'raw_text', 'heading'])
def test_invalid_inputs_do_not_create_csv(source, problem):
    path = source / 'results/api/records.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if problem == 'missing':
        rows.pop()
    elif problem == 'duplicate':
        rows[-1] = rows[0]
    elif problem == 'metadata':
        rows[0]['H'] = 5
    elif problem == 'raw_text':
        rows[0]['assistant_text'] += 'changed'
    else:
        rows[0]['assistant_text'] = 'No headings'
        rows[0]['raw_response']['choices'][0]['message']['content'] = 'No headings'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(ValueError):
        prepare.prepare(source)
    assert not (source / 'reasoning_for_scoring.csv').exists()


def test_existing_scored_file_is_protected(source):
    path = source / 'reasoning_for_scoring_scored_astra_high.csv'
    path.write_text('existing scores')
    with pytest.raises(FileExistsError):
        prepare.prepare(source)
    assert path.read_text() == 'existing scores'
    assert not (source / 'reasoning_for_scoring.csv').exists()
