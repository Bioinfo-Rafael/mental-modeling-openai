"""Offline checks for future execution; synthetic scores never leave tmp_path."""
import csv
from importlib import import_module
import json
from pathlib import Path
import shutil
from statistics import mean

import pytest

data = import_module('experiments.10_ICRL.analysis.results_data')
plots = import_module('experiments.10_ICRL.analysis.plot_comparisons')
viewer = import_module('experiments.10_ICRL.analysis.build_results_viewer')


@pytest.fixture
def source(tmp_path):
    api = tmp_path / 'results/api'
    api.mkdir(parents=True)
    for name in ('manifest.json', 'records.jsonl', 'requests.jsonl', 'responses.jsonl'):
        shutil.copyfile(data.HERE / 'results/api' / name, api / name)
    return tmp_path


def write_scores(root, rows):
    with (root / 'reasoning_for_scoring_scored_astra_high.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=data.prepare.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fixture_scores(root):
    records = data.read_journal(root / 'results/api/records.jsonl')
    rows = []
    for i, r in enumerate(records):
        row = {k: r[k] for k in data.prepare.FIELDS if k in r}
        row.update(model=r['model_alias'], Task=r['task'], Metrics=r['metric'],
                   reward_present=int(r['reward_present']), Reasoning=data.prepare.extract(r['assistant_text']),
                   score=str(i % 4 + 1))  # Synthetic test data, not judgments.
        rows.append(row)
    write_scores(root, rows)
    return rows


def test_missing_scores_allow_viewer_but_prevent_plot_output(source, monkeypatch):
    rows, hashes = data.load_rows(source)
    assert len(rows) == 72 and all(r['reasoning_score'] is None for r in rows)
    load = data.load_samples
    monkeypatch.setattr(data, 'load_samples', lambda: load(source))
    output = source / 'comparisons'
    with pytest.raises(ValueError, match='all 72 reasoning scores'):
        plots.main(['--output-dir', str(output)])
    assert not output.exists()


def test_all_requested_pooling_counts_and_means(source):
    fixture_scores(source)
    samples, hashes = data.load_samples(source)
    tables = plots.make_tables(samples)
    assert len(samples) == 72
    for table, keys, pooled_n, model_n, pooled_count in (
        ('prompt_summary', ('method',), 18, 9, 4),
        ('episode_summary', ('E',), 24, 12, 3),
        ('matrix_summary', ('method', 'E'), 6, 3, 12),
    ):
        rows = tables[table]
        assert len(rows) == pooled_count * 3
        for row in rows:
            selected = [r for r in samples if all(r[k] == row[k] for k in keys)
                        and (row['model'] == 'pooled' or r['model'] == row['model'])]
            assert row['n'] == len(selected) == (pooled_n if row['model'] == 'pooled' else model_n)
            assert row['accuracy_pct'] == pytest.approx(100 * mean(r['correct'] for r in selected))
            assert row['action_mae'] == pytest.approx(mean(r['absolute_error'] for r in selected))
            assert row['reasoning_mean'] == pytest.approx(mean(r['reasoning_score'] for r in selected))
    assert data.METHOD_LABELS == ('Direct', 'PPO', 'ICRL (K=2)', 'ICRL (K=5)')
    assert data.EPISODES == (1, 3, 9)


def test_blank_manual_score_remains_unscored(source):
    scores = fixture_scores(source)
    scores[0]['score'] = ''
    write_scores(source, scores)
    rows, _ = data.load_rows(source)
    assert sum(r['reasoning_score'] is None for r in rows) == 1
    with pytest.raises(ValueError, match='all 72 reasoning scores'):
        data.load_samples(source)


@pytest.mark.parametrize('problem', ['duplicate', 'unknown_id', 'metadata', 'body', 'invalid_score'])
def test_bad_reasoning_join_is_rejected(source, problem):
    scores = fixture_scores(source)
    if problem == 'duplicate':
        scores[-1] = scores[0]
    elif problem == 'unknown_id':
        scores[0]['query_id'] = 'not-a-planned-query'
    elif problem == 'metadata':
        scores[0]['E'] = 9
    elif problem == 'body':
        scores[0]['Reasoning'] += 'changed'
    else:
        scores[0]['score'] = '0'
    write_scores(source, scores)
    with pytest.raises(ValueError):
        data.load_rows(source)


def test_html_payload_escapes_saved_markup_without_changing_text(source):
    rows, hashes = data.load_rows(source)
    original = '</script><script>alert("saved text")</script>'
    rows[0]['assistant_text'] = original
    template = (data.HERE / 'templates/results_viewer_template.html').read_text()
    html = viewer.render_html(rows, hashes, template)
    payload = html.split('<script id="dataset" type="application/json">', 1)[1].split('</script>', 1)[0]
    assert '<' not in payload and '__DATASET__' not in html
    decoded = json.loads(payload)
    assert decoded['rows'][0]['assistant_text'] == original
    assert decoded['reasoning_scored'] == 0
    assert len(decoded['rows']) == 72
    assert 'r.shots' not in html and 'score_pattern' not in html


def test_existing_outputs_are_not_overwritten(tmp_path):
    path = tmp_path / 'existing.html'
    path.write_text('original')
    with pytest.raises(FileExistsError):
        viewer.main(['--output', str(path)])
    assert path.read_text() == 'original'
    with pytest.raises(FileExistsError):
        plots.main(['--output-dir', str(tmp_path)])
