"""Read-only monitor ETA uses the bottleneck queue and excludes bootstrap work."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch_watch as watch


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def setup_run(tmp_path, counts=(20, 10, 5), *, error=False):
    folder = tmp_path / 'api/batch'
    jobs = [dict(name='boot', model='sol', query_ids=['b'], collected=True, successful=1)]
    for i, (alias, completed) in enumerate(zip(('sol', 'terra', 'luna'), counts)):
        jobs.append(dict(name=alias, model=alias, query_ids=[str(n) for n in range(100)], batch_id=alias))
        write(folder / f'{alias}.remote.json', dict(id=alias, created_at=1000+i, status='in_progress',
              request_counts=dict(completed=completed, failed=int(error and alias=='sol'))))
    write(folder / 'state.json', dict(scope='rolling-comparison', bootstrap_jobs=['boot'], jobs=jobs,
          sets=[dict(admitted=True, jobs=['sol', 'terra', 'luna'])]))
    return folder


def test_eta_is_bottleneck_not_sum_and_monitor_does_not_write(tmp_path):
    setup_run(tmp_path)
    before = {p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    s = watch.snapshot(tmp_path, now=1100)
    assert s['elapsed'] == 100
    assert s['eta_seconds'] == 1900  # Luna: 5 successes / 100 seconds, 95 left.
    assert sum(s['planned'].values()) == 300
    assert sum(s['completed'].values()) == 35
    assert 'JST' in watch.render(s)
    assert before == {p:p.read_bytes() for p in tmp_path.rglob('*.json')}


@pytest.mark.parametrize('counts,error', [((10, 0, 5), False), ((10, 5, 5), True)])
def test_no_eta_without_all_model_samples_or_after_failure(tmp_path, counts, error):
    setup_run(tmp_path, counts, error=error)
    assert watch.snapshot(tmp_path, now=1100)['eta_seconds'] is None


def test_finished_and_stale_remote_identity(tmp_path):
    folder = setup_run(tmp_path, (100, 100, 100))
    assert watch.snapshot(tmp_path, now=1100)['eta_seconds'] == 0
    remote = json.loads((folder/'luna.remote.json').read_text())
    remote['id'] = 'different-batch'
    write(folder/'luna.remote.json', remote)
    assert watch.snapshot(tmp_path, now=1100)['eta_seconds'] is None
