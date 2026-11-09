"""The one-off schema-2 run converter: explicit kind map, new directory, payloads untouched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from translate.journal import REPO_ROOT

SPEC = importlib.util.spec_from_file_location("upgrade_events_v2", REPO_ROOT / "scripts" / "upgrade_events_v2.py")
upgrade_events_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upgrade_events_v2)


def _v2_run(directory: Path) -> Path:
    directory.mkdir()
    reasons = [{"kind": "initial", "source": "preset", "reference_id": "preset-step-45", "probability": None}]
    event = {"schema_version": 2, "record_type": "translation_event", "primary_trigger": "initial"}
    lines = {
        "events.jsonl": {**event, "trigger_reasons": reasons},
        "requests.jsonl": {"schema_version": 2, "call_id": "c", "payload": {"model": "m"}},
        "responses.jsonl": {"schema_version": 2, "call_id": "c", "status": "ok"},
    }
    for name, record in lines.items():
        (directory / name).write_text(json.dumps(record) + "\n", encoding="utf-8")
    (directory / "run.json").write_text("{}", encoding="utf-8")
    return directory


def test_upgrade_maps_every_kind_into_a_new_schema_3_run(tmp_path: Path) -> None:
    run = _v2_run(tmp_path / "old")

    code = upgrade_events_v2.main(["--run", str(run), "--out", str(tmp_path / "new"), "--map", "initial=start"])

    (event,) = [json.loads(line) for line in (tmp_path / "new" / "events.jsonl").read_text().splitlines()]
    (request,) = [json.loads(line) for line in (tmp_path / "new" / "requests.jsonl").read_text().splitlines()]
    assert code == 0
    kinds = (event["primary_trigger"], event["trigger_reasons"][0]["kind"])
    assert (event["schema_version"], kinds) == (3, ("start", "start"))
    assert request == {"schema_version": 3, "call_id": "c", "payload": {"model": "m"}}
    assert json.loads((run / "events.jsonl").read_text())["schema_version"] == 2  # the old run is untouched


def test_upgrade_refuses_an_unmapped_kind(tmp_path: Path) -> None:
    run = _v2_run(tmp_path / "old")

    with pytest.raises(SystemExit) as exc:
        upgrade_events_v2.main(["--run", str(run), "--out", str(tmp_path / "new"), "--map", "start=start"])

    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()
