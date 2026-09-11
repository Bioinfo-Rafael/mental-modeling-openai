"""Synthetic fixtures only. Never read or modify the running Exp.05 results."""
from importlib import import_module
import json
import pytest

estimate = import_module("experiments.05_new_models_single.analysis.estimate_exp06")


@pytest.fixture
def pilot():
    records, conditions = [], []
    for model in estimate.MODELS:
        for metric in estimate.METRICS:
            cid = f"{model}__{metric}"
            conditions.append({"condition_id": cid, "model": model, "status": "complete", "condition_elapsed_seconds": 40.0})
            for i in range(10):
                records.append({
                    "query_id": f"{cid}-{i}", "condition_id": cid, "model": model, "metric": metric,
                    "task": "Pendulum-v1", "H": 20, "ordinal": i, "status": "ignored",
                    "api_request_made": True, "reused": False, "api_attempts": 1,
                    "request": {"model": model, "messages": [], "reasoning_effort": "medium"},
                    "raw_response": {"usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                        "total_tokens": 1200, "completion_tokens_details": {"reasoning_tokens": 150}}},
                    "query_elapsed_seconds": 3.0, "request_elapsed_seconds": 2.0,
                })
    summary = {"experiment": estimate.EXPERIMENT, "status": "complete", "successful_queries": 120,
               "logical_queries": 120, "api_attempts": 120, "failed_queries": 0, "retry_attempts": 0,
               "reused_queries": 0, "experiment_elapsed_seconds": 500.0, "conditions": conditions}
    manifest = {"experiment": estimate.EXPERIMENT, "queries": records}
    prices = json.loads((estimate.HERE / "estimate_prices.json").read_text())
    return summary, manifest, records, prices


def test_tokens_cost_time_scaling_and_reasoning_not_double_counted(pilot):
    report = estimate.build_estimate(*pilot)
    extra = report["scenarios"]["exp06_additional_estimate"]
    sol = extra["models"][0]
    assert sol["queries"] == 280
    assert sol["input_tokens"] == 280_000
    assert sol["output_tokens"] == 56_000  # Includes reasoning; do not add 42,000 again.
    assert sol["input_usd"] == pytest.approx(1.12)
    assert sol["output_usd"] == pytest.approx(1.12)
    assert sol["total_usd"] == pytest.approx(2.24)
    assert sol["api_seconds"] == 560 and sol["query_seconds"] == 840
    assert sol["condition_seconds"] == 1120
    assert extra["experiment_seconds"] == 3500
    assert extra["total"]["queries"] == 840
    assert extra["total"]["total_usd"] == pytest.approx(2.24 + 1.232 + 0.1232)
    total = report["scenarios"]["exp05_plus_06_estimate"]
    assert total["total"]["queries"] == 960 and total["experiment_seconds"] == 4000
    assert report["details"][estimate.MODELS[0]]["cached_input"] == {"tokens": None, "known_queries": 0}
    assert report["details"][estimate.MODELS[0]]["reasoning_output"]["tokens"] == 6000
    text = estimate.markdown(report)
    assert "×7" in text and "×8" in text and "input金額 USD" in text and "全モデル合計" in text


@pytest.mark.parametrize("damage", ["running", "missing", "duplicate", "usage", "time", "retry", "old_temperature", "wrong_grid"])
def test_reject_incomplete_or_invalid_data(pilot, damage):
    summary, _, records, _ = pilot
    if damage == "running":
        summary["status"] = "running"
    elif damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records[-1] = records[0]
    elif damage == "usage":
        records[0]["raw_response"]["usage"]["completion_tokens"] = None
    elif damage == "time":
        records[0]["request_elapsed_seconds"] = None
    elif damage == "retry":
        summary["api_attempts"] = 121
    elif damage == "old_temperature":
        records[0]["request"]["temperature"] = 0
    else:
        records[0]["H"] = 10
    with pytest.raises(ValueError):
        estimate.build_estimate(*pilot)


def write_fixture(directory, pilot):
    summary, manifest, records, _ = pilot
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_cli_read_only_and_unique_outputs(pilot, tmp_path, monkeypatch):
    source = tmp_path / "results"
    write_fixture(source, pilot)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    prices = estimate.HERE / "estimate_prices.json"
    monkeypatch.setattr(estimate, "HERE", tmp_path / "analysis")
    args = ["--results", str(source), "--prices", str(prices)]
    assert estimate.main(args) == 0
    assert estimate.main(args) == 0
    outputs = list((tmp_path / "analysis/cost_estimates").iterdir())
    assert len(outputs) == 2
    for directory in outputs:
        assert (directory / "RESULTS.md").is_file()
        report = json.loads((directory / "estimate.json").read_text())
        assert len(report["source_sha256"]) == 3 and len(report["price_sha256"]) == 64
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_running_cli_creates_no_output(tmp_path, monkeypatch):
    source = tmp_path / "results"
    source.mkdir()
    (source / "summary.json").write_text('{"status": "running"}')
    monkeypatch.setattr(estimate, "HERE", tmp_path / "analysis")
    assert estimate.main(["--results", str(source)]) == 2
    assert not (tmp_path / "analysis").exists()
