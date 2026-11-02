"""Build a standalone offline input/output viewer; never change saved results."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from decision_experiment.evaluation import collect_rows
from decision_experiment.metrics import summarize
from decision_experiment.planning import request_for, verify_plan
from decision_experiment.storage import RESULTS, file_hash, read_json, resolve_run


def load_run(directory):
    manifest = verify_plan(directory)
    rows = collect_rows(directory, manifest)
    requests, responses = {}, {}
    queries = {q["id"]: q for q in manifest["queries"]}
    for row in rows:
        query = queries[row["query_id"]]
        sha = query["request_sha256"]
        if sha not in requests:
            requests[sha] = request_for(directory, manifest, query)
        row["request_sha256"] = sha
        path = directory / "responses" / f'{query["id"]}.json'
        if path.exists():
            responses[query["id"]] = read_json(path)
    return {"id": str(directory.relative_to(RESULTS)),
            "stage": manifest["stage"], "prompt_version": manifest["prompt_version"],
            "run_name": manifest["run_name"], "model": manifest["model"],
            "prompt_sha256": manifest["prompt_sha256"], "created_at": manifest["created_at"],
            "latest_response_at": max((r.get("finished_at", "") for r in responses.values()), default=""),
            "config": manifest["config"], "source": manifest["source"],
            "manifest_sha256": file_hash(directory / "manifest.json"),
            "rows": rows, "samples": read_json(directory / "samples.json"),
            "requests": requests, "responses": responses,
            "summary": {c: summarize([r for r in rows if r["condition"] == c])
                        for c in manifest["config"]["conditions"]}}


def render_html(bundle):
    template = (HERE / "templates/results_viewer.html").read_text(encoding="utf-8")
    if template.count("__DATASET__") != 1:
        raise ValueError("Expected one dataset placeholder")
    if template.count("__ANALYSIS_SCRIPT__") != 1:
        raise ValueError("Expected one analysis script placeholder")
    template = template.replace("__ANALYSIS_SCRIPT__", (HERE / "templates/analysis.js").read_text(encoding="utf-8"))
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # A model response must not terminate the JSON script block or inject markup.
    for char, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                         ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, escape)
    return template.replace("__DATASET__", payload)


def build_viewer(directories=None, output=None, include_planned=False):
    candidates = directories if directories is not None else [p.parent for p in sorted(RESULTS.glob("*/*/*/manifest.json"))]
    runs = []
    for directory in candidates:
        directory = Path(directory).resolve()
        attempted = (directory / "journal.jsonl").exists() or any((directory / "responses").glob("*.json"))
        if not include_planned and not attempted:
            continue
        runs.append(load_run(directory))
    if not runs:
        raise ValueError("No attempted runs found. Use --include-planned to include offline plans.")
    runs.sort(key=lambda r: (r["latest_response_at"], r["created_at"]))
    bundle = {"created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "default_run": runs[-1]["id"], "runs": runs}
    output = Path(output) if output else HERE / "view_rawdata.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(bundle), encoding="utf-8")
    return output, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", help="Stage/prompt/run path; repeat to include multiple runs")
    parser.add_argument("--include-planned", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directories = [resolve_run(value) for value in args.run] if args.run else None
    output, bundle = build_viewer(directories, args.output, args.include_planned)
    print(f'{output}\n{len(bundle["runs"])} runs / {sum(len(r["rows"]) for r in bundle["runs"])} queries; offline, no API calls')


if __name__ == "__main__":
    main()
