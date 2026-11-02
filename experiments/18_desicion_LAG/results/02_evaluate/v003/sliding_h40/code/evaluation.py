"""Export inspectable per-case predictions and explicitly scoped aggregate metrics."""
import csv
import html
from itertools import combinations
from pathlib import Path

from .metrics import match_events, ratio, stream_metrics, summarize
from .outputs import aircraft_choice, estimate_cost, parse_output
from .planning import verify_plan
from .storage import HERE, canonical, file_hash, read_json, read_jsonl, utc_now, write_json


def collect_rows(directory, manifest):
    samples = read_json(directory / "samples.json")
    pricing = read_json(directory / "pricing.json")
    started = {r["query_id"] for r in read_jsonl(directory / "journal.jsonl") if r["event"] == "started"}
    rows = []
    for query in manifest["queries"]:
        sample = samples[query["sample_id"]]
        window = sample["window"]
        path = directory / "responses" / f'{query["id"]}.json'
        result = read_json(path) if path.exists() else {}
        status = result.get("status", "uncertain" if query["id"] in started else "pending")
        parsed, error = None, result.get("error")
        if status in ("ok", "invalid_output"):
            try:
                parsed = parse_output(result["raw"], query["condition"], window,
                                      decisions_task=manifest["config"].get("decisions_task", "binary"))
                status = "ok"
            except (ValueError, TypeError, KeyError) as exc:
                status, error = "invalid_output", str(exc)
        parsed = parsed or {}
        decision, expected = parsed.get("decision"), sample["expected_decision"]
        classification = {("YES", "YES"): "TP", ("NO", "NO"): "TN",
                          ("YES", "NO"): "FP", ("NO", "YES"): "FN"}.get((decision, expected))
        predicted = parsed.get("events")
        matches = match_events(predicted or [], sample["expected_events"], manifest["config"]["step_tolerance"])
        exact = match_events(predicted or [], sample["expected_events"], 0)
        usage = result.get("raw", {}).get("usage")
        expected_aircraft = sorted({e["aircraft_id"] for e in sample["expected_events"]})
        selected_aircraft = parsed.get("selected_aircraft")
        aircraft_supported = query["condition"] == "responses_diagnostic" or (
            query["condition"] == "decisions" and manifest["config"].get("decisions_task") == "aircraft_set")
        rows.append({"query_id": query["id"], "sample_id": query["sample_id"],
                     "condition": query["condition"], "repeat": query["repeat"],
                     "start_step": window["start_step"], "end_step": window["end_step"],
                     "expected_decision": expected, "expected_events": sample["expected_events"],
                     "label_basis": sample["label_basis"], "status": status, "decision": decision,
                     "classification": classification, "predicted_events": predicted,
                     "aircraft_selection_supported": aircraft_supported,
                     "expected_aircraft": expected_aircraft, "expected_aircraft_choice": aircraft_choice(expected_aircraft),
                     "selected_aircraft": selected_aircraft, "aircraft_choice": parsed.get("aircraft_choice"),
                     "aircraft_set_correct": selected_aircraft == expected_aircraft if selected_aircraft is not None else None,
                     "choice_probabilities": parsed.get("choice_probabilities"),
                     "aircraft_probabilities": parsed.get("aircraft_probabilities"),
                     "reason": parsed.get("reason"), "probability_yes": parsed.get("probability_yes"),
                     "matches": matches, "exact_matches": exact,
                     "latency_s": result.get("latency_s"), "usage": usage,
                     "estimated_usd": estimate_cost(usage, query["condition"], pricing), "error": error})
    return rows


def paired_comparisons(rows):
    comparisons = []
    conditions = sorted({r["condition"] for r in rows})
    indexed = {(r["condition"], r["sample_id"], r["repeat"]): r for r in rows}
    for a, b in combinations(conditions, 2):
        pairs = [(r, indexed[(b, r["sample_id"], r["repeat"])]) for r in rows
                 if r["condition"] == a and (b, r["sample_id"], r["repeat"]) in indexed]
        valid = [(x, y) for x, y in pairs if x["status"] == y["status"] == "ok"]
        disagreements = [{"sample_id": x["sample_id"], "repeat": x["repeat"],
                          a: x["decision"], b: y["decision"], "expected": x["expected_decision"]}
                         for x, y in valid if x["decision"] != y["decision"]]
        aircraft_pairs = [(x, y) for x, y in valid if x.get("selected_aircraft") is not None and y.get("selected_aircraft") is not None]
        aircraft_disagreements = [{"sample_id": x["sample_id"], "repeat": x["repeat"],
                                   a: x["aircraft_choice"], b: y["aircraft_choice"], "expected": x["expected_aircraft_choice"]}
                                  for x, y in aircraft_pairs if x["selected_aircraft"] != y["selected_aircraft"]]
        comparisons.append({"a": a, "b": b, "planned_pairs": len(pairs), "valid_pairs": len(valid),
                            "agreement": ratio(len(valid) - len(disagreements), len(valid)),
                            "disagreements": disagreements,
                            "aircraft_valid_pairs": len(aircraft_pairs),
                            "aircraft_set_agreement": ratio(len(aircraft_pairs) - len(aircraft_disagreements), len(aircraft_pairs)),
                            "aircraft_disagreements": aircraft_disagreements,
                            "mean_paired_latency_a_minus_b_s": ratio(sum(x["latency_s"] - y["latency_s"] for x, y in valid), len(valid))})
    return comparisons


def csv_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def display(value):
    return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def write_review_html(directory, manifest, rows):
    def esc(value):
        return html.escape(str(value if value is not None else "—"))
    cards = []
    for row in rows:
        reference = ", ".join(f'{e["aircraft_id"]}@{e["step"]} ({e["label"]})' for e in row["expected_events"]) or "なし（未注釈）"
        predictions = "\n".join(f'{e["aircraft_id"]}@{e["step"]} evidence={e["evidence_steps"]}: {e["reason"]}'
                                for e in row["predicted_events"] or [])
        cards.append(f'''<article data-result="{esc(row['classification'] or row['status'])}">
<h3>{esc(row['sample_id'])} · {row['start_step']}–{row['end_step']} · {esc(row['condition'])} · repeat {row['repeat']}</h3>
<p><b>{esc(row['classification'] or row['status'])}</b>　reference={esc(row['expected_decision'])}　prediction={esc(row['decision'])}　{esc(display(row['latency_s']))} s</p>
<p>参考注釈: {esc(reference)}</p><pre>{esc(predictions)}</pre><p>{esc(row['reason'])}</p>
<p>機体集合: 参考 {esc(row['expected_aircraft_choice'])} → 回答 {esc(row['aircraft_choice'])}</p>
<p>{esc(row['error'] or '')}</p><a href="../inputs/{esc(row['sample_id'])}.txt">入力</a> ·
<a href="../responses/{esc(row['query_id'])}.json">応答</a></article>''')
    content = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>LAG decision review</title>
<style>body{{font:15px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f7f8fa;color:#17212b}}article{{background:white;padding:16px 24px;margin:16px 0;border:1px solid #d9dfe5;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}select{{padding:8px}}a{{color:#1766a7}}</style>
<h1>{esc(manifest['stage'])} / {esc(manifest['prompt_version'])} / {esc(manifest['run_name'])}</h1>
<p>model={esc(manifest['model'])} · H={manifest['config']['history']} · prompt SHA={esc(manifest['prompt_sha256'][:12])}</p>
<p>未注釈窓は暫定負例。同一試合で調整・評価するため、未知試合への汎化性能ではありません。理由は人手で確認してください。</p>
<p><a href="summary.md">集計</a> · <a href="predictions.csv">全応答CSV</a> · <a href="review_template.csv">レビュー用CSV</a> · <a href="../prompt/system.txt">保存プロンプト</a></p>
<label>表示 <select id="filter"><option value="all">すべて</option>''' + "".join(
        f'<option>{label}</option>' for label in ("FN", "FP", "TP", "TN", "invalid_output", "api_error", "uncertain", "pending")) + '''</select></label>
''' + "\n".join(cards) + '''<script>document.querySelector('#filter').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.result!==e.target.value)})</script></html>'''
    (directory / "review.html").write_text(content, encoding="utf-8")


def export_run(directory):
    directory = Path(directory)
    manifest = verify_plan(directory)
    rows = collect_rows(directory, manifest)
    events = read_json(directory / "annotations.json")["events"]
    conditions = manifest["config"]["conditions"]
    summary = {"created_at": utc_now(), "stage": manifest["stage"], "run_name": manifest["run_name"],
               "prompt_version": manifest["prompt_version"], "prompt_sha256": manifest["prompt_sha256"],
               "config": manifest["config"], "limitations": manifest["limitations"],
               "analysis_code_sha256": {p.name: file_hash(p) for p in sorted((HERE / "decision_experiment").glob("*.py"))},
               "conditions": {c: summarize([r for r in rows if r["condition"] == c]) for c in conditions},
               "paired_comparisons": paired_comparisons(rows),
               "streams": {c: stream_metrics([r for r in rows if r["condition"] == c], events)
                           for c in conditions} if manifest["stage"] == "evaluate" else {}}
    report = directory / "report"
    report.mkdir(exist_ok=True)
    write_json(report / "summary.json", summary)
    write_json(report / "predictions.json", rows)
    csv_rows(report / "predictions.csv", rows)
    # Human edits belong in a copy; never overwrite an existing template with annotations.
    if not (report / "review_template.csv").exists():
        review = [{"query_id": r["query_id"], "sample_id": r["sample_id"], "condition": r["condition"],
                   "reference_decision": r["expected_decision"], "human_decision": "",
                   "reason_supported_0_1_2": "", "onset_supported_0_1_2": "", "error_category": "", "notes": ""} for r in rows]
        csv_rows(report / "review_template.csv", review)
    lines = [f'# {manifest["stage"]} / {manifest["prompt_version"]} / {manifest["run_name"]}', "",
             f'Model: `{manifest["model"]}` · H={manifest["config"]["history"]} · stride={manifest["config"]["stride"]}', "",
             f'Prompt SHA256: `{manifest["prompt_sha256"]}`', "",
             "| condition | valid/planned | TP | TN | FP | FN | recall | F2 | p50 s | p95 s | known USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, m in summary["conditions"].items():
        values = [condition, f'{m["valid"]}/{m["planned"]}', *[m[k] for k in ("TP", "TN", "FP", "FN", "recall_valid", "f2_valid", "latency_valid_p50_s", "latency_valid_p95_s", "known_cost_usd")]]
        lines.append("| " + " | ".join(display(v) for v in values) + " |")
    lines += ["", "| condition | aircraft set accuracy | A1 recall | A2 recall |", "|---|---:|---:|---:|"]
    for condition, m in summary["conditions"].items():
        aircraft = m["aircraft_selection"]
        if aircraft:
            values = [condition, aircraft["exact_set_accuracy_valid"], aircraft["per_aircraft"]["A0100"]["recall"], aircraft["per_aircraft"]["A0200"]["recall"]]
            lines.append("| " + " | ".join(display(v) for v in values) + " |")
    lines += ["", "- Classification metrics above use valid outputs only; errors, pending counts and error-as-miss metrics are in summary.json.",
              "- Cost is a usage-based estimate, not an invoice. Unknown-cost attempts are reported separately; null is not zero.",
              "- Step matching requires the same aircraft and is one-to-one. Exact and ±tolerance metrics are both saved.",
              "- Binary event coverage is only an overlapping-window proxy. Localization and duplicate counts require diagnostic output.",
              "- Aircraft-set accuracy requires the exact NONE/A1/A2/BOTH set. It does not establish correct step localization.",
              "- Responses diagnostic and Decisions choice produce different outputs; latency/cost compare these workloads, not an isolated API effect.",
              "- Reasons are for manual review; no automatic score claims that a tactical interpretation is correct.",
              *[f"- {limitation}" for limitation in manifest["limitations"]], "",
              "[Review responses](review.html) · [Predictions](predictions.csv) · [Detailed metrics](summary.json)", ""]
    (report / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_review_html(report, manifest, rows)
    return summary


def compare_runs(directories):
    """A table for iteration history; flag incompatible samples rather than pooling them."""
    runs = []
    for directory in directories:
        directory = Path(directory)
        manifest = verify_plan(directory)
        summary = export_run(directory)
        runs.append((directory, manifest, summary))
    comparable = len({(m["files"]["samples.json"], m["config"]["step_tolerance"])
                      for _, m, _ in runs}) == 1
    lines = [f"Same samples and matching tolerance: {comparable}", "",
             "| run | prompt hash | condition | valid/planned | recall | FP | aircraft set accuracy | event recall | p50 s | known USD |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for directory, manifest, summary in runs:
        for condition, metric in summary["conditions"].items():
            event = metric["event_localization"] or {}
            values = [f'{manifest["stage"]}/{manifest["prompt_version"]}/{directory.name}', manifest["prompt_sha256"][:12], condition,
                      f'{metric["valid"]}/{metric["planned"]}', metric["recall_valid"], metric["FP"],
                      (metric["aircraft_selection"] or {}).get("exact_set_accuracy_valid"), event.get("recall"),
                      metric["latency_valid_p50_s"], metric["known_cost_usd"]]
            lines.append("| " + " | ".join(display(v) for v in values) + " |")
    lines += ["", "Counts include repeats and overlapping windows; these are not independent episodes.",
              "Check manifests for reasoning effort, repeats, request order and pricing before attributing a difference to a prompt or API."]
    return "\n".join(lines)
