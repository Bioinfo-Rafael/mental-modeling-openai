"""Read a completed Exp.05; estimate additional Exp.06 usage without any API calls.

Standalone standard-library code: deliberately does not import the experiment runner.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
EXPERIMENT = "05_new_models_single"
MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
METRICS = ("next-action", "last-action", "next-state", "last-state")
SCENARIOS = {"exp05_observed": 1, "exp06_additional_estimate": 7, "exp05_plus_06_estimate": 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, integer=False):
    """Missing usage is an error, not a fabricated zero."""
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Missing/invalid {name}: {value!r}")
    require(not integer or type(value) is int, f"Expected integer {name}")
    return value


def load_completed(results):
    """Read-only snapshot; refuse running/incomplete runs and changing files."""
    summary_bytes = (results / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    require(summary.get("status") == "complete", "05 is not complete; wait until execution finishes.")
    blobs = {"summary.json": summary_bytes}
    for name in ("manifest.json", "records.jsonl"):
        blobs[name] = (results / name).read_bytes()
    for name, content in blobs.items():
        require((results / name).read_bytes() == content, f"Source changed while reading: {name}")
    records = [json.loads(line) for line in blobs["records.jsonl"].splitlines() if line.strip()]
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in blobs.items()}
    return summary, json.loads(blobs["manifest.json"]), records, hashes


def validate_run(summary, manifest, records):
    """Require the intended 3 models × 4 metrics × N10 pilot, without retries/reuse."""
    require(summary.get("experiment") == manifest.get("experiment") == EXPERIMENT, "Not an Exp.05 run")
    require(summary.get("status") == "complete", "05 is not complete")
    for key, expected in {"successful_queries": 120, "logical_queries": 120, "api_attempts": 120,
                          "failed_queries": 0, "retry_attempts": 0, "reused_queries": 0}.items():
        require(summary.get(key) == expected, f"Expected {key}={expected}; retry/partial runs are not supported")
    planned = manifest["queries"]
    ids = [r["query_id"] for r in records]
    require(len(records) == len(planned) == len(set(ids)) == 120, "Missing/duplicate pilot queries")
    require(set(ids) == {q["query_id"] for q in planned}, "Records differ from manifest")
    expected_grid = Counter({(model, metric): 10 for model in MODELS for metric in METRICS})
    require(Counter((r["model"], r["metric"]) for r in records) == expected_grid, "Expected 40 queries/model")
    planned_by_id = {q["query_id"]: q for q in planned}
    for r in records:
        q = planned_by_id[r["query_id"]]
        for field in ("model", "task", "metric", "H", "ordinal", "condition_id"):
            require(r[field] == q[field], f"Manifest mismatch: {field}")
        require(r["task"] == "Pendulum-v1" and r["H"] == 20, "Expected Pendulum H20")
        require(r["status"] in {"match", "mismatch", "ignored"}, "Failed/unscored record")
        require(r.get("api_request_made") is True and r.get("reused") is False
                and r.get("api_attempts") == 1, "Expected one new API attempt per query")
        request = r["request"]
        require(set(request) == {"model", "messages", "reasoning_effort"}
                and request["reasoning_effort"] == "medium" and request["model"] == r["model"],
                "Expected temperature omitted and reasoning_effort=medium")
        usage = r["raw_response"]["usage"]
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            number(usage[field], field, integer=True)
        require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Inconsistent usage")
        require(usage["prompt_tokens"] <= 272_000, "Long-context pricing is outside this estimator")
        for field in ("request_elapsed_seconds", "query_elapsed_seconds"):
            number(r[field], field)
    for model in MODELS:
        for metric in METRICS:
            rows = [r for r in records if (r["model"], r["metric"]) == (model, metric)]
            require({r["ordinal"] for r in rows} == set(range(10)), "Expected ordinals 0..9")
    conditions = summary["conditions"]
    require(len(conditions) == 12 and len({c["condition_id"] for c in conditions}) == 12,
            "Missing/duplicate condition times")
    require({c["condition_id"] for c in conditions} == {r["condition_id"] for r in records},
            "Condition time IDs differ from records")
    for c in conditions:
        rows = [r for r in records if r["condition_id"] == c["condition_id"]]
        require(c["status"] == "complete" and c["model"] == rows[0]["model"], "Incomplete/mismatched condition")
        number(c["condition_elapsed_seconds"], "condition_elapsed_seconds")
    number(summary["experiment_elapsed_seconds"], "experiment_elapsed_seconds")


def optional_detail(rows, section, field):
    """Retain missing detail as null; report how many responses supplied it."""
    values = [(r["raw_response"]["usage"].get(section) or {}).get(field) for r in rows]
    known = [number(v, field, integer=True) for v in values if v is not None]
    return {"tokens": sum(known) if len(known) == len(rows) else None, "known_queries": len(known)}


def build_estimate(summary, manifest, records, prices):
    """Aggregate measured usage/time once, then scale by 1, 7 and 8."""
    validate_run(summary, manifest, records)
    rows_by_scenario = {name: [] for name in SCENARIOS}
    detail = {}
    for model in MODELS:
        rows = [r for r in records if r["model"] == model]
        rate = prices["models"][model]
        input_rate = number(rate["input_usd_per_million"], "input price")
        output_rate = number(rate["output_usd_per_million"], "output price")
        inputs = sum(r["raw_response"]["usage"]["prompt_tokens"] for r in rows)
        outputs = sum(r["raw_response"]["usage"]["completion_tokens"] for r in rows)
        base = {
            "queries": len(rows), "input_tokens": inputs, "output_tokens": outputs,
            "total_tokens": inputs + outputs,
            "api_seconds": sum(r["request_elapsed_seconds"] for r in rows),
            "query_seconds": sum(r["query_elapsed_seconds"] for r in rows),
            "condition_seconds": sum(c["condition_elapsed_seconds"] for c in summary["conditions"] if c["model"] == model),
            "input_usd": inputs / 1_000_000 * input_rate,
            "output_usd": outputs / 1_000_000 * output_rate,
        }
        base["total_usd"] = base["input_usd"] + base["output_usd"]
        detail[model] = {
            "per_query_mean": {k: v / len(rows) for k, v in base.items() if k != "queries"},
            "cached_input": optional_detail(rows, "prompt_tokens_details", "cached_tokens"),
            "reasoning_output": optional_detail(rows, "completion_tokens_details", "reasoning_tokens"),
        }
        for name, factor in SCENARIOS.items():
            rows_by_scenario[name].append({"model": model, **{k: v * factor for k, v in base.items()}})
    scenarios = {}
    for name, rows in rows_by_scenario.items():
        total = {k: sum(r[k] for r in rows) for k in rows[0] if k != "model"}
        scenarios[name] = {"factor": SCENARIOS[name], "models": rows, "total": total,
                           "experiment_seconds": summary["experiment_elapsed_seconds"] * SCENARIOS[name]}
    return {"schema_version": 1, "pricing": prices, "scenarios": scenarios, "details": detail}


def markdown(report):
    """Japanese report: measured 05 and estimated 06 must not be confused."""
    lines = ["# Exp.05実測とExp.06の概算", "",
             "05の40件/modelを基準に、06の追加280件/modelは7倍、05＋06の320件/modelは8倍。",
             "05のtoken・時間のみ実測。金額は全て通常入力単価による概算で、確定請求額ではありません。", "",
             "## 計算の前提", "",
             "- input = raw_response.usage.prompt_tokens、output = completion_tokens（tiktoken再計測ではない）。",
             "- reasoning tokenはoutputの内訳であり、completion_tokensに別途加算しない。",
             "- input料金 = input tokens ÷ 1,000,000 × input単価。outputも同様。合計料金は両者の和。",
             "- cache割引・cache write料金・Batch/Flex/Priority・税・為替・契約割引は反映しない。上限額ではない。",
             "- 05はPendulum/H20のみ。06の別task・別H・出力長・混雑・rate limitによる違いは補正しない。",
             "- 06で再利用する40件/modelには追加API token・料金を計上しない。再生処理の時間は見積もらない。",
             "- retryなし、逐次実行を仮定。API・query・条件・実験全体時間は範囲が重なるため足さない。", ""]
    titles = {"exp05_observed": "05 実測（120件、40件/model）",
              "exp06_additional_estimate": "06 追加分の概算（840件、280件/model、×7）",
              "exp05_plus_06_estimate": "05＋06 累計の概算（960件、320件/model、×8）"}
    for name, scenario in report["scenarios"].items():
        lines += [f"## {titles[name]}", "",
                  "| モデル | 件数 | input tokens | output tokens | 合計tokens | input金額 USD | output金額 USD | 合計金額 USD |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        rows = [*scenario["models"], {"model": "全モデル合計", **scenario["total"]}]
        for r in rows:
            lines.append(f"| {r['model']} | {r['queries']:,} | {r['input_tokens']:,} | {r['output_tokens']:,} | "
                         f"{r['total_tokens']:,} | {r['input_usd']:.6f} | {r['output_usd']:.6f} | {r['total_usd']:.6f} |")
        lines += ["", "| モデル | API時間 秒 | query時間 秒 | 条件時間 秒 | 条件時間 分 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for r in rows:
            lines.append(f"| {r['model']} | {r['api_seconds']:.3f} | {r['query_seconds']:.3f} | "
                         f"{r['condition_seconds']:.3f} | {r['condition_seconds']/60:.2f} |")
        seconds = scenario["experiment_seconds"]
        lines += ["", f"全モデルの実験処理全体（summaryの実測×{scenario['factor']}）："
                  f"{seconds:.3f}秒 ≒ {seconds/60:.2f}分 ≒ {seconds/3600:.3f}時間。", ""]
    lines += ["## 05の1 query平均", "",
              "| モデル | input tokens | output tokens | API時間 秒 | query時間 秒 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for model, detail in report["details"].items():
        mean = detail["per_query_mean"]
        lines.append(f"| {model} | {mean['input_tokens']:.3f} | {mean['output_tokens']:.3f} | "
                     f"{mean['api_seconds']:.3f} | {mean['query_seconds']:.3f} |")
    lines += ["", "## 単価と参照元", "", f"価格確認日：{report['pricing']['checked_on']}。USD / 1M tokens。", "",
              "| モデル | 通常input | output | 公式資料 |", "| --- | ---: | ---: | --- |"]
    for model in MODELS:
        rate = report["pricing"]["models"][model]
        lines.append(f"| {model} | {rate['input_usd_per_million']} | {rate['output_usd_per_million']} | [公式価格]({rate['source_url']}) |")
    lines += ["", "## 時間の定義・再現情報", "",
              "API時間はrecords.jsonlのrequest_elapsed_seconds合計（通信を含む）。query時間はquery_elapsed_seconds合計。",
              "条件時間はsummary.jsonのconditions[].condition_elapsed_seconds合計（読込・準備・保存などを含む）。",
              "実験処理全体はsummary.jsonのexperiment_elapsed_seconds。起動・事前計画の時間は含まない。",
              "秒の丸めは表示時のみ。7倍の全体時間は固定処理も比例する簡易見積もりで、上限や信頼区間ではない。",
              "", "estimate.jsonには丸め前の値、単価設定、入力ファイルのSHA256、cache/reasoning内訳の取得件数も保存。",
              "内訳が欠損している場合はnullとし、0とみなさない。", "",
              f"入力ディレクトリ：`{report.get('source_directory', '')}`", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parent / "results")
    parser.add_argument("--prices", type=Path, default=HERE / "estimate_prices.json")
    args = parser.parse_args(argv)
    try:
        summary, manifest, records, hashes = load_completed(args.results)
        price_bytes = args.prices.read_bytes()
        report = build_estimate(summary, manifest, records, json.loads(price_bytes))
        report.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      source_directory=str(args.results.resolve()), source_sha256=hashes,
                      price_file=str(args.prices.resolve()), price_sha256=hashlib.sha256(price_bytes).hexdigest())
        rendered = markdown(report)
        # New directory per invocation; never write under results/ or overwrite a report.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = HERE / "cost_estimates" / f"{stamp}_{uuid.uuid4().hex[:8]}"
        destination.mkdir(parents=True, exist_ok=False)
        for name, content in {"RESULTS.md": rendered, "estimate.json": json.dumps(report, ensure_ascii=False, indent=2)}.items():
            with (destination / name).open("x", encoding="utf-8") as stream:
                stream.write(content + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Saved: {destination / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
