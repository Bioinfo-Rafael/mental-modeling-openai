"""Offline four-model comparison: shared scoring, explicit missingness, immutable inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from importlib import import_module
import json
from pathlib import Path
import sys
import time

from experiments import common
from experiments.split_execution import merge_results
from .joint_data import ROOT, BOUNDS, WIDTHS, TASKS, METRICS, HISTORIES, load_and_parse, select_subsets
from .joint_aggregate import aggregate_all
from .comparison_plots import MODEL_ORDER, MODEL_COLORS, MODEL_LABELS, CM_GROUPS, make_comparison_figures
from .runner import (digest, flatten_records, parse_summary, protected_files, validate_tables,
                     validate_saved_outputs, write_csv, write_json)

BASE = ROOT / "experiments/06_new_models_n10/analysis"


def selected_records(records, subsets):
    ids = {q["query_id"] for group in subsets["conditions"].values() for q in group["10"]}
    result = [r for r in records if r["query_id"] in ids]
    if len(result) != 320 or len(ids) != 320:
        raise ValueError("Each model must contribute exactly 32 conditions x 10 selected queries")
    return result


def condition_coverage(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["condition_id"]].append(r)
    output = []
    for cid, rows in sorted(groups.items()):
        first = rows[0]
        failed = sum(r.get("api_error", False) for r in rows)
        output.append({**{k: first[k] for k in ("model_alias", "model", "task", "metric", "H")},
                       "condition_id": cid, "selected_n": len(rows), "api_error_n": failed,
                       "response_n": len(rows) - failed,
                       "all_required_valid_n": sum(r["all_required"] for r in rows)})
    return output


def results_markdown(tables, coverage, figures, repetitions=1000):
    lines = ["# 4モデル比較結果（N=10）", "",
             f"PNG {len(figures)}枚。SVG・新規API送信なし。計算式は03_1の共通評価を再利用。", "",
             "## 選択件数と欠損", "",
             "| モデル | 選択query | API応答あり | APIエラー | 全required component有効 |",
             "|---|---:|---:|---:|---:|"]
    for alias in MODEL_ORDER:
        group = [r for r in coverage if r["model_alias"] == alias]
        totals = [sum(r[k] for r in group) for k in ("selected_n", "response_n", "api_error_n", "all_required_valid_n")]
        lines.append(f"| {MODEL_LABELS[alias]} | " + " | ".join(map(str, totals)) + " |")
    lines += ["", "GPT-3.5は03_1の各条件30件から既存seed=42で10件を抽出。新3モデルは05+06の各条件10件すべて（失敗queryも選択数に残す）。",
              "GPT-3.5と新モデルは同一query集合とは限らず、paired比較ではない。選択IDは`query_subsets.json`、条件別件数は`tables/condition_coverage.csv`。", "",
              "APIエラーとparse失敗は指標ごとの有効分母から除き、0点・0 token・架空の応答にしない。補充抽出・再送もしない。",
              "API失敗にかかった通信時間は`api_failures.json`に残すが、成功応答の処理時間比較には混ぜない。",
              "parse_success_rate図はAPI応答取得を含む全required componentの成功率。component別の成否もCSVへ保存。", "",
              "## Action Matching Rate / State-change Accuracy", "",
              "値は%（括弧内は有効query数/選択10件）。Stateの正解率の分母は有効query×次元数。", "",
              "| Task | Metric | H | " + " | ".join(MODEL_LABELS[a] for a in MODEL_ORDER) + " |",
              "|---|---|---:|---:|---:|---:|---:|"]
    values = {(r["task"], r["metric"], r["H"], r["model_alias"]): r for r in tables["accuracy_metrics"] if r["metric_name"] == "accuracy"}
    for task in TASKS:
        for metric in METRICS:
            for h in HISTORIES:
                cells = []
                for alias in MODEL_ORDER:
                    row = values[task, metric, h, alias]
                    cells.append(("N/A" if row["value"] is None else f"{row['value']:.2f}") + f" ({row['valid_n']}/10)")
                lines.append(f"| {task} | {metric} | {h} | " + " | ".join(cells) + " |")
    lines += ["", "## 読み取り上の注意", "",
              "- 全図でGPT-3.5=青、Terra=緑、Luna=橙、Sol=紫。混同行列もモデルに対応した色相。",
              f"- NRMSE帯はpaired-query bootstrap {repetitions:,}回の標準偏差。95%信頼区間ではない。",
              "- 有効1件ではbootstrap SD=0となり得る。安定性・優位性の証拠ではない。Pearsonが定義不能ならN/A。",
              "- token/time帯は有効query間のsample SD。05再利用分は元の実測時間を保持。追加課金の合計ではない。",
              "- 各モデル・H・componentで有効集合が異なり、重なる履歴windowも独立とは限らない。差を統計的優位やHの因果効果と解釈しない。",
              "- 数値とstd、valid_n、na_reasonは`tables/`、全図のパスは`figure_index.json`に保存。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "comparison_4models")
    parser.add_argument("--seed", type=int, default=42, choices=(42,), help="Keep the existing Exp.03_1 seed-42 selection")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--allow-api-failures", action="store_true",
                        help="Explicitly retain API failures as missing predictions; use valid-only estimates")
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    if not output.is_relative_to(BASE) or output.resolve() != output or output == BASE or output.exists():
        parser.error("Choose a NEW output directory inside Exp.06/analysis; existing outputs are never overwritten")
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    started = time.perf_counter()
    spec35 = import_module("experiments.03_1_gpt35_history_n30_joint.run").EXPERIMENT
    spec06 = import_module("experiments.06_new_models_n10.run").EXPERIMENT
    source35 = common.latest_results(ROOT / "experiments" / spec35.name / "results")
    protected = {}
    for source in (source35, ROOT / "experiments/05_new_models_single/results",
                   ROOT / "experiments/06_new_models_n10/results/batches"):
        protected.update(protected_files(source))
    source06 = merge_results(spec06)  # No requests; preserves original batches/05.
    source35, manifest35, rows35, hashes35 = load_and_parse(source35, spec35)
    source06, manifest06, rows06, hashes06 = load_and_parse(source06, spec06, allow_api_failures=args.allow_api_failures)
    cached = common.check_inputs(spec06, manifest06)
    reused = {r["query_id"]: r for r in common.read_jsonl(source06 / "records.jsonl") if r.get("reused")}
    if set(reused) != set(cached) or any(r["api_request_made"] or r["raw_response"] != cached[qid]["raw_response"] for qid, r in reused.items()):
        raise ValueError("Exp.05 reuse changed or was counted twice")

    tables, subsets_by_model, selected, validations = defaultdict(list), {}, [], {}
    for alias in MODEL_ORDER:
        rows = rows35 if alias == "3.5" else [r for r in rows06 if r["model_alias"] == alias]
        subsets = select_subsets(rows, args.seed, sizes=(10,), expected_n=30 if alias == "3.5" else 10)
        model_tables = aggregate_all(rows, subsets, args.bootstrap, args.seed, sizes=(10,))
        validations[alias] = validate_tables(rows, subsets, model_tables)
        chosen = selected_records(rows, subsets)
        selected.extend(chosen)
        subsets_by_model[alias] = subsets
        for key, values in model_tables.items():
            tables[key].extend(values)
        tables["parse_summary"].extend(parse_summary(chosen))
        print(f"{alias}: selected {len(chosen)} queries; API errors {sum(r.get('api_error', False) for r in chosen)}", flush=True)
    # Verify that the default GPT-3.5 selection reproduces the existing notebook-era analysis.
    old_subsets = ROOT / "experiments/03_1_gpt35_history_n30_joint/analysis/query_subsets.json"
    if args.seed == 42 and old_subsets.exists():
        old = common.read_json(old_subsets)
        if any(v["10"] != old["conditions"][cid]["10"] for cid, v in subsets_by_model["3.5"]["conditions"].items()):
            raise ValueError("GPT-3.5 N10 subset differs from the existing Exp.03_1 analysis")
    tables["parsed_records"] = flatten_records(selected)
    tables["condition_coverage"] = condition_coverage(selected)
    if len(selected) != 1280 or len({r["query_id"] for r in selected}) != 1280:
        raise ValueError("Four-model selection must contain 1280 unique query IDs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    for name, rows in tables.items():
        write_csv(output / "tables" / (name + ".csv"), rows)
    write_json(output / "query_subsets.json", subsets_by_model)
    failures = [r for r in common.read_jsonl(source06 / "records.jsonl") if r["status"] == "failed"]
    write_json(output / "api_failures.json", failures)
    print("Drawing 65 PNG figures (no SVG)...", flush=True)
    figures = make_comparison_figures(tables, output)
    validate_saved_outputs(tables, figures, output)
    if list(output.rglob("*.svg")):
        raise RuntimeError("Unexpected SVG output")
    for relative, expected in protected.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Protected input changed: {relative}")
    episode_hashes = {**hashes35, **hashes06}
    for relative, expected in episode_hashes.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Raw episode changed: {relative}")
    write_json(output / "figure_index.json", figures)
    write_json(output / "protected_sha256.json", protected)
    write_json(output / "validation.json", {"models": validations, "selected_queries": len(selected),
               "api_failed_queries": len(failures), "raw_data_unchanged": True, "protected_inputs_unchanged": True,
               "png_count": len(figures), "svg_count": 0, "image_readback_checks": len(figures),
               "csv_readback_checks": len(tables), "api_calls": 0})
    metadata = {"sources": {"3.5": str(source35.relative_to(ROOT)), "modern_models": str(source06.relative_to(ROOT))},
                "source_records_sha256": {str(p.relative_to(ROOT)): digest(p / "records.jsonl") for p in (source35, source06)},
                "model_colors": MODEL_COLORS, "model_order": MODEL_ORDER, "seed": args.seed,
                "bootstrap_repetitions": args.bootstrap, "subset_n": 10, "api_calls": 0,
                "missing_policy": "component-wise valid-only; API errors retained as missing, no replacement",
                "api_failures_explicitly_allowed": args.allow_api_failures,
                "resource_policy": "successful responses only; failed latency retained separately; original Exp.05 latency preserved",
                "gpt35_selection": "existing per-condition seed-42 N10 selection from 30; not necessarily paired with new models",
                "normalization_bounds": BOUNDS, "normalization_widths": WIDTHS,
                "raw_episode_sha256": episode_hashes, "cm_groups": CM_GROUPS,
                "analysis_code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "analysis_metadata.json", metadata)
    (output / "RESULTS.md").write_text(results_markdown(tables, tables["condition_coverage"], figures, args.bootstrap), encoding="utf-8")
    (output / "README.md").write_text((BASE / "COMPARISON_README.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved: {output}\nPNG: {len(figures)}; SVG: 0; API calls: 0; protected files unchanged", flush=True)


if __name__ == "__main__":
    main()
