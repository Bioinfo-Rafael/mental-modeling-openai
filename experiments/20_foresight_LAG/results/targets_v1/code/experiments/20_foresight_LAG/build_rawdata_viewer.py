"""Build the bilingual viewer at the experiment root, without API calls.

Usage: .venv/bin/python experiments/20_foresight_LAG/build_rawdata_viewer.py
Reads saved reports and translations; leaves the experiment's frozen sources intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
AIRCRAFT = ("A0100", "A0200", "B0100", "B0200")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one template match: {old[:80]}")
    return text.replace(old, new, 1)


def build(run="foresight_short_v1"):
    directory = (HERE / "results" / run).resolve()
    if not directory.is_relative_to(HERE / "results"):
        raise ValueError("Run must be inside results/")
    report = directory / "report"
    plan = read_json(directory / "manifest.json")
    rows = read_json(report / "responses.json")
    summary = read_json(report / "summary.json")
    translations = read_json(report / "translations_ja.json")
    translated = {}
    for record in translations["records"]:
        key = (record["query_id"], record["aircraft_id"])
        if key in translated:
            raise ValueError(f"Duplicate translation: {key}")
        translated[key] = record
    used = set()
    for row in rows:
        row["narratives_ja"] = {}
        for aircraft, original in (row.get("narratives") or {}).items():
            key = (row["query_id"], aircraft)
            record = translated.get(key)
            if record is None:
                raise ValueError(f"Missing translation: {key}")
            actual_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
            if record["source_sha256"] != actual_hash or record["english"] != original:
                raise ValueError(f"Translation source differs: {key}")
            if record["model"] != row["model"] or record["sample_id"] != row["sample_id"]:
                raise ValueError(f"Translation metadata differs: {key}")
            if not record["japanese"].strip():
                raise ValueError(f"Empty translation: {key}")
            row["narratives_ja"][aircraft] = record["japanese"]
            used.add(key)
    if used != set(translated):
        raise ValueError("Translation records do not match saved responses")

    data = dict(plan_id=plan["plan_id"], rows=rows, summary=summary,
                samples=plan["samples"], prompt_spec=plan["prompt_spec"])
    template = (HERE / "templates/review.html").read_text(encoding="utf-8")
    report_url = quote(report.relative_to(HERE).as_posix(), safe="/")
    csv_name = f"{directory.name}_ja.csv"
    if not (report / csv_name).is_file():
        raise FileNotFoundError(report / csv_name)
    template = replace_once(
        template, '<a href="review_template.csv" download>人手評価用CSV</a>',
        f'<div><a href="{report_url}/{quote(csv_name)}" download>日本語訳付きCSV</a>'
        f'　<a href="{report_url}/review_template.csv" download>人手評価用CSV</a></div>')
    template = replace_once(template, "  const narrative=row.narratives?.[aircraft];",
        "  const original=row.narratives?.[aircraft];\n"
        "  const narrative=row.narratives_ja?.[aircraft]||original;\n"
        "  if(row.narratives_ja?.[aircraft])card.append(el('div','日本語訳','sub'));")
    template = replace_once(template, "  if(narrative){const words=",
        "  if(original&&row.narratives_ja?.[aircraft])card.append(details('英語原文',original));\n"
        "  if(narrative){const words=")
    template = replace_once(template, "`${words}語 / 上限55語", "`英語原文 ${words}語 / 上限55語")
    template = replace_once(template,
        "  if(row.narratives)card.append(details('全4機の回答',JSON.stringify(row.narratives,null,2)));",
        "  if(row.narratives)card.append(details('全4機の英語原文',JSON.stringify(row.narratives,null,2)));\n"
        "  if(row.narratives_ja)card.append(details('全4機の日本語訳',JSON.stringify(row.narratives_ja,null,2)));")
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = replace_once(template, "__DATA__", serialized)
    path = HERE / "view_rawdata.html"
    path.write_text(html, encoding="utf-8")
    print(f"Viewer: {path}\nResponses: {len(rows)}; translated aircraft narratives: {len(used)}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="foresight_short_v1")
    args = parser.parse_args()
    build(args.run)
