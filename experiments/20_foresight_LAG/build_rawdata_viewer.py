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


def load_run(run):
    directory = (HERE / "results" / run).resolve()
    if not directory.is_relative_to(HERE / "results"):
        raise ValueError("Run must be inside results/")
    report = directory / "report"
    plan = read_json(directory / "manifest.json")
    rows = read_json(report / "responses.json")
    summary = read_json(report / "summary.json")
    translations = read_json(report / "translations_ja.json") if (report / "translations_ja.json").exists() else {"records": []}
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
                continue
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

    return dict(run=run, plan_id=plan["plan_id"], rows=rows, summary=summary,
                samples=plan["samples"], prompt_spec=plan["prompt_spec"],
                report_url=quote(report.relative_to(HERE).as_posix(), safe="/"),
                csv_name=f"{directory.name}_ja.csv" if (report / f"{directory.name}_ja.csv").exists() else None)


def build(runs=None):
    if isinstance(runs, str):
        runs = [runs]
    if runs is None:
        runs = [p.parent.parent.name for p in sorted((HERE / "results").glob("*/report/responses.json"))
                if any(r.get("raw_response") is not None for r in read_json(p))]
    datasets = [load_run(run) for run in runs]
    if not datasets:
        raise ValueError("No saved responses found")
    template = (HERE / "templates/review.html").read_text(encoding="utf-8")
    template = replace_once(template, '<a href="review_template.csv" download>人手評価用CSV</a>',
        '<div><a id="translation-csv" download>日本語訳付きCSV</a>　<a id="review-csv" download>人手評価用CSV</a></div>')
    template = replace_once(template, '<div class="filters">',
        '<div class="filters"><label>実験 / Batch <select id="run" aria-label="実験 / Batch"></select></label>')
    template = template.replace('n=−1・n=1の予測と、61時点を観測した参照回答を同じ対象区間で比較。',
        '選択した実験の予測と、61時点を観測した参照回答を同じ対象区間で比較。')
    template = template.replace('今回のプロンプトとv2からの全差分', '選択したプロンプトと元の版からの全差分').replace('3条件でsystem/userテンプレートは共通。', '各入力条件でsystem/userテンプレートは共通。')
    template = replace_once(template, "const data=JSON.parse(document.getElementById('dataset').textContent);",
        "const datasets=JSON.parse(document.getElementById('dataset').textContent);\nlet data,events,models,stateLabels;")
    template = replace_once(template, "const events=[...new Map", "function selectRun(){\n const previousEvent=$('event').value,previousModel=$('model').value;\n data=datasets.find(d=>d.run===$('run').value);\n for(const id of ['event','model','prompts','excluded'])$(id).replaceChildren();\n $('review-csv').href=data.report_url+'/review_template.csv';\n $('translation-csv').hidden=!data.csv_name;\n $('translation-csv').href=data.csv_name?data.report_url+'/'+encodeURIComponent(data.csv_name):'';\n events=[...new Map")
    template = replace_once(template, 'const models=[...new Set', 'models=[...new Set')
    template = replace_once(template, 'const stateLabels=', 'stateLabels=')
    template = template.replace('× 3入力条件 ×', '× ${count.input_conditions}入力条件 ×').replace('全3条件から除外。', '選択した全条件から除外。')
    template = replace_once(template, 'function render(){', " if(events.some(e=>e.id===previousEvent))$('event').value=previousEvent;\n if(models.includes(previousModel))$('model').value=previousModel;\n render();\n}\nfunction render(){")
    template = replace_once(template, " for(const n of [-1,1,30]){", " const offsets=[...new Set(data.rows.map(r=>r.offset))].sort((a,b)=>a-b);\n $('comparison').style.gridTemplateColumns=window.innerWidth>950?`repeat(${offsets.length},minmax(0,1fr))`:'1fr';\n for(const n of offsets){")
    template = replace_once(template, "\nrender();\n</script>", "\nfor(const d of datasets){const o=el('option',d.run);o.value=d.run;$('run').append(o); }\n$('run').value=datasets[datasets.length-1].run;\n$('run').addEventListener('change',selectRun);\nwindow.addEventListener('resize',render);\nselectRun();\n</script>")
    template = replace_once(template, "  const narrative=row.narratives?.[aircraft];",
        "  const original=row.narratives?.[aircraft];\n"
        "  const narrative=row.narratives_ja?.[aircraft]||original;\n"
        "  if(row.narratives_ja?.[aircraft])card.append(el('div','日本語訳','sub'));")
    template = replace_once(template, "  if(narrative){const words=",
        "  if(original&&row.narratives_ja?.[aircraft])card.append(details('英語原文',original));\n"
        "  if(narrative){const words=")
    template = replace_once(template, "`${words}語 / 上限", "`英語原文 ${words}語 / 上限")
    template = replace_once(template,
        "  if(row.narratives)card.append(details('全4機の回答',JSON.stringify(row.narratives,null,2)));",
        "  if(row.narratives)card.append(details('全4機の英語原文',JSON.stringify(row.narratives,null,2)));\n"
        "  if(row.narratives_ja)card.append(details('全4機の日本語訳',JSON.stringify(row.narratives_ja,null,2)));")
    serialized = json.dumps(datasets, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = replace_once(template, "__DATA__", serialized)
    path = HERE / "view_rawdata.html"
    path.write_text(html, encoding="utf-8")
    print(f"Viewer: {path}")
    for data in datasets:
        print(data["run"], len(data["rows"]), data["summary"]["statuses"])
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", default=None)
    args = parser.parse_args()
    build(args.run)
