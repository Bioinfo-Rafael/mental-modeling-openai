"""Write the fixed pattern pools and README tables from an actual dry-run snapshot."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
START='<!-- FIXED_POOL_REPORT_START -->'
END='<!-- FIXED_POOL_REPORT_END -->'
MODELS=('sol','terra','luna','3.5')


def report(snapshot):
    payload=json.loads((snapshot/'pattern_examples.json').read_text())
    source=HERE/'selected_examples_by_id.json'
    original=json.loads(source.read_text())['examples']
    groups=payload['groups']
    if len(groups)!=4 or any(Counter(e['example_type'] for e in g['examples'])!={'correct':6,'incorrect':6} for g in groups):
        raise ValueError('Report requires both tasks/patterns at 12 shots')
    ids={e['query_id'] for g in groups for e in g['examples']}
    payload.update(source_candidate_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   original_unique_count=len(original),retained_unique_count=len(ids),
                   pattern_memberships=sum(len(g['examples']) for g in groups),
                   removed_ids=[e['query_id'] for e in original if e['query_id'] not in ids])
    (HERE/'fixed_pattern_examples.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    long=[]
    lines=[START,'## 固定した例の内訳（2k=12）','',
           '各Task×パターンで正解6件・不正解6件。元の74件は削除せず、パターン別の使用候補を絞った。',
           f'4群で延べ{payload["pattern_memberships"]}件、パターン間の重複を除くと{len(ids)}件。元候補のうち{len(original)-len(ids)}件は今回使わない。',
           '以下は例の件数であり、144件の評価リクエスト数ではない。モデルは例を回答した元モデル。',
           'パターン1の正解はsol/terra/luna各2件、GPT-3.5は候補0件。それ以外はsol 1・terra 1・luna 2・GPT-3.5 2件。',
           'モデル配分とScore条件を満たす範囲でlast-actionを最小化した。パターン1の不正解にS4、パターン2の正解にS2を少なくとも1件含める。',
           '結果としてパターン2の正解は全件S2。Score 1も許容する条件は維持し、next-action優先によってこの構成になった。','',
           '| Task | パターン | 正解 | 不正解 | next-action | last-action | 合計 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        es=g['examples'];c=Counter(e['example_type'] for e in es);m=Counter(e['metric'] for e in es)
        lines.append(f"| {g['task']} | {g['score_pattern']} | {c['correct']} | {c['incorrect']} | {m['next-action']} | {m['last-action']} | {len(es)} |")
    for g in groups:
        lines+=['',f"### {g['task']} / {g['score_pattern']}",'',
                '| 区分 | 元モデル | S1 | S2 | S3 | S4 | next-action | last-action | 合計 |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for kind in ('correct','incorrect'):
            for model in MODELS:
                es=[e for e in g['examples'] if e['example_type']==kind and e['model']==model]
                scores=Counter(e['reasoning_score'] for e in es);metrics=Counter(e['metric'] for e in es)
                values=['正解' if kind=='correct' else '不正解',model,*[scores[s] for s in range(1,5)],metrics['next-action'],metrics['last-action'],len(es)]
                lines.append('| '+' | '.join(map(str,values))+' |')
                for score in range(1,5):
                    for metric in ('next-action','last-action'):
                        long.append(dict(task=g['task'],pattern=g['score_pattern'],example_type=kind,model=model,
                                         score=score,metric=metric,count=sum(e['reasoning_score']==score and e['metric']==metric for e in es)))
    lines+=['','パターン1の正解S1はnext-action候補がないため、両Taskとも6件全てlast-action。Pendulumのパターン2正解はGPT-3.5の2件だけlast-actionで、それ以外はnext-action。',
            '', '4例・8例はこの固定集合から正解／不正解を各2件・各4件取る。元モデルをsol→terra→luna→3.5の順で巡回するので、先頭4件（片側）では可能なモデルを1件ずつ含める。',
            '', '固定した全例と順序: [fixed_pattern_examples.json](data_prep/fixed_pattern_examples.json)。Score×next/lastまでの完全なクロス集計（0件セルを含む）: [fixed_pattern_counts.csv](data_prep/fixed_pattern_counts.csv)。',
            '', '[更新済みプレビュー](results/dry_run/preview.html)。元の74件JSONは保持し、実行時は同じ規則で固定集合を再現する。',END]
    with (HERE/'fixed_pattern_counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(long[0]));writer.writeheader();writer.writerows(long)
    readme=HERE.parent/'README.md';text=readme.read_text()
    block='\n'.join(lines)+'\n'
    if START in text:
        text=text[:text.index(START)]+block+text[text.index(END)+len(END):]
    else:
        text+='\n'+block
    readme.write_text(text)
    print(f'Fixed: {len(ids)} unique examples; {len(groups)} groups; {len(long)} cross-tab rows')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    report(parser.parse_args().snapshot.resolve())
