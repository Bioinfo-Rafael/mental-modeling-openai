"""Build per-task accuracy tables from a read-only snapshot of API results."""
from collections import Counter
from datetime import datetime
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parents[1]


def main():
    source = HERE / 'results/api'
    manifest = json.loads((source / 'manifest.json').read_text())
    queries = manifest['queries']
    # Only consume complete JSONL lines if an execution is still appending results.
    content = (source / 'records.jsonl').read_bytes() if (source / 'records.jsonl').exists() else b''
    records = [json.loads(line) for line in content.splitlines(keepends=True) if line.endswith(b'\n')]
    by_id = {r['query_id']: r for r in records}
    assert len(by_id) == len(records), 'Duplicate result IDs'
    assert set(by_id) <= {q['query_id'] for q in queries}, 'Unknown result IDs'
    timestamp = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
    counts = Counter(r['status'] for r in records)
    notes = [f'集計日時（JST）: {timestamp}',
             f'保存済み回答: {len(records)}/{len(queries)}。正解 {counts["match"]}、不正解 {counts["mismatch"]}、解析不可 {counts["ignored"]}、空応答 {counts["empty_response"]}。',
             '正解率 = 正解数 / 保存済み回答数。各条件が3件揃うとn=3の正誤（1/0）の平均。解析不可・空応答は正解数に含めず、分母に含める。未回収の回答（APIエラー・未送信など）は分母に含めず、未完了と表示する。',
             'パターン1: 正解例S1、不正解例S3・S4。パターン2: 正解例S1・S2、不正解例S1・S2・S3。',
             '2kはfew-shot総例数（正解k件＋不正解k件）。予測対象はnext-action。解析不可の列は各条件の件数。']
    md = ['# Task別 few-shot正解率', '', *[n+'\n' for n in notes]]
    html = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>Few-shot正解率</title>',
            '<style>body{font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1150px;color:#172033}table{border-collapse:collapse;width:100%;margin:20px 0 36px}th,td{padding:10px 14px;border:1px solid #dbe2ea;text-align:center}th{background:#edf2f8}tr:nth-child(even){background:#f8fafc}p{line-height:1.7}h2{margin-top:36px}.scroll{overflow:auto}</style><h1>Task別 few-shot正解率</h1>',
            *['<p>'+escape(n)+'</p>' for n in notes]]
    headers=['H','パターン','総例数2k','Terra 正解率（正解/回答）','Luna 正解率（正解/回答）','Terra 解析不可','Luna 解析不可']
    all_tables = []
    for task in dict.fromkeys(q['task'] for q in queries):
        md += ['## '+task, '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        html += ['<h2>'+escape(task)+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>']
        conditions = sorted({(q['H'],q['score_pattern'],q['shots']) for q in queries if q['task']==task})
        for H, pattern, shots in conditions:
            cells=[str(H),pattern.replace('pattern',''),str(shots)]
            missing_parse=[]
            for model in ('terra','luna'):
                selected=[q for q in queries if (q['task'],q['H'],q['score_pattern'],q['shots'],q['model_alias'])==(task,H,pattern,shots,model)]
                saved=[by_id[q['query_id']] for q in selected if q['query_id'] in by_id]
                correct=sum(r['status']=='match' for r in saved)
                ignored=sum(r['status']=='ignored' for r in saved)
                value=f'{100*correct/len(saved):.1f}%（{correct}/{len(saved)}）' if saved else '—'
                if len(saved)!=len(selected):
                    value+=f' 未完了 {len(saved)}/{len(selected)}件'
                cells.append(value);missing_parse.append(str(ignored))
                all_tables.append(dict(task=task,H=H,score_pattern=pattern,shots=shots,model=model,
                                       correct=correct,received=len(saved),expected=len(selected),ignored=ignored,
                                       accuracy=correct/len(saved) if saved else None))
            cells+=missing_parse
            md.append('| '+' | '.join(cells)+' |')
            html.append('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'</tr>')
        md.append('');html.append('</tbody></table></div>')
    output=HERE/'results/analysis'
    output.mkdir(exist_ok=True)
    (output/'accuracy_tables.md').write_text('\n'.join(md)+'\n')
    (output/'accuracy_tables.html').write_text('\n'.join(html)+'</html>\n')
    (output/'accuracy_tables.json').write_text(json.dumps(dict(created_at_jst=timestamp,rows=all_tables),ensure_ascii=False,indent=2)+'\n')
    assert sum(x['received'] for x in all_tables)==len(records)
    assert sum(x['correct'] for x in all_tables)==counts['match']
    print(output/'accuracy_tables.html')
    print(f'{len(all_tables)} conditions; {len(records)}/{len(queries)} responses; {counts["match"]} correct')


if __name__=='__main__':
    main()
