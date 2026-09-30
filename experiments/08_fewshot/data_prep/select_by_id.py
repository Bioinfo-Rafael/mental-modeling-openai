"""Select action-ID examples per task/model; preserve original scores and report shortages."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
TASKS=('Pendulum-v1','MountainCar-v0')
MODELS=('sol','terra','luna','3.5')
RULES=(('correct',1,(1,)),('correct',2,(2,3,4)),
       ('incorrect',3,(3,)),('incorrect',4,(4,)))


def action_match(e):
    if e['metric'] not in ('next-action','last-action') or not e['response_available']:
        return None
    key='action_bin' if e['task']=='Pendulum-v1' else 'action'
    c=e['comparison'][key]
    if not c['parsed']:
        return None
    prediction,truth=c['prediction'],c['ground_truth']
    if (not isinstance(prediction,list) or not isinstance(truth,list)
            or len(prediction)!=1 or len(truth)!=1):
        raise ValueError(f"Invalid action ID shape: {e['query_id']}")
    matches=prediction==truth
    if c['exact_match'] is not matches:
        raise ValueError(f"Stored comparison disagrees with ID equality: {e['query_id']}")
    return matches


def preference(e):
    # Score tier is selected first. Within it, next-action always precedes last-action.
    return (e['metric']!='next-action',e['H'],e['episode_path'],e['query_index'],e['ordinal'],e['query_id'])


def select(rows,n=3):
    if type(n) is not int or n<1:
        raise ValueError('n must be a positive integer')
    if len({e['query_id'] for e in rows})!=len(rows):
        raise ValueError('Duplicate source query IDs')
    selected,audit,summary=[],[],[]
    used=set()
    for task in TASKS:
        for model in MODELS:
            group=[e for e in rows if e['task']==task and e['model']==model]
            eligible={'correct':[],'incorrect':[]}
            for e in group:
                match=action_match(e)
                if match is not None:
                    eligible['correct' if match else 'incorrect'].append(e)
            for kind,desired,tiers in RULES:
                chosen=[]
                for score in tiers:
                    candidates=sorted((e for e in eligible[kind]
                                       if e['reasoning_score']==score and e['query_id'] not in used),key=preference)
                    chosen.extend(candidates[:n-len(chosen)])
                    used.update(e['query_id'] for e in chosen)
                    if len(chosen)==n:
                        break
                for e in chosen:
                    fallback=e['reasoning_score']!=desired
                    note=(f"行動ID{'一致' if kind=='correct' else '不一致'}で自動選択。"
                          f"希望Score {desired}、実際Score {e['reasoning_score']}。"
                          + ('Score 2不足のため補完。' if fallback else '')
                          + '同一Task・モデル内で各Scoreのnext-actionを優先。')
                    selected.append({**e,'example_type':kind,'note':note,'selection_order':len(selected)})
                    audit.append(dict(query_id=e['query_id'],task=task,model=model,example_type=kind,
                                      requested_score=desired,actual_score=e['reasoning_score'],fallback=fallback,
                                      metric=e['metric'],H=e['H'],ordinal=e['ordinal'],
                                      id_component='action_bin' if task=='Pendulum-v1' else 'action'))
                primary=[e for e in eligible[kind] if e['reasoning_score']==desired]
                summary.append(dict(task=task,model=model,example_type=kind,requested_score=desired,
                                    requested_count=n,selected_count=len(chosen),missing=n-len(chosen),
                                    available_next=sum(e['metric']=='next-action' for e in primary),
                                    available_last=sum(e['metric']=='last-action' for e in primary),
                                    selected_next=sum(e['metric']=='next-action' for e in chosen),
                                    selected_last=sum(e['metric']=='last-action' for e in chosen),
                                    fallback_count=sum(e['reasoning_score']!=desired for e in chosen),
                                    actual_score_counts=dict(sorted(Counter(e['reasoning_score'] for e in chosen).items()))))
    return selected,audit,summary


def generate(source,output):
    raw=source.read_bytes();source_hash=hashlib.sha256(raw).hexdigest()
    rows=json.loads(raw.decode('utf-8-sig'))
    selected,audit,summary=select(rows)
    now=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    paths=[output,output.with_suffix('.summary.json'),output.with_suffix('.summary.csv'),output.with_suffix('.md')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Output already exists. Specify a different --output filename.')
    requested=len(TASKS)*len(MODELS)*len(RULES)*3
    report=dict(created_at=now,source=str(source.resolve()),source_sha256=source_hash,
                requested=requested,selected=len(selected),missing=requested-len(selected),
                fallback_count=sum(row['fallback'] for row in audit),
                actual_score_counts=dict(sorted(Counter(e['reasoning_score'] for e in selected).items())),
                metric_counts=dict(Counter(e['metric'] for e in selected)),
                missing_reasoning=sum(not e['reasoning_available'] for e in selected),
                groups=summary,selection_audit=audit)
    payload=dict(schema_version=1,kind='candidate_pool',created_at=now,
                 selection_policy=dict(method='action ID equality',per_task_model_bucket=3,
                                       source=str(source.resolve()),source_sha256=source_hash,
                                       rules=[dict(example_type=k,requested_score=s,allowed_fallback_scores=list(t[1:])) for k,s,t in RULES],
                                       preference='score tier, next-action before last-action, H, episode, query index, ordinal, ID',
                                       no_other_fallback=True,selection_audit=audit),examples=selected)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:
        raise ValueError('Source changed while selecting')
    output.parent.mkdir(parents=True,exist_ok=True)
    for p,value in [(paths[0],payload),(paths[1],report)]:
        with p.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with paths[2].open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader()
        w.writerows({**r,'actual_score_counts':json.dumps(r['actual_score_counts'])} for r in summary)
    text=['# 行動ID一致／不一致による例の選択','',
          f'目標 {requested}件 / 選択 {len(selected)}件 / 不足 {requested-len(selected)}件。',
          f'Score 2不足の補完: {report["fallback_count"]}件。元Scoreは変更していません。','',
          '各Task×モデルで、正解Score 1・正解Score 2・不正解Score 3・不正解Score 4を各3件まで選択。',
          'Pendulumはbin ID、MountainCarは行動IDを比較。ID抽出失敗・API失敗・state予測は除外。',
          '各Score内でnext-actionを優先し、不足ならlast-action。正解Score 2だけは両予測対象でも不足した後にScore 3→4で補完。',
          'Hは制限せず、同順位はH・episode・index順。異なるH/modelの回答はquery_idが別なら別例として数えます。',
          '候補不足を他モデル・Task・未許可Scoreで埋めません。希望数はTaskごとに各区分12件（4モデル×3件）、計48件です。','',
          '| Task | モデル | 正解 S1 | 正解 S2枠 | 不正解 S3 | 不正解 S4 | 合計 |',
          '|---|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for model in MODELS:
            group=[g for g in summary if g['task']==task and g['model']==model]
            nums=[g['selected_count'] for g in group]
            text.append('| '+' | '.join([task,model,*map(str,nums),str(sum(nums))])+' |')
    text+=['','## 不足','']
    for g in summary:
        if g['missing']:
            text.append(f"- {g['task']} / {g['model']} / {g['example_type']} / 希望Score {g['requested_score']}: {g['selected_count']}/3件（不足{g['missing']}）")
    text+=['',f"元Reasoningが空欄の選択例: {report['missing_reasoning']}件。既存Scoreを保持しています。",'',
           f'07への読み込み／08のdry-run入力: `{output.name}`。',
           '選択JSONは既存exportと同じcandidate_pool形式です。summary JSON/CSVは監査用で、例の入力として自動検出されません。','']
    with paths[3].open('x',encoding='utf-8') as f:f.write('\n'.join(text))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'examples.json')
    p.add_argument('--output',type=Path,default=HERE/'selected_examples_by_id.json')
    args=p.parse_args()
    report=generate(args.source,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in {'groups','selection_audit'}},ensure_ascii=False,indent=2))
    print(args.output.resolve())


if __name__=='__main__':
    main()
