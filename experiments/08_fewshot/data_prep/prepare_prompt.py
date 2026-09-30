"""Filter a selected-example JSON and generate Question/Answer/Label few-shot blocks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREAMBLE = 'The following are past Question / Answer pairs and evaluation Labels for those Answers.\n\nUse Correct examples as references for your answer.\nUse Incorrect examples to avoid mistakes, and do not imitate their answers or reasoning.\nLabels are evaluation information and must not be included in your output.\n\nAnswer only the final New Question, and follow the output format specified in that question.'


def rank(seed, query_id):
    """Same FNV-1a ordering as the offline viewer (no global RNG state)."""
    h = 2166136261
    for c in f'{seed}:{query_id}'.encode():
        h = ((h ^ c) * 16777619) & 0xffffffff
    return h


def select(examples, *, task=None, metric=None, model=None, H=None, scores=(1,2,3,4),
           correct=0, incorrect=0, sampling='first', seed=42, order='selection'):
    if any(type(n) is not int or n < 0 for n in (correct, incorrect, seed)) or seed > 0xffffffff:
        raise ValueError('Counts must be nonnegative integers; seed must be uint32')
    if sampling not in {'first','random'} or order not in {'selection','correct-first','alternate','shuffle'}:
        raise ValueError('Invalid sampling/order')
    if any(s not in (1,2,3,4) for s in scores):
        raise ValueError('Invalid reasoning score')
    if len({e['query_id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate query_id')
    for e in examples:
        if e.get('example_type') not in {'correct','incorrect'}:
            raise ValueError('Use a selected-example export; examples.json is an unlabelled candidate database')
        if not isinstance(e.get('question'), str) or not isinstance(e.get('answer'), str) or not e['answer']:
            raise ValueError('Missing question/answer')
        if 'system_prompt' in e:
            raise ValueError('system_prompt must not be included in reusable examples')
    # File order is authoritative; exported selection_order describes its original provenance.
    candidates=[e for e in examples if (not task or e['task']==task) and (not metric or e['metric']==metric)
                and (not model or e['model']==model) and (not H or e['H']==H) and e['reasoning_score'] in scores]
    groups=[]
    for kind,n in [('correct',correct),('incorrect',incorrect)]:
        group=[e for e in candidates if e['example_type']==kind]
        if len(group)<n:
            raise ValueError(f'{kind}: requested {n}, available {len(group)}')
        if sampling=='random':
            group=sorted(group,key=lambda e:(rank(seed,e['query_id']),e['query_id']))
        groups.append(group[:n])
    chosen=groups[0]+groups[1]
    if not chosen:
        raise ValueError('Choose at least one example')
    if order=='selection':
        positions={e['query_id']:i for i,e in enumerate(examples)}
        chosen.sort(key=lambda e:positions[e['query_id']])
    elif order=='alternate':
        chosen=[g[i] for i in range(max(map(len,groups))) for g in groups if i<len(g)]
    elif order=='shuffle':
        chosen.sort(key=lambda e:(rank((seed+1)&0xffffffff,e['query_id']),e['query_id']))
    return chosen


def label(example):
    # Translate display labels only; preserve the source records and numerical values.
    def english(value):
        if isinstance(value, dict):
            return {key: english(item) for key, item in value.items()}
        if isinstance(value, list):
            return [english(item) for item in value]
        if isinstance(value, str):
            return {'行動ID': 'Action ID', 'トルク': 'Torque'}.get(value, value)
        return value
    return english({key:example[key] for key in ('example_type','reasoning_score','reasoning_available','ground_truth','comparison')})


def prompt_text(examples):
    return PREAMBLE+'\n\n'+'\n\n---\n\n'.join(
        f"Question:\n{e['question']}\n\nAnswer:\n{e['answer']}\n\nLabel:\n{json.dumps(label(e),ensure_ascii=False,indent=2)}"
        for e in examples)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='selected_examples_*.json exported by the viewer')
    p.add_argument('--task',choices=['MountainCar-v0','Pendulum-v1'])
    p.add_argument('--metric',choices=['next-action','last-action','next-state','last-state'])
    p.add_argument('--model',choices=['3.5','sol','terra','luna'])
    p.add_argument('--history',type=int,choices=[5,10,20,30])
    p.add_argument('--scores',type=int,nargs='+',default=[1,2,3,4],choices=[1,2,3,4])
    p.add_argument('--correct',type=int,default=0)
    p.add_argument('--incorrect',type=int,default=0)
    p.add_argument('--sampling',choices=['first','random'],default='first')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--order',choices=['selection','correct-first','alternate','shuffle'],default='selection')
    p.add_argument('--output',type=Path,default=HERE/'fewshot_prompt.txt')
    args=p.parse_args()
    data=json.loads(args.input.read_text())
    examples=data if isinstance(data,list) else data['examples']
    options=dict(task=args.task,metric=args.metric,model=args.model,H=args.history,scores=args.scores,
                 correct=args.correct,incorrect=args.incorrect,sampling=args.sampling,seed=args.seed,order=args.order)
    try:
        selected=select(examples,**options)
    except (ValueError,KeyError) as exc:
        p.error(str(exc))
    output=args.output.resolve()
    if output.suffix!='.txt':
        p.error('--output must end in .txt')
    companion=output.with_suffix('.json')
    if output.exists() or companion.exists():
        p.error('Output already exists; choose a different --output name')
    output.parent.mkdir(parents=True,exist_ok=True)
    companion.write_text(json.dumps(dict(schema_version=1,kind='fewshot_subset',options=options,examples=selected),ensure_ascii=False,indent=2)+'\n')
    output.write_text(prompt_text(selected))
    print(f'{len(selected)} examples: {output}\n{companion}')


if __name__=='__main__':
    main()
