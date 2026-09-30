from importlib import import_module
import math

m=import_module('experiments.08_fewshot.analysis.plot_previous_comparison')


def test_existing_only_selection_scores_and_pools():
    records,_=m.load_records()
    old=[r for r in records if r['experiment']=='07']
    new=[r for r in records if r['experiment']=='08']
    assert len(old)==160 and len(new)==144
    assert all(r['reasoning_score']==5-r['old_reasoning_score'] for r in old)
    assert all(r['old_reasoning_score'] is None for r in new)
    assert sum(r['absolute_error'] is None for r in old)==6
    for task in m.TASKS:
        all_rows=m.summarize(records,task,'all')
        pooled=m.summarize(records,task,'pool_all')
        for row in pooled:
            pieces=[r for r in all_rows if (r['model'],r['experiment'],r['metric'])==(row['model'],row['experiment'],row['metric'])]
            assert math.isclose(row['mean'],sum(r['mean'] for r in pieces)/len(pieces))
            assert row['n']==(20 if row['experiment']=='07' else 36)
        for view in m.VIEWS:
            specs,_=m.group_specs(view)
            assert len(specs)=={'all':16,'pool_shots':8,'pool_pattern':10,'pool_all':4}[view]


def test_h_equal_weight_for_missing_error_values():
    rows=[]
    for h,values in [(5,[1]*10),(20,[3]*5+[None]*5)]:
        for value in values:
            rows.append(dict(task=m.TASKS[0],experiment='07',model='terra',H=h,
                             score_pattern='zero',shots=0,correct=0,absolute_error=value,reasoning_score=2))
    # Supply the other expected groups, preserving each group's required count.
    for model in ('luna',):
        rows += [{**r,'model':model} for r in list(rows)]
    for model in ('terra','luna'):
        for h in (5,20):
            for pattern in ('pattern1','pattern2'):
                for shots in (4,8,12):
                    rows += [dict(task=m.TASKS[0],experiment='08',model=model,H=h,score_pattern=pattern,
                                  shots=shots,correct=1,absolute_error=0,reasoning_score=3)]*3
    result=next(r for r in m.summarize(rows,m.TASKS[0],'pool_all') if r['experiment']=='07' and r['model']=='terra' and r['metric']=='mae')
    assert result['mean']==2 and result['valid_n']==15
