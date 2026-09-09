#!/usr/bin/env python3
"""List 20 official tasks and 5 candidates, optionally write the derived index."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_adapters.registry import dataset_ids
from tools.describe_dataset import describe
from tools.count_input_tokens import _write_csv

def inventory_rows():
    rows=[]
    for name in dataset_ids():
        d=describe(name); arrays=d.get('raw_arrays_example',{})
        r=dict(dataset_id=name,category=d['category'],format=','.join(d['format']),files=d['number_of_files'],sequences=d['number_of_sequences'],records=d['number_of_records'])
        for field in ('state','action','reward'):
            spec=arrays.get(field+'s',{})
            r[field+'_shape']=json.dumps(['T']+spec['shape'][1:] if spec else None)
            r[field+'_dtype']=spec.get('dtype')
            r[field+'_available']=d[field+'_available']
        r.update(sequential=d['sequential'],multi_agent=d['multi_agent'],prompt_mode=d['prompt_mode'],compatibility=d['mental_modeling_compatibility'],schema_doc=d['schema_doc'])
        rows.append(r)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    rows=inventory_rows()
    print(f"{'Dataset':38} {'Sequences':>9} {'Records':>12} State Action Reward Sequential Prompt")
    for r in rows: print(f"{r['dataset_id']:38} {str(r['sequences']):>9} {str(r['records']):>12} {str(r['state_available']):5} {str(r['action_available']):6} {str(r['reward_available']):6} {str(r['sequential']):10} {r['prompt_mode']}")
    print(f'{len(rows)} datasets (20 official + 5 candidates)')
    if a.output: _write_csv(a.output,rows)
if __name__=='__main__': main()
