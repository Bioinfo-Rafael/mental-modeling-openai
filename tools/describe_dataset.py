#!/usr/bin/env python3
"""Describe raw schemas and separately report original prompt compatibility."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_adapters.registry import get_adapter, EXTERNAL
from dataset_adapters.base import json_safe

def describe(dataset_id):
    adapter = get_adapter(dataset_id)
    d = adapter.describe()
    if dataset_id not in EXTERNAL:
        from preprocessing.llmx_original import REGISTERED_TASKS
        from tools.render_schema_docs import STATE_MEANINGS, FETCH_MEANING, MINIGRID_MEANING
        d['dimension_meanings'] = STATE_MEANINGS.get(dataset_id, FETCH_MEANING if 'Fetch' in dataset_id and not dataset_id.startswith('MiniGrid') else MINIGRID_MEANING if dataset_id.startswith('MiniGrid') else 'unconfirmed')
        d['original_llmx_compatible'] = dataset_id in REGISTERED_TASKS
        d['prompt_mode'] = 'original_llmx' if d['original_llmx_compatible'] else 'unsupported'
        d['compatibility_reason'] = 'original upstream task and questions available' if d['original_llmx_compatible'] else 'task class/description absent from upstream registry; no safe version alias; raw readable; no fabricated benchmark'
    else:
        d['prompt_mode'] = 'unavailable' if not adapter.sequences() else 'external_generic' if d['sequential'] else 'external_static'
        d['raw_schema_groups'] = [] if not hasattr(adapter,'files') else [dict(schema_group=g, **adapter.inventory['groups'][g]) for g in sorted({r['schema_group'] for r in adapter.files})]
    d['reader_units'] = [dict(sequence_id=s.sequence_id, episode_id=s.episode_id, sequence_key=s.key, source_file=s.source_file, records=s.length, selector=s.selector, sequential=s.sequential) for s in adapter.sequences()]
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset',required=True); p.add_argument('--include-units',action='store_true'); a=p.parse_args()
    d=describe(a.dataset)
    if not a.include_units: d['reader_units']=d['reader_units'][:3]; d['units_note']='First 3 reader units; --include-units lists all. IDs start at 0.'
    print(json.dumps(json_safe(d),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
