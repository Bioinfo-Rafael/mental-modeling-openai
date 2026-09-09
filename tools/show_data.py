#!/usr/bin/env python3
"""Default: raw record. Model input only with explicit --processed."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_adapters.registry import get_adapter, EXTERNAL
from dataset_adapters.base import json_safe

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    ids=p.add_mutually_exclusive_group();ids.add_argument('--episode',type=int);ids.add_argument('--sequence',type=int);ids.add_argument('--row',type=int)
    p.add_argument('--index',type=int,default=0);p.add_argument('--processed',action='store_true');p.add_argument('--preprocessing-config',type=Path);p.add_argument('--history-size',type=int,default=1);a=p.parse_args()
    if a.preprocessing_config and not a.processed: p.error('--preprocessing-config requires --processed')
    adapter=get_adapter(a.dataset)
    if a.episode is not None and a.dataset in EXTERNAL: p.error('external datasets use --sequence or --row, not --episode')
    if a.row is not None and adapter.describe()['sequential']: p.error('sequential datasets use --episode or --sequence with --index')
    raw=adapter.global_row(a.row) if a.row is not None else adapter.record(a.episode if a.episode is not None else a.sequence or 0,a.index)
    result=raw
    if a.processed:
        from preprocessing.prompting import load_config, config_identity, build_external_query
        config=load_config(a.dataset,a.preprocessing_config)
        h=0 if config['prompt_mode']=='external_static' else a.history_size
        if a.dataset in EXTERNAL:
            query,current=build_external_query(adapter,adapter.sequence(raw['sequence_id']),raw['index'],config,h)
        else:
            from preprocessing.llmx_original import Episode, default_metric_question, build_prompt_queries
            source=adapter._sources[raw['sequence_id']]
            metric,question=default_metric_question(a.dataset,Episode.load(source.path))
            query=next((q for q in build_prompt_queries(source,metric=metric,question_name=question,history_size=h) if q.query_index==raw['index']),None)
            if query is None: p.error('index is not a valid original LLM-X query for this history size')
            current={'note':'Original upstream prompt below is authoritative; raw arrays are available without --processed.'}
        config_text,digest=config_identity(config,h)
        result=dict(representation='processed',dataset_id=a.dataset,source_file=raw['source_file'],episode_id=raw['episode_id'],sequence_id=raw['sequence_id'],row_id=raw.get('row_id'),query_index=raw['index'],preprocessing_name=config['name'],preprocessing_config=json.loads(config_text),preprocessing_config_sha256=digest,prompt_mode=config['prompt_mode'],processed=current,system_prompt=query.system_prompt,user_prompt=query.user_prompt)
    print(json.dumps(json_safe(result),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
