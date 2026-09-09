"""Local processed-prompt inventory; no API client is created."""
import argparse
import bisect
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from dataset_adapters.base import ROOT
from dataset_adapters.registry import dataset_ids, get_adapter, EXTERNAL
from preprocessing.llmx_original import Episode, REGISTERED_TASKS, build_prompt_queries, default_metric_question
from preprocessing.prompting import load_config, config_identity, eligible_sequences, build_external_query
from tools.count_input_tokens import resolve_encoding, count_query, token_statistics, _atomic_text, _write_csv
from tools.output_safety import derived_path

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    scope=p.add_mutually_exclusive_group(required=True)
    for name in ('all','external-all','everything'): scope.add_argument('--'+name,action='store_true')
    scope.add_argument('--dataset');scope.add_argument('--task')
    h=p.add_mutually_exclusive_group();h.add_argument('--history-size',type=int);h.add_argument('--history-sizes',nargs='+',type=int)
    p.add_argument('--external-history-sizes',nargs='+',type=int)
    p.add_argument('--model',required=True);p.add_argument('--prompt-mode',choices=['original_llmx','external_generic','external_static'])
    p.add_argument('--preprocessing-config',type=Path);p.add_argument('--metric');p.add_argument('--question-name')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--sample-size',type=int,default=2000);p.add_argument('--sample-threshold',type=int,default=100000)
    p.add_argument('--full-count',action='store_true');p.add_argument('--fallback-encoding',default='cl100k_base')
    p.add_argument('--tokens-per-message',type=int,default=3);p.add_argument('--priming-tokens',type=int,default=3)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/token_counts')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/llmx_data');p.add_argument('--data-path',type=Path)
    return p

def main(argv=None):
    p=parser();a=p.parse_args(argv)
    histories=sorted(set(a.history_sizes or [a.history_size if a.history_size is not None else 1]))
    if min(histories+(a.external_history_sizes or []))<0: p.error('history sizes must be non-negative')
    if a.sample_size<1 or a.sample_threshold<1: p.error('sample size and threshold must be positive')
    if (a.metric is None)!=(a.question_name is None): p.error('metric and question-name must be specified together')
    selected=[a.dataset or a.task] if a.dataset or a.task else [d for d in dataset_ids() if a.everything or (d in EXTERNAL if a.external_all else d not in EXTERNAL)]
    if a.preprocessing_config and len(selected)!=1: p.error('custom configuration requires one dataset')
    out=derived_path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    encoding,fallback=resolve_encoding(a.model,a.fallback_encoding)
    if fallback: print(f'WARNING: unknown model; explicit fallback tokenizer {encoding.name}',flush=True)
    groups=defaultdict(list);seqgroups=defaultdict(list);scopes=[];skipped=[];extrema={};n=0
    temp=derived_path(out/'queries.csv.partial')
    with temp.open('w',newline='') as handle:
        writer=None
        for name in selected:
            if name not in EXTERNAL and name not in REGISTERED_TASKS:
                skipped.append(dict(dataset_id=name,reason='task class/description absent from upstream registry; raw readable; no safe version alias'));continue
            adapter=get_adapter(name)
            if name not in EXTERNAL and (a.data_path or a.data_root.resolve()!= (ROOT/'data/llmx_data').resolve()):
                from dataset_adapters.llmx import LLMXAdapter
                adapter=LLMXAdapter(name,a.data_root)
            if not adapter.sequences(): skipped.append(dict(dataset_id=name,reason='NOT_ACQUIRED'));continue
            config=load_config(name,a.preprocessing_config);mode=config['prompt_mode']
            if a.prompt_mode and mode!=a.prompt_mode and not(a.prompt_mode=='external_generic' and mode=='external_static'):
                p.error(f'{name}: incompatible prompt-mode {a.prompt_mode}; configuration uses {mode}')
            hs=[0] if mode=='external_static' else (a.external_history_sizes or histories) if name in EXTERNAL else histories
            for h in sorted(set(hs)):
                ctext,digest=config_identity(config,h)
                _atomic_text(out/'preprocessing_configs'/f'{digest}.json',ctext+'\n')
                seqs=list(eligible_sequences(adapter,config)) if name in EXTERNAL else adapter.sequences()
                if a.data_path: seqs=[s for s in seqs if (ROOT/s.source_file).resolve()==a.data_path.resolve()]
                if not seqs: raise ValueError(f'no source files selected for {name}')
                counts=[]
                for s in seqs:
                    if name in EXTERNAL: count=max(0,s.length-h)
                    else:
                        from preprocessing.llmx_original import _query_indices
                        metric=a.metric or 'next-action';count=len(_query_indices(metric,h,s.length))
                    counts.append(count)
                cumulative=[];total=0
                for count in counts: total+=count;cumulative.append(total)
                sampled=name in EXTERNAL and not a.full_count and total>a.sample_threshold
                chosen=sorted(random.Random(a.seed).sample(range(total),min(total,a.sample_size))) if sampled else None
                scope='deterministic_sample' if sampled else 'full_selected_scope'
                scope_row=dict(dataset_id=name,prompt_mode=mode,history_size=h,preprocessing_config_sha256=digest,count_scope=scope,eligible_query_pool=total,selected_source_units=len(seqs),dataset_reader_records=sum(s.length for s in adapter.sequences()),sample_size=len(chosen) if chosen is not None else total,seed=a.seed,source_globs=json.dumps(config.get('source_globs',['official episodes'])))
                scopes.append(scope_row)
                def queries():
                    if name not in EXTERNAL:
                        for s in seqs:
                            source=adapter._sources[s.sequence_id]
                            metric,question=(a.metric,a.question_name) if a.metric else default_metric_question(name,Episode.load(source.path))
                            for q in build_prompt_queries(source,metric=metric,question_name=question,history_size=h,seed=a.seed): yield s,q,None
                    else:
                        offsets={};offset=0
                        for s in adapter.sequences(): offsets[s.sequence_id]=offset;offset+=s.length
                        if chosen is not None:
                            positions=((bisect.bisect_right(cumulative,k),k) for k in chosen)
                        else:
                            positions=((j,k) for j,c in enumerate(counts) for k in range((cumulative[j-1] if j else 0),cumulative[j]))
                        for j,k in positions:
                            s=seqs[j];index=k-(cumulative[j-1] if j else 0)+h
                            q,_=build_external_query(adapter,s,index,config,h)
                            yield s,q,(offsets[s.sequence_id]+index if mode=='external_static' else None)
                print(f'{name} H={h}: {scope}, {scope_row["sample_size"]}/{total} queries',flush=True)
                for s,q,row_id in queries():
                    r=count_query(q,encoding,model=a.model,encoding_name=encoding.name,fallback_used=fallback,tokens_per_message=a.tokens_per_message,priming_tokens=a.priming_tokens)
                    r.update(dataset_id=name,source_file=s.source_file,episode_id=s.episode_id,sequence_id=s.sequence_id,row_id=row_id,sequence_key=s.key,preprocessing_name=config['name'],preprocessing_config=ctext,preprocessing_config_sha256=digest,prompt_mode=mode,history_tokens=len(encoding.encode(q.history_text)),question_tokens=len(encoding.encode(q.question_text)),total_content_tokens=r['content_only_tokens'],estimated_api_input_tokens=r['estimated_total_input_tokens'],tokenizer=encoding.name,count_scope=scope,eligible_query_pool=total,sample_seed=a.seed,system_prompt_sha256=__import__('hashlib').sha256(q.system_prompt.encode()).hexdigest(),user_prompt_sha256=__import__('hashlib').sha256(q.user_prompt.encode()).hexdigest())
                    if writer is None: writer=csv.DictWriter(handle,fieldnames=list(r));writer.writeheader()
                    writer.writerow(r);n+=1
                    compact={'estimated_total_input_tokens':r['estimated_total_input_tokens']}
                    key=(name,mode,h,digest);groups[key].append(compact);seqgroups[key+(s.sequence_id,s.episode_id)].append(compact)
                    for label,compare in [('min',lambda x,y:x<y),('max',lambda x,y:x>y)]:
                        if label not in extrema or compare(r['estimated_api_input_tokens'],extrema[label]['estimated_api_input_tokens']): extrema[label]=r.copy()
    temp.replace(out/'queries.csv')
    summaries=[];bysequence=[]
    for key,values in groups.items():
        name,mode,h,digest=key;stats=token_statistics(values);scope=next(s for s in scopes if (s['dataset_id'],s['prompt_mode'],s['history_size'],s['preprocessing_config_sha256'])==key)
        summaries.append(dict(dataset=name,dataset_id=name,data_unit='query' if mode=='original_llmx' else 'static_record' if mode=='external_static' else 'history_window',n_units=len(values),**{k:v for k,v in scope.items() if k!='dataset_id'},mean_input_tokens=stats['mean'],std_input_tokens=stats['std'],min_input_tokens=stats['min'],max_input_tokens=stats['max'],median_input_tokens=stats['median'],p95_input_tokens=stats['p95'],total_input_tokens=stats['total_input_tokens']))
    for key,values in seqgroups.items():
        name,mode,h,digest,sid,eid=key;stats=token_statistics(values)
        scope=next(s for s in scopes if (s['dataset_id'],s['prompt_mode'],s['history_size'],s['preprocessing_config_sha256'])==key[:4])
        bysequence.append(dict(dataset_id=name,episode_id=eid,sequence_id=sid,prompt_mode=mode,history_size=h,preprocessing_config_sha256=digest,count_scope=scope['count_scope'],eligible_query_pool=scope['eligible_query_pool'],number_of_queries=len(values),**{k+'_tokens':stats[k] for k in ['mean','std','min','max','median','p95']},total_tokens=stats['total_input_tokens']))
    for filename,rows in [('token_inventory.csv',summaries),('summary_by_task.csv',summaries),('summary_by_dataset.csv',summaries),('summary_by_sequence.csv',bysequence)]: _write_csv(out/filename,rows)
    byh=[]
    for mode,h in sorted({(r['prompt_mode'],r['history_size']) for r in summaries}):
        rows=[r for r in summaries if r['prompt_mode']==mode and r['history_size']==h]
        byh.append(dict(prompt_mode=mode,history_size=h,number_of_queries=sum(r['n_units'] for r in rows),total_input_tokens=sum(r['total_input_tokens'] for r in rows),scope_note='Sum of counted queries only; external may mix full selected scopes and samples, never a full-dataset extrapolation.'))
    result=dict(schema_version=2,model=a.model,tokenizer=encoding.name,fallback_encoding_used=fallback,number_of_queries=n,number_of_datasets=len({k[0] for k in groups}),api_requests=0,by_history=byh,scopes=scopes,skipped=skipped,extrema=extrema,chat_framing_note=f'Exact content tokens; estimated API input = content + 2*{a.tokens_per_message}+{a.priming_tokens}. No server verification. history/question component token counts are not necessarily additive due to BPE boundaries.',original_upstream_revision='441c5644f8eb93d6aee9c53631feb5fa5bf75b3c')
    _atomic_text(out/'summary.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Counted {n} processed prompts locally; API requests=0. Results: {out}',flush=True)
    return 0
