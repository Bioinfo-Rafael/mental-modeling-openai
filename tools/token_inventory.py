"""Local processed-prompt inventory; no API client is created."""
import argparse
import bisect
import csv
import json
import random
from dataclasses import dataclass
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

@dataclass
class TokenInventoryResult:
    queries: list
    summary: list
    by_sequence: list
    scopes: list
    skipped: list
    metadata: dict
    preprocessing_configs: dict


def build_token_inventory(
    datasets="everything", history_sizes=(1,), model="gpt-4o", *,
    external_sample_threshold=100000, external_sample_size=2000, seed=0,
    external_full=False, external_history_sizes=None, preprocessing_config=None,
    prompt_mode=None, metric=None, question_name=None, fallback_encoding="cl100k_base",
    tokens_per_message=3, priming_tokens=3, data_root=ROOT / "data/llmx_data",
    data_path=None, collect_queries=True, query_sink=None,
):
    """Compute live processed-prompt counts without writing files or calling an API.

    Default returns all query rows in memory. CLI uses query_sink + collect_queries=False
    to stream large CSVs. Use inventory_context(live_official, live_external) to make
    readers use a fresh in-memory raw scan instead of saved inventory.
    """
    histories = sorted(set(history_sizes))
    external_history_sizes = sorted(set(external_history_sizes)) if external_history_sizes is not None else None
    if not histories or external_history_sizes == []:
        raise ValueError("history size lists must not be empty")
    if min(histories + (external_history_sizes or [])) < 0:
        raise ValueError("history sizes must be non-negative")
    if external_sample_size < 1 or external_sample_threshold < 1:
        raise ValueError("sample size and threshold must be positive")
    if (metric is None) != (question_name is None):
        raise ValueError("metric and question_name must be specified together")
    available = dataset_ids()
    if isinstance(datasets, str):
        if datasets in {"everything", "all", "external-all"}:
            selected = [d for d in available if datasets == "everything" or
                        (d in EXTERNAL if datasets == "external-all" else d not in EXTERNAL)]
        else:
            selected = [datasets]
    else:
        selected = list(dict.fromkeys(datasets))
    if not selected or set(selected) - set(available):
        raise ValueError(f"unknown or empty dataset selection: {selected}")
    if preprocessing_config and len(selected) != 1:
        raise ValueError("custom configuration requires one dataset")
    data_root = Path(data_root)
    data_path = Path(data_path) if data_path is not None else None
    query_records, preprocessing_configs = [], {}
    encoding,fallback=resolve_encoding(model,fallback_encoding)
    if fallback: print(f'WARNING: unknown model; explicit fallback tokenizer {encoding.name}',flush=True)
    groups=defaultdict(list);seqgroups=defaultdict(list);scopes=[];skipped=[];extrema={};n=0
    for name in selected:
        if name not in EXTERNAL and name not in REGISTERED_TASKS:
            skipped.append(dict(dataset_id=name,reason='task class/description absent from upstream registry; raw readable; no safe version alias'));continue
        adapter=get_adapter(name)
        if name not in EXTERNAL and (data_path or data_root.resolve()!= (ROOT/'data/llmx_data').resolve()):
            from dataset_adapters.llmx import LLMXAdapter
            adapter=LLMXAdapter(name,data_root)
        if not adapter.sequences(): skipped.append(dict(dataset_id=name,reason='NOT_ACQUIRED'));continue
        config=load_config(name,preprocessing_config);mode=config['prompt_mode']
        if prompt_mode and mode!=prompt_mode and not(prompt_mode=='external_generic' and mode=='external_static'):
            raise ValueError(f'{name}: incompatible prompt-mode {prompt_mode}; configuration uses {mode}')
        hs=[0] if mode=='external_static' else (external_history_sizes or histories) if name in EXTERNAL else histories
        for h in sorted(set(hs)):
            ctext,digest=config_identity(config,h)
            preprocessing_configs[digest] = ctext
            seqs=list(eligible_sequences(adapter,config)) if name in EXTERNAL else adapter.sequences()
            if data_path: seqs=[s for s in seqs if (ROOT/s.source_file).resolve()==data_path.resolve()]
            if not seqs: raise ValueError(f'no source files selected for {name}')
            counts=[]
            for s in seqs:
                if name in EXTERNAL: count=max(0,s.length-h)
                else:
                    from preprocessing.llmx_original import _query_indices
                    count=len(_query_indices(metric or 'next-action',h,s.length))
                counts.append(count)
            cumulative=[];total=0
            for count in counts: total+=count;cumulative.append(total)
            sampled=name in EXTERNAL and not external_full and total>external_sample_threshold
            chosen=sorted(random.Random(seed).sample(range(total),min(total,external_sample_size))) if sampled else None
            scope='deterministic_sample' if sampled else 'full_selected_scope'
            scope_row=dict(dataset_id=name,prompt_mode=mode,history_size=h,preprocessing_config_sha256=digest,preprocessing_name=config['name'],count_scope=scope,eligible_query_pool=total,selected_source_units=len(seqs),dataset_reader_records=sum(s.length for s in adapter.sequences()),sample_size=len(chosen) if chosen is not None else total,seed=seed,source_globs=json.dumps(config.get('source_globs',['official episodes'])))
            scopes.append(scope_row)
            def queries():
                if name not in EXTERNAL:
                    for s in seqs:
                        source=adapter._sources[s.sequence_id]
                        query_metric,question=(metric,question_name) if metric else default_metric_question(name,Episode.load(source.path))
                        for q in build_prompt_queries(source,metric=query_metric,question_name=question,history_size=h,seed=seed): yield s,q,None
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
                r=count_query(q,encoding,model=model,encoding_name=encoding.name,fallback_used=fallback,tokens_per_message=tokens_per_message,priming_tokens=priming_tokens)
                r.update(dataset_id=name,source_file=s.source_file,episode_id=s.episode_id,sequence_id=s.sequence_id,row_id=row_id,sequence_key=s.key,preprocessing_name=config['name'],preprocessing_config=ctext,preprocessing_config_sha256=digest,prompt_mode=mode,history_tokens=len(encoding.encode(q.history_text)),question_tokens=len(encoding.encode(q.question_text)),total_content_tokens=r['content_only_tokens'],estimated_api_input_tokens=r['estimated_total_input_tokens'],tokenizer=encoding.name,count_scope=scope,eligible_query_pool=total,sample_size=scope_row['sample_size'],seed=seed,sample_seed=seed,system_prompt_sha256=__import__('hashlib').sha256(q.system_prompt.encode()).hexdigest(),user_prompt_sha256=__import__('hashlib').sha256(q.user_prompt.encode()).hexdigest())
                if collect_queries: query_records.append(r)
                if query_sink is not None: query_sink(r)
                n += 1
                compact={'estimated_total_input_tokens':r['estimated_total_input_tokens']}
                key=(name,mode,h,digest);groups[key].append(compact);seqgroups[key+(s.sequence_id,s.episode_id)].append(compact)
                for label,compare in [('min',lambda x,y:x<y),('max',lambda x,y:x>y)]:
                    if label not in extrema or compare(r['estimated_api_input_tokens'],extrema[label]['estimated_api_input_tokens']): extrema[label]=r.copy()
    summaries=[];bysequence=[]
    for key,values in groups.items():
        name,mode,h,digest=key;stats=token_statistics(values);scope=next(s for s in scopes if (s['dataset_id'],s['prompt_mode'],s['history_size'],s['preprocessing_config_sha256'])==key)
        summaries.append(dict(dataset=name,dataset_id=name,data_unit='query' if mode=='original_llmx' else 'static_record' if mode=='external_static' else 'history_window',n_units=len(values),**{k:v for k,v in scope.items() if k!='dataset_id'},mean_input_tokens=stats['mean'],std_input_tokens=stats['std'],min_input_tokens=stats['min'],max_input_tokens=stats['max'],median_input_tokens=stats['median'],p95_input_tokens=stats['p95'],total_input_tokens=stats['total_input_tokens']))
    for key,values in seqgroups.items():
        name,mode,h,digest,sid,eid=key;stats=token_statistics(values)
        scope=next(s for s in scopes if (s['dataset_id'],s['prompt_mode'],s['history_size'],s['preprocessing_config_sha256'])==key[:4])
        bysequence.append(dict(dataset_id=name,episode_id=eid,sequence_id=sid,prompt_mode=mode,history_size=h,preprocessing_config_sha256=digest,count_scope=scope['count_scope'],eligible_query_pool=scope['eligible_query_pool'],number_of_queries=len(values),**{k+'_tokens':stats[k] for k in ['mean','std','min','max','median','p95']},total_tokens=stats['total_input_tokens']))
    byh=[]
    for mode,h in sorted({(r['prompt_mode'],r['history_size']) for r in summaries}):
        rows=[r for r in summaries if r['prompt_mode']==mode and r['history_size']==h]
        byh.append(dict(prompt_mode=mode,history_size=h,number_of_queries=sum(r['n_units'] for r in rows),total_input_tokens=sum(r['total_input_tokens'] for r in rows),scope_note='Sum of counted queries only; external may mix full selected scopes and samples, never a full-dataset extrapolation.'))
    result=dict(schema_version=2,model=model,tokenizer=encoding.name,fallback_encoding_used=fallback,number_of_queries=n,number_of_datasets=len({k[0] for k in groups}),api_requests=0,by_history=byh,scopes=scopes,skipped=skipped,extrema=extrema,chat_framing_note=f'Exact content tokens; estimated API input = content + 2*{tokens_per_message}+{priming_tokens}. No server verification. history/question component token counts are not necessarily additive due to BPE boundaries.',original_upstream_revision='441c5644f8eb93d6aee9c53631feb5fa5bf75b3c')
    return TokenInventoryResult(query_records, summaries, bysequence, scopes, skipped,
                                result, preprocessing_configs)


def save_token_inventory(result, output_dir, *, include_queries=True):
    """Explicit derived-output writer; never used by live Notebook analysis."""
    out = derived_path(output_dir)
    if include_queries:
        _write_csv(out / "queries.csv", result.queries)
    for filename, rows in [
        ("token_inventory.csv", result.summary), ("summary_by_task.csv", result.summary),
        ("summary_by_dataset.csv", result.summary), ("summary_by_sequence.csv", result.by_sequence),
    ]:
        _write_csv(out / filename, rows)
    for digest, text in result.preprocessing_configs.items():
        _atomic_text(out / "preprocessing_configs" / f"{digest}.json", text + "\n")
    _atomic_text(out / "summary.json", json.dumps(result.metadata, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    args = parser().parse_args(argv)
    datasets = args.dataset or args.task or ("everything" if args.everything else
                                             "external-all" if args.external_all else "all")
    out = derived_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    temporary = derived_path(out / "queries.csv.partial")
    with temporary.open("w", newline="") as handle:
        writer = None
        def write_query(row):
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
        result = build_token_inventory(
            datasets=datasets, history_sizes=args.history_sizes or [args.history_size if args.history_size is not None else 1],
            model=args.model, external_sample_threshold=args.sample_threshold,
            external_sample_size=args.sample_size, seed=args.seed, external_full=args.full_count,
            external_history_sizes=args.external_history_sizes, preprocessing_config=args.preprocessing_config,
            prompt_mode=args.prompt_mode, metric=args.metric, question_name=args.question_name,
            fallback_encoding=args.fallback_encoding, tokens_per_message=args.tokens_per_message,
            priming_tokens=args.priming_tokens, data_root=args.data_root, data_path=args.data_path,
            collect_queries=False, query_sink=write_query,
        )
    temporary.replace(out / "queries.csv")
    save_token_inventory(result, out, include_queries=False)
    print(f"Counted {result.metadata['number_of_queries']} processed prompts locally; API requests=0. Results: {out}")
    return 0
