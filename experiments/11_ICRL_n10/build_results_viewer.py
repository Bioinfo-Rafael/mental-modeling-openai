"""Build the Exp.10 offline viewer for Exp.11, including paired reward answers."""
import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import common

original = import_module('experiments.10_ICRL.analysis.build_results_viewer')
data = original.data
HERE = Path(__file__).resolve().parent


def load_rows(root=HERE):
    api = root / 'results/api'
    paths = [api / name for name in ('manifest.json', 'records.jsonl', 'responses.jsonl', 'requests.jsonl')]
    hashes = {str(p.relative_to(root)): common.file_sha256(p) for p in paths}
    queries = common.read_json(paths[0])['queries']
    planned = data.index_rows(queries, 'manifest')
    records, responses, requests = [data.index_rows(common.read_jsonl(p), p.name) for p in paths[1:]]
    data.require(len(planned) == 40 and set(records) == set(responses) == set(requests) == set(planned),
                 'Expected 40 completed, journaled queries')
    data.require(Counter((q['model_alias'], q['reward_present']) for q in queries)
                 == Counter({(m, r): 10 for m in ('terra', 'luna') for r in (True, False)}),
                 'Expected four N10 conditions')
    pairs = {}
    rows = []
    for q in queries:
        qid = q['query_id']
        r = records[qid]
        data.require(all(r.get(k) == v for k, v in q.items()), 'Record differs from manifest')
        data.require(r['raw_response'] == responses[qid]['raw_response'], 'Response journal mismatch')
        data.require(r['assistant_text'] == r['raw_response']['choices'][0]['message']['content'],
                     'Answer differs from raw response')
        data.require(requests[qid]['kwargs'] == q['request'], 'Request journal mismatch')
        data.require((q['history_start'], q['query_index'], q['history_end_exclusive'])
                     == (q['ordinal'], q['ordinal'] + 20, q['ordinal'] + 20), 'Unexpected target window')
        data.validate_metrics(r)
        pairs.setdefault(q['pair_id'], []).append(q)
        rows.append({**r, 'reasoning_score': None})
    data.require(len(pairs) == 20 and all(len(pair) == 2 and {q['reward_present'] for q in pair} == {True, False}
                 for pair in pairs.values()), 'Missing reward pair')
    for pair in pairs.values():
        on, off = sorted(pair, key=lambda q: not q['reward_present'])
        data.require(all(on[k] == off[k] for k in ('model_alias', 'ordinal', 'episode_path', 'query_index',
                     'system_prompt', 'ground_truth', 'action_value_ground_truth')), 'Pair metadata differs')
        data.require(off['user_prompt'] == ''.join(line for line in on['user_prompt'].splitlines(keepends=True)
                     if not line.startswith('  reward:')), 'Unexpected paired prompt difference')
    data.require(all(common.file_sha256(root / p) == h for p, h in hashes.items()), 'Source changed while reading')
    return rows, hashes


def make_template():
    template = (data.HERE / 'templates/results_viewer_template.html').read_text(encoding='utf-8')
    template = template.replace('10 ICRL', '11 ICRL N10').replace('repeat_id', 'ordinal')
    template = template.replace('反復番号', 'query番号 (0–9)').replace('反復${r.ordinal}', 'query ${r.ordinal}')
    old = 'Pendulum-v1 · episode_9_seed3407 · H=20 · t=0 · 24条件 × 同一質問への3反復。Prompt 4通り × E=1/3/9 × 2モデル。'
    data.require(old in template, 'Exp.10 template description changed')
    template = template.replace(old, 'Pendulum-v1 · episode_0 · H=20 · ICR-FQI K=5 · E=1。2モデル × reward有無 × 異なる10 query = 40回答。history開始0–9、予測step20–29。')
    template = template.replace("method:'Prompt',K:'FQI反復 K',E:'Episode数 E',ordinal:",
                                "reward_present:'履歴reward',ordinal:")
    template = template.replace(" if(key==='method')", " if(key==='reward_present')return v==='true'?'あり':'なし';\n if(key==='method')", 1)
    template = template.replace(' / E=${r.E} / query ${r.ordinal}', " / reward ${r.reward_present?'あり':'なし'} / query ${r.ordinal}")
    template = template.replace('user（補助episode・手順指示を含む送信全文）', 'user（ICR-FQI手順＋target history＋質問の送信全文）')
    template = template.replace('元のhistoryとnext-action question', '当該reward条件のhistoryとnext-action question')
    template = template.replace('ordinal:r.ordinal,reward_present:', 'ordinal:r.ordinal,pair_id:r.pair_id,reward_mode:r.reward_mode,reward_present:')
    template = template.replace('<a href="#answer-start">回答</a>', '<a href="#answer-start">回答</a><a href="#pair-start">reward有無の比較</a>')
    template = template.replace('</style>', '.paired{display:grid;grid-template-columns:1fr 1fr;gap:20px}.paired article{min-width:0;padding:16px;background:#f7f9fc;border-radius:8px}.paired h3{margin-top:0}@media(max-width:700px){.paired{grid-template-columns:1fr}}</style>')
    needle = " const truth={"
    data.require(template.count(needle) == 1, 'Exp.10 detail renderer changed')
    paired = ''' const pairPanel=el('section');pairPanel.className='panel';pairPanel.id='pair-start';
 pairPanel.append(el('h2',`同じqueryの比較 · ${caption('model_alias',r.model_alias)} · query ${r.ordinal} / step ${r.query_index}`));
 pairPanel.append(el('p','左：rewardあり ／ 右：rewardなし。同じstate・history範囲の回答です。絞り込みに関係なく対応する両回答を表示します。'));
 const paired=el('div');paired.className='paired';
 for(const reward of [true,false]){
  const other=rows.find(q=>q.pair_id===r.pair_id&&q.reward_present===reward),article=el('article');
  article.append(el('h3',reward?'rewardあり':'rewardなし'));
  if(other){article.append(el('p',`${names[other.status]||other.status} | bin ${value(other.prediction)} / 正解 ${value(other.ground_truth)} | action ${value(other.action_value_prediction)} / 正解 ${value(other.action_value_ground_truth)}`),el('pre',other.assistant_text??'応答なし'))}
  else article.append(el('p','対応する回答なし'));
  paired.append(article);
 }
 pairPanel.append(paired);$('content').append(pairPanel);
'''
    return template.replace(needle, paired + needle)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the offline Exp.11 viewer; no API calls.')
    parser.add_argument('--output', type=Path, default=HERE / 'view_rawdata.html')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('HTML already exists; choose a new --output path')
    rows, hashes = load_rows()
    html = original.render_html(rows, hashes, make_template())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(html)
    print(f'{len(rows)} answers / 20 reward pairs: {args.output}')


if __name__ == '__main__':
    main()
