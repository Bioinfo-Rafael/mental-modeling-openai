'use strict';
const records=JSON.parse(document.getElementById('dataset').textContent);
const $=id=>document.getElementById(id);
const byId=new Map(records.map(r=>[r.query_id,r]));
const fields={model:'model_alias',task:'task',metric:'metric',history:'H'};
const modelNames={'3.5':'GPT-3.5 (Joint)',sol:'Sol',terra:'Terra',luna:'Luna'};
const rubric={1:'明示的な定量・数学的推論',2:'具体的な履歴・観測に基づく推論',3:'Agent・policy傾向の一般化',4:'一般的な物理直感・曖昧な推測など'};
const typeNames={correct:'正解例',incorrect:'不正解例'};
const storageKey='joint-fewshot-selection-v1';
let selections=new Map(),sampleOrdinal=0,folder=null;
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
function button(text,fn){const b=el('button',text);b.addEventListener('click',fn);return b}
function notify(text){$('notice').textContent=text}
function populate(id,values,names={},all=false){const select=$(id);if(all)select.append(new Option('すべて',''));for(const v of values)select.append(new Option(names[v]??String(v),String(v)))}
for(const [id,key] of Object.entries(fields)){
 let values=[...new Set(records.map(r=>r[key]))];
 if(id==='model')values=['sol','terra','luna','3.5'].filter(v=>values.includes(v));
 else if(id==='history')values.sort((a,b)=>a-b);else values.sort();
 populate(id,values,id==='model'?modelNames:{});
 populate('export-'+id,values,id==='model'?modelNames:{},true);
 $(id).addEventListener('change',render);
}
function restore(items){
 if(!Array.isArray(items))throw Error('examples配列がありません。');
 const next=new Map();
 for(const x of items){
  const r=byId.get(x.query_id);
  if(!r||!r.fewshot.response_available||!['correct','incorrect'].includes(x.example_type))throw Error('不明なID・回答なし・無効な区分を含みます。');
  if(next.has(x.query_id))throw Error('query_idが重複しています。');
  if('response_sha256' in x&&x.response_sha256!==r.fewshot.response_sha256)throw Error('元回答が一致しません。');
  if('question' in x&&x.question!==r.fewshot.question)throw Error('元入力が一致しません。');
  if('answer' in x&&x.answer!==r.fewshot.answer)throw Error('元回答の本文が一致しません。');
  next.set(x.query_id,{example_type:x.example_type,note:String(x.note??'')});
 }
 return next;
}
try{const saved=localStorage.getItem(storageKey);if(saved)selections=restore(JSON.parse(saved));}catch(e){notify('選択を復元できませんでした。JSONから読み込めます。 '+e.message)}
function selectedExamples(){return [...selections].map(([id,s],i)=>({...byId.get(id).fewshot,...s,selection_order:i}))}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(selectedExamples().map(e=>({query_id:e.query_id,example_type:e.example_type,note:e.note,response_sha256:e.response_sha256}))));}catch(e){notify('ブラウザ保存が使えません。選択をJSONで出力してください。')}updateCount()}
function updateCount(){const values=[...selections.values()];const n=values.filter(v=>v.example_type==='correct').length;$('open-selection').textContent=`選択済み ${values.length}件 ☰`;$('selection-count').textContent=`正解例 ${n}件 / 不正解例 ${values.length-n}件`}
function selected(){return records.filter(r=>Object.entries(fields).every(([id,key])=>String(r[key])===$(id).value)).sort((a,b)=>a.ordinal-b.ordinal)}
function current(){return selected().find(r=>r.ordinal===sampleOrdinal)}
function panel(title,text){const box=el('section');box.className='panel';box.append(el('h3',title));if(text!==undefined)box.append(el('pre',text));return box}
function details(title,value){const box=el('details');box.className='panel';box.append(el('summary',title),el('pre',JSON.stringify(value,null,2)));return box}
function fmt(x){return typeof x==='number'?Number(x.toPrecision(7)).toString():String(x)}
function compact(c){if(!c.parsed)return `${c.label}：抽出できず`;if(c.element_matches){const n=c.element_matches.filter(Boolean).length;return `${c.label}：${n===c.element_matches.length?'一致':'不一致'} (${n}/${c.element_matches.length})`;}return `${c.label}：絶対誤差 [${c.absolute_error.map(fmt).join(', ')}]`;}
function render(){
 const rows=selected();let index=rows.findIndex(r=>r.ordinal===sampleOrdinal);if(index<0){index=0;sampleOrdinal=rows[0]?.ordinal??0}
 $('content').replaceChildren();$('quick-score').replaceChildren();$('quick-comparison').replaceChildren();$('empty').textContent=rows.length?'':'該当するデータはありません。';
 $('previous').disabled=index<=0;$('next').disabled=index>=rows.length-1||!rows.length;
 $('position').textContent=rows.length?`${index+1} / ${rows.length} · ordinal ${sampleOrdinal}`:'0 / 0';
 if(!rows.length)return;
 const r=rows[index],e=r.fewshot;
 $('status').textContent=e.response_available?'応答あり':'回答なし';
 for(const t of ['correct','incorrect']){const b=$('select-'+t);b.disabled=!e.response_available;b.setAttribute('aria-pressed',String(selections.get(r.query_id)?.example_type===t));}
 $('quick-score').textContent=`Reasoning score：${e.reasoning_score} — ${rubric[e.reasoning_score]}${e.reasoning_available?'':'（Reasoning抽出なし・既存スコア）'}`;
 for(const c of Object.values(e.comparison))$('quick-comparison').append(el('span',e.response_available?compact(c):`${c.label}：回答なし`));
 const messages=r.request?.messages??[{role:'system',content:r.system_prompt},{role:'user',content:r.user_prompt}];
 for(const [i,m] of messages.entries()){const block=panel(`入力 ${i+1} — ${m.role}`,typeof m.content==='string'?m.content:JSON.stringify(m.content,null,2));if(i===0)block.id='input-start';$('content').append(block)}
 const answerBlock=panel('出力 — assistant_text',e.answer??'回答は保存されていません。');answerBlock.id='answer-start';$('content').append(answerBlock);
 const truth=panel('正解データ・回答との比較');truth.id='truth-start';
 const scroll=el('div');scroll.className='table-scroll';const table=el('table'),head=el('tr');for(const title of ['項目','次元','モデルの予測','正解','一致／不一致','絶対誤差'])head.append(el('th',title));table.append(head);
 for(const c of Object.values(e.comparison))for(let i=0;i<c.ground_truth.length;i++){
  const row=el('tr');const match=c.element_matches?.[i];const cells=[c.label,c.dimensions[i],c.parsed?fmt(c.prediction[i]):e.response_available?'抽出できず':'回答なし',fmt(c.ground_truth[i]),!c.parsed?'—':match===undefined?'閾値未設定':match?'一致':'不一致',c.absolute_error?fmt(c.absolute_error[i]):'—'];
  for(const text of cells)row.append(el('td',text));table.append(row);
 }
 scroll.append(table);truth.append(scroll);
 truth.append(el('p',`Reasoning score：${e.reasoning_score} — ${rubric[e.reasoning_score]}`));
 const note=el('p','Scoreは正誤ではありません。状態変化量はNext・Lastとも後の状態 − 前の状態。連続値の誤差は表示上のみ丸め、JSONは元の精度を保持します。');note.className='subnote';truth.append(note);
 if(!e.reasoning_available)truth.append(el('p','Reasoning本文は抽出されていません。スコアは元CSVの値を保持しています。'));
 $('content').append(truth);
 const metadata=panel('条件・識別子・時間・token等（プロンプトとは別表示）');metadata.id='metadata-start';metadata.classList.add('metadata');const mt=el('table');
 for(const [key,value] of Object.entries(r)){
  if(['system_prompt','user_prompt','assistant_text'].includes(key)||(value!==null&&typeof value==='object'))continue;
  const tr=el('tr'),td=el('td');td.append(el('pre',String(value)));tr.append(el('th',key),td);mt.append(tr);
 }
 metadata.append(mt);$('content').append(metadata);
 if(r.exception)$('content').append(details('例外 — exception',r.exception));
 $('content').append(details('API送信設定（messages以外）',Object.fromEntries(Object.entries(r.request??{}).filter(([k])=>k!=='messages'))));
 for(const key of ['usage','score','prediction','ground_truth','attempts','raw_response'])if(key in r)$('content').append(details(['score','prediction','ground_truth'].includes(key)?`${key}（取得時の旧parser情報）`:key,r[key]));
 $('content').append(details('このqueryの全フィールド（JSON）',r));updateCount();
}
function toggle(type,id=current()?.query_id){if(!id||!byId.get(id).fewshot.response_available)return;const old=selections.get(id);if(old?.example_type===type)selections.delete(id);else selections.set(id,{example_type:type,note:old?.note??''});persist();render();if($('selection-dialog').open)renderSelections()}
for(const type of ['correct','incorrect'])$('select-'+type).addEventListener('click',()=>toggle(type));
$('previous').addEventListener('click',()=>{const rows=selected(),i=rows.findIndex(r=>r.ordinal===sampleOrdinal);if(i>0){sampleOrdinal=rows[i-1].ordinal;render()}});
$('next').addEventListener('click',()=>{const rows=selected(),i=rows.findIndex(r=>r.ordinal===sampleOrdinal);if(i<rows.length-1){sampleOrdinal=rows[i+1].ordinal;render()}});
$('reset').addEventListener('click',()=>{if(selections.size&&!confirm('選択済みの例をすべて解除しますか？'))return;selections.clear();persist();render();renderSelections()});
function move(id,delta){const entries=[...selections],i=entries.findIndex(([k])=>k===id),j=i+delta;if(j<0||j>=entries.length)return;[entries[i],entries[j]]=[entries[j],entries[i]];selections=new Map(entries);persist();renderSelections()}
function renderSelections(){
 updateCount();const list=$('selection-list');list.replaceChildren();if(!selections.size)list.append(el('p','まだ選択されていません。回答を見て「正解例」「不正解例」を押してください。'));
 selectedExamples().forEach((e,i)=>{
  const box=el('article');box.className='selection-item';box.append(el('h3',`${i+1}. ${typeNames[e.example_type]} · ${modelNames[e.model]}`));
  box.append(el('div',`${e.task} / ${e.metric} / H${e.H} / ordinal ${e.ordinal}`));const summary=el('div',`Score ${e.reasoning_score} · ${Object.values(e.comparison).map(compact).join(' / ')}`);summary.className='summary';box.append(summary);
  const bar=el('div');bar.className='toolbar';bar.append(button('表示',()=>{const r=byId.get(e.query_id);for(const [id,key] of Object.entries(fields))$(id).value=r[key];sampleOrdinal=r.ordinal;$('selection-dialog').close();render();$('answer-start').scrollIntoView()}));
  for(const type of ['correct','incorrect']){const b=button(typeNames[type],()=>toggle(type,e.query_id));b.className=type+(e.example_type===type?' active':'');b.setAttribute('aria-pressed',String(e.example_type===type));bar.append(b)}
  const up=button('↑',()=>move(e.query_id,-1)),down=button('↓',()=>move(e.query_id,1));up.disabled=i===0;down.disabled=i===selections.size-1;up.setAttribute('aria-label','一つ上へ');down.setAttribute('aria-label','一つ下へ');bar.append(up,down,button('削除',()=>{selections.delete(e.query_id);persist();render();renderSelections()}));box.append(bar);
  const note=el('textarea');note.value=e.note;note.placeholder='選んだ理由・メモ';note.setAttribute('aria-label','例のメモ');note.addEventListener('input',()=>{selections.get(e.query_id).note=note.value;persist()});box.append(note);list.append(box);
 });
}
$('open-selection').addEventListener('click',()=>{renderSelections();$('selection-dialog').showModal()});
$('close-selection').addEventListener('click',()=>$('selection-dialog').close());
$('import-button').addEventListener('click',()=>$('import-file').click());
$('import-file').addEventListener('change',async()=>{try{const file=$('import-file').files[0];if(!file)return;const data=JSON.parse(await file.text());const next=restore(Array.isArray(data)?data:data.examples);if(selections.size&&!confirm('JSONの内容で現在の選択を置き換えますか？'))return;selections=next;persist();render();renderSelections();notify(`${next.size}件を読み込みました。`)}catch(e){alert('読み込み失敗：'+e.message)}finally{$('import-file').value=''}});
// Identical FNV-1a seeded ordering is implemented in prepare_prompt.py.
function rank(seed,id){let h=2166136261;for(const c of new TextEncoder().encode(`${seed}:${id}`)){h^=c;h=Math.imul(h,16777619)>>>0}return h}
function randomOrder(rows,seed){return [...rows].sort((a,b)=>rank(seed,a.query_id)-rank(seed,b.query_id)||(a.query_id<b.query_id?-1:a.query_id>b.query_id?1:0))}
function exportOptions(){const scores=[...document.querySelectorAll('input[name=export-score]:checked')].map(x=>Number(x.value));return {task:$('export-task').value,metric:$('export-metric').value,model:$('export-model').value,H:$('export-history').value?Number($('export-history').value):null,scores,correct:Number($('correct-count').value),incorrect:Number($('incorrect-count').value),sampling:$('sampling').value,seed:Number($('seed').value),order:$('ordering').value}}
function filteredExamples(o){return selectedExamples().filter(e=>(!o.task||e.task===o.task)&&(!o.metric||e.metric===o.metric)&&(!o.model||e.model===o.model)&&(!o.H||e.H===o.H)&&o.scores.includes(e.reasoning_score))}
function subset(o){
 for(const k of ['correct','incorrect','seed'])if(!Number.isSafeInteger(o[k])||o[k]<0)throw Error('件数とseedは0以上の整数にしてください。');if(o.seed>4294967295)throw Error('seedは4294967295以下にしてください。');
 const rows=filteredExamples(o);let groups=[];
 for(const type of ['correct','incorrect']){let group=rows.filter(e=>e.example_type===type);if(group.length<o[type])throw Error(`${typeNames[type]}が不足しています（指定${o[type]} / 候補${group.length}）。`);if(o.sampling==='random')group=randomOrder(group,o.seed);groups.push(group.slice(0,o[type]));}
 let chosen=groups.flat();if(!chosen.length)throw Error('出力件数を1件以上にしてください。');
 if(o.order==='selection')chosen.sort((a,b)=>a.selection_order-b.selection_order);
 if(o.order==='alternate'){chosen=[];for(let i=0;i<Math.max(...groups.map(g=>g.length));i++)for(const g of groups)if(g[i])chosen.push(g[i]);}
 if(o.order==='shuffle')chosen=randomOrder(chosen,(o.seed+1)>>>0);
 return chosen;
}
function availability(resetCounts=false){const o=exportOptions(),rows=filteredExamples(o),n=rows.filter(e=>e.example_type==='correct').length;if(resetCounts){$('correct-count').value=n;$('incorrect-count').value=rows.length-n;}$('export-availability').textContent=`条件に合う候補：正解例 ${n}件 / 不正解例 ${rows.length-n}件`;$('export-message').textContent='';}
function openExport(){if($('selection-dialog').open)$('selection-dialog').close();$('export-task').value=$('task').value;$('export-metric').value=$('metric').value;availability(true);$('export-dialog').showModal()}
$('open-export').addEventListener('click',openExport);$('selection-export').addEventListener('click',openExport);$('close-export').addEventListener('click',()=>$('export-dialog').close());
for(const id of ['export-task','export-metric','export-model','export-history'])$(id).addEventListener('change',()=>availability(true));
for(const checkbox of document.querySelectorAll('input[name=export-score]'))checkbox.addEventListener('change',()=>availability(true));
function label(e){return {example_type:e.example_type,reasoning_score:e.reasoning_score,reasoning_available:e.reasoning_available,ground_truth:e.ground_truth,comparison:e.comparison}}
const preamble='以下は過去のQuestion / Answerと、その評価Labelです。正解例は参考にし、不正解例は誤りを繰り返さないために参照してください。Labelは評価情報であり、新しい質問への出力形式ではありません。新しい質問には、その質問で指定された形式で回答してください。';
function promptText(examples){return preamble+'\n\n'+examples.map(e=>`Question:\n${e.question}\n\nAnswer:\n${e.answer}\n\nLabel:\n${JSON.stringify(label(e),null,2)}`).join('\n\n---\n\n')+'\n'}
function envelope(examples,kind,options=null){return {schema_version:1,kind,created_at:new Date().toISOString(),options,examples}}
$('choose-folder').addEventListener('click',async()=>{if(!window.showDirectoryPicker){$('folder-status').textContent='このブラウザは非対応です。ダウンロード先からdata_prepへ移動してください。';return}try{folder=await window.showDirectoryPicker({id:'fewshot-data-prep',mode:'readwrite'});$('folder-status').textContent=`保存先：${folder.name}`;}catch(e){if(e.name!=='AbortError')$('folder-status').textContent=e.message;}});
async function saveFile(name,text,mime){
 if(folder){const handle=await folder.getFileHandle(name,{create:true});const stream=await handle.createWritable();await stream.write(text);await stream.close();return `保存しました：${folder.name}/${name}`;}
 const url=URL.createObjectURL(new Blob([text],{type:mime}));const a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);return `ダウンロードしました：${name}（data_prepに保存してください）`;
}
function stamp(){return new Date().toISOString().replace(/[:.]/g,'-')+'_'+Math.random().toString(36).slice(2,6)}
async function exportAction(kind){try{const o=exportOptions();const examples=kind==='all'?selectedExamples():subset(o);if(!examples.length)throw Error('選択済みの例がありません。');let message;
 if(kind==='copy'){await navigator.clipboard.writeText(promptText(examples));message=`${examples.length}件のテキストをコピーしました。`;}
 else if(kind==='txt')message=await saveFile(`fewshot_prompt_${stamp()}.txt`,promptText(examples),'text/plain;charset=utf-8');
 else message=await saveFile(`${kind==='all'?'selected_examples':'fewshot_examples'}_${stamp()}.json`,JSON.stringify(envelope(examples,kind==='all'?'candidate_pool':'fewshot_subset',kind==='all'?null:o),null,2)+'\n','application/json');
 $('export-message').className='ok';$('export-message').textContent=message;
 }catch(e){$('export-message').className='error';$('export-message').textContent=e.message;}}
$('export-all').addEventListener('click',()=>exportAction('all'));$('export-json').addEventListener('click',()=>exportAction('json'));$('export-txt').addEventListener('click',()=>exportAction('txt'));$('copy-txt').addEventListener('click',()=>exportAction('copy'));
render();
