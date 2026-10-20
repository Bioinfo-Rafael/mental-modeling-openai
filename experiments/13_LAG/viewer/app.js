import {stateFields,actionFields,derivedFields,defaults,derived} from './model.js';
import {Playback} from './playback.js';
import {renderProbabilities,renderSeries} from './charts.js';
const $=id=>document.getElementById(id);
let data, step=0, uid='A0100', activeTab='series', rows=new Map(), selected=new Set(defaults), request=0;
const allFields=[...stateFields,...actionFields,...derivedFields];
const playback=new Playback($('video'), s=>{step=s;render();}, playing=>{$('play').textContent=playing?'Ⅱ 停止':'▶ 再生';}, (frame,pending,error)=>{
  $('videoTime').textContent=frame?`映像 ${(frame.simulationTime).toFixed(1)} s${pending?' · 同期中':''}`:'映像を読み込み中';
  $('videoStatus').textContent=error||(pending?'映像を同期中':`表示映像 ${(frame.simulationTime).toFixed(1)} s`);
});
function fail(e){$('error').hidden=false;$('error').textContent=`読み込みエラー: ${e.message}`;}
async function getJSON(url){const response=await fetch(url);if(!response.ok)throw new Error(`${url} (${response.status})`);return response.json();}
function option(value,text){const n=document.createElement('option');n.value=value;n.textContent=text;return n;}
function buildRows(){
  rows=new Map(data.rows.filter(r=>r.object_id===uid).map(r=>[Number(r.step),{...r}]));
  for(const [s,row] of rows)Object.assign(row,derived(row,rows.get(s-1),data.dt));
  const events=launches();$('launches').replaceChildren(option('','発射を選択'));
  for(const event of events)$('launches').append(option(event.action_step,`${event.missile_id} · step ${event.action_step} / ${(event.action_step*data.dt).toFixed(1)} s → 初出 ${(event.first_acmi_step*data.dt).toFixed(1)} s`));
  $('launches').disabled=!events.length;
}
function launches(){return data.launches.filter(e=>e.launcher_id===uid).sort((a,b)=>a.action_step-b.action_step);}
function renderAnalysis(){
  if(!data)return;
  $('analysisContext').textContent=`${uid} · step ${step} · ${(step*data.dt).toFixed(1)} s`;
  if(activeTab==='probability')renderProbabilities($('probabilities'),rows.get(step),data.actions);
  else {const scroll=$('charts').scrollTop;renderSeries($('charts'),{fields:allFields.filter(f=>selected.has(f[0])),rows,step,dt:data.dt,lastStep:data.lastStep,windowSize:$('window').value,launches:launches(),onSeek:s=>playback.seek(s)});$('charts').scrollTop=scroll;}
}
function render(){
  if(!data)return;
  $('time').replaceChildren(document.createTextNode((step*data.dt).toFixed(1)+' '),Object.assign(document.createElement('small'),{textContent:'s'}));
  $('stepLabel').textContent=`STEP ${step}`;
  if(document.activeElement!==$('stepInput'))$('stepInput').value=step;
  if(document.activeElement!==$('timeInput'))$('timeInput').value=(step*data.dt).toFixed(1);
  $('scrubber').value=step;
  $('back').disabled=$('backSecond').disabled=step===0;
  $('forward').disabled=$('forwardSecond').disabled=step===data.lastStep;
  $('prevLaunch').disabled=!launches().some(e=>e.action_step<step);
  $('nextLaunch').disabled=!launches().some(e=>e.action_step>step);
  renderAnalysis();
}
async function loadEpisode(item){
  const token=++request;playback.pause();$('error').hidden=true;
  $('play').disabled=true;
  try{
    const loaded=await getJSON(item.data);if(token!==request)return;
    data=loaded;step=0;
    $('aircraft').replaceChildren(...data.aircraft.map(id=>option(id,`${id} · ${id.startsWith('A')?'学習モデル':'固定ルール'}`)));$('aircraft').value=uid;
    buildRows();$('scrubber').max=$('stepInput').max=data.lastStep;$('timeInput').max=(data.lastStep*data.dt).toFixed(1);
    $('duration').textContent=`${(data.lastStep*data.dt).toFixed(1)} s / ${data.lastStep} steps`;
    playback.load(data);$('play').disabled=false;
  }catch(e){if(token===request)fail(e);}
}
function switchTab(tab){activeTab=tab;const isSeries=tab==='series';$('seriesPanel').hidden=!isSeries;$('probabilityPanel').hidden=isSeries;$('tabSeries').setAttribute('aria-selected',isSeries);$('tabProbability').setAttribute('aria-selected',!isSeries);renderAnalysis();}
function buildFields(){
  $('fields').replaceChildren();for(const [key,title] of allFields){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=selected.has(key);input.addEventListener('change',()=>{input.checked?selected.add(key):selected.delete(key);renderAnalysis();});label.append(input,document.createTextNode(title));$('fields').append(label);}
}
$('aircraft').addEventListener('change',()=>{uid=$('aircraft').value;buildRows();render();});
$('play').addEventListener('click',()=>playback.toggle());
for(const [id,delta]of [['back',-1],['forward',1],['backSecond',-5],['forwardSecond',5]])$(id).addEventListener('click',()=>playback.seek(step+delta));
$('scrubber').addEventListener('input',e=>playback.seek(e.target.value));
$('stepInput').addEventListener('change',e=>{playback.seek(e.target.value);e.target.value=step;});
$('timeInput').addEventListener('change',e=>{if(data)playback.seek(Number(e.target.value)/data.dt);e.target.value=data?(step*data.dt).toFixed(1):0;});
$('speed').addEventListener('change',e=>playback.setSpeed(e.target.value));
$('window').addEventListener('change',renderAnalysis);
$('tabSeries').addEventListener('click',()=>switchTab('series'));
$('tabProbability').addEventListener('click',()=>switchTab('probability'));
$('prevLaunch').addEventListener('click',()=>{const e=launches().filter(e=>e.action_step<step).at(-1);if(e)playback.seek(e.action_step);});
$('nextLaunch').addEventListener('click',()=>{const e=launches().find(e=>e.action_step>step);if(e)playback.seek(e.action_step);});
$('launches').addEventListener('change',e=>{if(e.target.value!=='')playback.seek(e.target.value);e.target.value='';});
for(const [id,keys] of [['allFields',allFields.map(f=>f[0])],['defaultFields',defaults],['clearFields',[]]])$(id).addEventListener('click',()=>{selected=new Set(keys);buildFields();renderAnalysis();});
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA','SUMMARY'].includes(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;
  if(e.code==='Space'){e.preventDefault();playback.toggle();}
  if(e.code==='ArrowLeft'||e.code==='ArrowRight'){e.preventDefault();playback.seek(step+(e.code==='ArrowLeft'?-1:1)*(e.shiftKey?5:1));}
});
buildFields();
try {const catalog=await getJSON('/data/viz/viewer/index.json');$('episode').replaceChildren(...catalog.map(item=>option(item.id,item.title)));$('episode').addEventListener('change',()=>loadEpisode(catalog.find(item=>item.id===$('episode').value)));await loadEpisode(catalog[0]);}catch(e){fail(e);}
