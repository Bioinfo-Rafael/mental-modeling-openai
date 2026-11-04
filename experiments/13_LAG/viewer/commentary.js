import {replayState,narrativeState} from './commentary-model.js';
const el = (tag, text, cls) => Object.assign(document.createElement(tag), {textContent:text || '', className:cls || ''});
async function json(url) {
  const response = await fetch(url, {cache:'no-store'});
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}
export class Commentary {
  constructor(panel, onChange) {
    this.panel=panel; this.onChange=onChange; this.token=0; this.count=4; this.repeat=1;
    const toolbar=el('div','','goal-toolbar');
    this.runs=el('select');this.runs.setAttribute('aria-label','目的の実験run');
    this.counts=el('select');this.counts.setAttribute('aria-label','目的の選択肢数');
    for(const k of [4,8,32])this.counts.append(Object.assign(el('option',`${k}択`),{value:String(k)}));
    this.repeats=el('select');this.repeats.setAttribute('aria-label','実験の反復');
    const refresh=el('button','結果を更新');
    toolbar.append(this.runs,this.counts,this.repeats,refresh);
    this.note=el('p','','goal-note');this.cards=el('div','','goal-cards');
    this.history=el('div','','goal-history');this.details=el('details');
    this.details.append(el('summary','計測・棄却の詳細'));this.audit=el('div','','goal-audit');this.details.append(this.audit);
    this.historyTitle=el('h3','目的の変更履歴');
    panel.append(toolbar,this.note,this.cards,this.historyTitle,this.history,this.details);
    this.runs.addEventListener('change',()=>this.loadRun().catch(e=>this.showError(e)));
    this.counts.addEventListener('change',()=>{this.count=Number(this.counts.value);this.onChange();});
    this.repeats.addEventListener('change',()=>{this.repeat=Number(this.repeats.value);this.onChange();});
    refresh.addEventListener('click',()=>this.loadEpisode(this.data).catch(e=>this.showError(e)));
  }
  showError(error){this.note.textContent=`結果を読み込めません: ${error.message}`;}
  async loadEpisode(data) {
    const token=++this.token;this.data=data;this.replay=null;
    let index=[];
    try {index=await json('/data/viz/viewer/commentary/index.json');} catch(e) {
      if(!String(e.message).endsWith(': 404'))this.showError(e);
    }
    if(token!==this.token)return;
    this.index=index.filter(r=>r.episode_id===data.id);
    const selected=this.runs.value;
    this.runs.replaceChildren(...this.index.map(r=>Object.assign(el('option',`${r.label ? r.label+' / ' : ''}${r.id}${r.complete?'':'（計測途中）'}`),{value:r.id})));
    if(this.index.some(r=>r.id===selected))this.runs.value=selected;
    if(!this.index.length){this.note.textContent='この試合の目的判定結果はまだありません。';this.onChange();return;}
    await this.loadRun();
  }
  async loadRun() {
    const selected=this.index.find(r=>r.id===this.runs.value);if(!selected)return;
    const token=++this.token;
    const replay=await json(selected.url);if(token!==this.token)return;
    this.replay=replay;
    this.counts.hidden=replay.kind==='narrative';
    this.historyTitle.textContent=replay.kind==='narrative'?'A1・A2の回答履歴（各対象窓）':'目的の変更履歴';
    this.repeats.replaceChildren(...Array.from({length:replay.repeats},(_,i)=>Object.assign(el('option',`反復 ${i+1}`),{value:String(i+1)})));
    this.repeat=Math.min(this.repeat,replay.repeats);this.repeats.value=String(this.repeat);
    this.onChange();
  }
  render(now) {
    this.cards.replaceChildren();this.history.replaceChildren();this.audit.replaceChildren();
    if(!this.replay){this.note.textContent='この試合の目的判定結果はまだありません。';return;}
    const step=Math.min(Math.floor((now+1e-8)/this.data.dt),this.data.recordingLastStep ?? this.data.lastStep);
    const latest=new Map();
    for(const row of this.data.rows)if(Number(row.step)<=step && (!latest.has(row.object_id) || Number(latest.get(row.object_id).step)<Number(row.step)))latest.set(row.object_id,row);
    const inactive=[...latest.values()].filter(r=>r.alive==='0').map(r=>r.object_id);
    if(this.replay.kind==='narrative'){this.renderNarrative(now,inactive);return;}
    const state=replayState(this.replay,this.repeat,this.count,now,inactive);
    const ended=now>(this.data.recordingLastStep ?? this.data.lastStep)*this.data.dt;
    this.note.textContent=`${ended?'試合終了後の結果待ち · ':''}各APIは3秒以内のみ採用。${this.replay.complete?'計測完了':'計測途中の保存済み結果'} · 実測遅延のリプレイ`;
    for(const uid of ['A0100','A0200']) {
      const card=el('article','','goal-card'),event=state.current[uid];
      card.append(el('h3',`${uid} · 推定目的`));
      card.append(el('p',inactive.includes(uid)?'非活動':event?event.label:'まだ推定結果がありません','goal-label'));
      const timing=event?`観測 ${event.source_time_s.toFixed(1)}秒 → 到着 ${event.ready_sim_s.toFixed(2)}秒`:'40stepの履歴から判定します';
      card.append(el('small',timing));
      if(state.pending.includes(uid))card.append(el('span','新しい判定を待機中','goal-pending'));
      this.cards.append(card);
    }
    for(const e of state.history) {
      const item=el('div','','goal-event');
      item.append(el('small',`${e.ready_sim_s.toFixed(2)}秒 · ${e.aircraft_id}（観測 ${e.source_time_s.toFixed(1)}秒）`),el('p',e.label));
      this.history.append(item);
    }
    if(!state.history.length)this.history.append(el('p','目的の回答が到着すると、ここに表示します。','hint'));
    this.audit.append(el('p',this.replay.timing_note));
    if(state.latestGate)this.audit.append(el('p',`直近の検知: step ${state.latestGate.source_end_step} / ${state.latestGate.status==='late'?'3秒超過':state.latestGate.choice || state.latestGate.api_status}`));
    for(const e of state.rejected.slice().reverse())this.audit.append(el('p',`${e.event_sim_s.toFixed(2)}秒 · ${e.stage==='gate'?'検知':e.aircraft_id} · ${e.status} · 観測step ${e.source_end_step}`));
  }
  renderNarrative(now,inactive) {
    const state=narrativeState(this.replay,this.repeat,now,inactive),summary=this.replay.summary;
    this.note.textContent=`GPT-5.6 sol · A1/A2解説 · 成功 ${summary.ok_queries}/${summary.planned_queries}窓 · 応答時間は表示のみ（遅延・3秒棄却なし）`;
    for(const [uid,name] of [['A0100','A1'],['A0200','A2']]){
      const card=el('article','','goal-card narrative-card'),event=state.current[uid];
      card.append(el('h3',`${name}（${uid}）· 解説`));
      card.append(el('p',inactive.includes(uid)?'非活動':event?event.label:'この時刻までの解説はまだありません','narrative-text'));
      if(event)card.append(el('small',`観測 ${event.source_time_s.toFixed(1)}秒 / step ${event.source_end_step} · API応答 ${event.latency_s.toFixed(3)}秒（2機共通）`));
      this.cards.append(card);
    }
    for(const event of state.history){
      const item=el('div','','goal-event');
      item.append(el('small',`観測 ${event.source_time_s.toFixed(1)}秒 / step ${event.source_end_step} · ${event.aircraft_id} · API応答 ${event.latency_s.toFixed(3)}秒`),el('p',event.label));
      this.history.append(item);
    }
    if(!state.history.length)this.history.append(el('p','陽性だった観測窓の時刻から、保存済みの回答を表示します。','hint'));
    this.audit.append(el('p',this.replay.timing_note),el('p',`保存済み ${summary.recorded_queries}/${summary.planned_queries}件 · 正常回答 ${summary.answer_count}機分 · 3秒超 ${summary.over_3s}件（表示対象）`));
    if(summary.p50_s!==null)this.audit.append(el('p',`API応答時間 中央値 ${summary.p50_s.toFixed(3)}秒 / p95 ${summary.p95_s.toFixed(3)}秒`));
  }
}
