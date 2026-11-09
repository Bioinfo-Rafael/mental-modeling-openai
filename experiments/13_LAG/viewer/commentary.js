import {foresightState} from './foresight-model.js?v=foresight-2';
const el=(tag,text='',cls='')=>Object.assign(document.createElement(tag),{textContent:text,className:cls});
const choices=(label,items,value)=>{
  const select=el('select');select.setAttribute('aria-label',label);
  for(const [key,text] of items)select.append(Object.assign(el('option',text),{value:String(key)}));
  select.value=String(value);return select;
};
export class Commentary {
  constructor(panel,onChange) {
    this.onChange=onChange;this.token=0;this.index=[];this.error='';
    const ids=['exp20_foresight_short_v1','exp20_targets_v1','exp21_foresight_LAG2'];
    this.runs=choices('実験',ids.map(id=>[id,id]),ids[0]);
    this.languages=choices('言語',[['ja','日本語'],['en','English']],'ja');
    this.teams=choices('チーム',[['A','A'],['B','B']],'A');
    this.offsets=choices('条件',[[-1,'n=-1'],[1,'n=1'],[30,'n=30']],30);
    const toolbar=el('div','','goal-toolbar');
    for(const [title,select] of [['実験',this.runs],['言語',this.languages],['チーム',this.teams],['条件',this.offsets]]){
      const label=el('label',title);label.append(select);toolbar.append(label);
      select.addEventListener('change',()=>this.onChange());
    }
    this.note=el('p','','goal-note');this.cards=el('div','','goal-cards');
    panel.append(toolbar,this.note,this.cards);
  }
  async loadEpisode(data) {
    const token=++this.token;this.data=data;this.index=[];this.error='';this.loading=true;
    this.onChange();
    try {
      const response=await fetch('/data/viz/viewer/foresight.json',{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const index=await response.json();
      if(token!==this.token)return;
      this.index=index;
    }catch(error){if(token!==this.token)return;this.error=`結果を読み込めません: ${error.message}`;}
    this.loading=false;this.onChange();
  }
  render(now) {
    this.cards.replaceChildren();
    if(!this.data){this.note.textContent='結果を読み込み中…';return;}
    const run=this.index.find(item=>item.id===this.runs.value && item.episode_id===this.data?.id);
    const step=Math.round(now/this.data.dt),offset=Number(this.offsets.value);
    const state=foresightState(run,offset,step,this.data.recordingLastStep ?? this.data.lastStep);
    const english=this.languages.value==='en';
    const empty=this.error || (this.loading?'結果を読み込み中…':!run?'この試合の保存済み結果はありません。':
      `この実験には n=${offset} の結果がありません。保存済み条件: ${run.offsets.map(n=>`n=${n}`).join(', ') || 'なし'}`);
    this.note.textContent=state?
      `gpt-5.6-sol · 表示区間 step ${state.start}–${state.end} · 最初の回答はstep 0から表示` : empty;
    for(const suffix of ['0100','0200']){
      const uid=this.teams.value+suffix;
      const card=el('article','',`goal-card narrative-card${state?.highlighted?' output-step':''}`);
      card.append(el('h3',`${uid} · ${english?'Estimated intent':'推定目的'}`));
      const text=state?.event[this.languages.value]?.[uid];
      card.append(el('p',text || (state?(english?'No saved answer.':'この回答の日本語訳はありません。'):empty),'narrative-text'));
      if(state){
        const event=state.event;
        card.append(el('small',`step ${event.step} · 入力 ${event.observation_range.join('–')} · 対象 ${event.target_interval.join('–')}`));
        const badge=el('small',english?'Output at this step':'このstepで出力しました','output-badge');
        badge.style.visibility=state.highlighted?'visible':'hidden';card.append(badge);
      }
      this.cards.append(card);
    }
  }
}
