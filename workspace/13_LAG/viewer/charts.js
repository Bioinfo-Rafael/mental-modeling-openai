import {numeric,labels,chartOrders} from './model.js';
const ns='http://www.w3.org/2000/svg';
const svgNode=(name, attrs={},text)=>{const n=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const pretty=v=>Math.abs(v)>=1000?v.toFixed(0):Math.abs(v)<0.01&&v!==0?v.toExponential(1):Number(v.toFixed(2)).toString();

export function renderProbabilities(container,row,actions) {
  container.replaceChildren();
  for(const [key,title] of [['altitude_action','高度'],['heading_action','方位'],['speed_action','速度'],['shoot_action','射撃']]) {
    const card=document.createElement('div');card.className='prob-card';
    const heading=document.createElement('h3');heading.textContent=title;card.append(heading);
    for(const label of actions[key]) {
      const field=`p_${key.replace('_action','')}_${label}`, p=numeric(row?.[field]);
      const line=document.createElement('div');line.className='prob-row'+(row?.[key]===label?' chosen':'');
      const text=document.createElement('div');text.className='prob-label';
      const name=document.createElement('span');name.textContent=(row?.[key]===label?'● ':'')+labels[label];
      const value=document.createElement('span');value.textContent=p===null?'未収録':`${(p*100).toFixed(2)}%`;value.title=row?.[field]??'';
      text.append(name,value);const track=document.createElement('div');track.className='bar-track';
      const bar=document.createElement('div');bar.className='bar';bar.style.width=`${p===null?0:p*100}%`;track.append(bar);line.append(text,track);card.append(line);
    }
    container.append(card);
  }
}

export function renderSeries(container,{fields,rows,step,dt,lastStep,windowSize,launches,onSeek}) {
  container.replaceChildren();
  if(!fields.length){const p=document.createElement('p');p.className='empty';p.textContent='「表示項目を選ぶ」からグラフを追加してください。';container.append(p);return;}
  const time=step*dt, end=lastStep*dt;
  let lo=0,hi=end;
  if(windowSize!=='all') { const span=Number(windowSize)*2;lo=Math.max(0,Math.min(time-Number(windowSize),end-span));hi=Math.min(end,lo+span); }
  const width=1100,height=142,left=78,right=20,top=16,bottom=25;
  const x=t=>left+(t-lo)/(hi-lo)*(width-left-right);
  for(const [key,title,unit] of fields) {
    const order=chartOrders[key];
    const get=r=>order?(r&&order.includes(r[key])?order.indexOf(r[key]):null):numeric(r?.[key]);
    const samples=[];
    for(let s=Math.max(0,Math.floor(lo/dt)-1);s<=Math.min(lastStep,Math.ceil(hi/dt)+1);s++)samples.push([s*dt,get(rows.get(s))]);
    const valid=samples.map(p=>p[1]).filter(v=>v!==null&&Number.isFinite(v));
    let ymin=order?0:Math.min(...valid), ymax=order?order.length-1:Math.max(...valid);
    if(!valid.length){ymin=0;ymax=1;}
    if(!order){const pad=Math.max((ymax-ymin)*.12,Math.abs(ymax)*.00001,.01);ymin-=pad;ymax+=pad;}else{ymin-=.3;ymax+=.3;}
    const y=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
    const card=document.createElement('div');card.className='chart-card';
    const heading=document.createElement('div');heading.className='chart-title';
    const name=document.createElement('strong');name.textContent=title+(unit?` · ${unit}`:'');
    const readout=document.createElement('span');const current=get(rows.get(step));
    const valueLabel=v=>v===null?'未収録':order?labels[order[v]]:pretty(v)+(unit?` ${unit}`:'');
    readout.textContent=`現在 ${valueLabel(current)}`;heading.append(name,readout);
    const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,preserveAspectRatio:'none',role:'img','aria-label':`${title}の時系列`});
    const ticks=order?order.map((_,i)=>i):[ymin,(ymin+ymax)/2,ymax];
    for(const value of ticks){svg.append(svgNode('line',{x1:left,x2:width-right,y1:y(value),y2:y(value),stroke:'#e9eef2'}));svg.append(svgNode('text',{x:left-8,y:y(value)+3,'text-anchor':'end',fill:'#738394','font-size':10},order?labels[order[value]]:pretty(value)));}
    for(let i=0;i<=5;i++){const t=lo+(hi-lo)*i/5;svg.append(svgNode('line',{x1:x(t),x2:x(t),y1:top,y2:height-bottom,stroke:'#eef2f5'}));svg.append(svgNode('text',{x:x(t),y:height-7,'text-anchor':'middle',fill:'#738394','font-size':10},`${t.toFixed(1)} s`));}
    // Clip paths at the time-window boundary, including held categorical intervals.
    const clipId=`clip-${key}`;const defs=svgNode('defs'),clip=svgNode('clipPath',{id:clipId});clip.append(svgNode('rect',{x:left,y:top,width:width-left-right,height:height-top-bottom}));defs.append(clip);svg.append(defs);
    const plot=svgNode('g',{'clip-path':`url(#${clipId})`});
    let d='',prev=null;
    for(const [t,v] of samples) {
      if(v===null||!Number.isFinite(v)){prev=null;continue;}
      if(order){d+=`M${x(t)},${y(v)}H${x(t+dt)}`;if(prev!==null)d+=`M${x(t)},${y(prev)}V${y(v)}`;}
      else d+=`${prev===null?'M':'L'}${x(t)},${y(v)}`;
      prev=v;
    }
    plot.append(svgNode('path',{d,fill:'none',stroke:'#16858a','stroke-width':1.8,'vector-effect':'non-scaling-stroke'}));
    for(const event of launches)for(const [s,color,dash] of [[event.action_step,'#cc7423','4 3'],[event.first_acmi_step,'#8860c3','2 3']]) {
      const marker=svgNode('line',{x1:x(s*dt),x2:x(s*dt),y1:top,y2:height-bottom,stroke:color,'stroke-dasharray':dash,'stroke-width':1.5});
      marker.append(svgNode('title',{},`${event.missile_id}: ${s===event.action_step?'発射結果':'ACMI初出'} step ${s}`));plot.append(marker);
    }
    plot.append(svgNode('line',{x1:x(time),x2:x(time),y1:top,y2:height-bottom,stroke:'#203746','stroke-width':1.2}));
    if(current!==null)plot.append(svgNode('circle',{cx:x(time),cy:y(current),r:3.2,fill:'#16858a',stroke:'white','stroke-width':1}));
    svg.append(plot);
    if(!valid.length)svg.append(svgNode('text',{x:width/2,y:height/2,'text-anchor':'middle',fill:'#8a99a5','font-size':12},'この範囲には記録がありません'));
    const stepAt=e=>{const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*width;return Math.max(0,Math.min(lastStep,Math.round((lo+(Math.max(left,Math.min(width-right,px))-left)/(width-left-right)*(hi-lo))/dt)));};
    svg.addEventListener('click',e=>onSeek(stepAt(e)));
    svg.addEventListener('pointermove',e=>{const s=stepAt(e);readout.textContent=`step ${s} / ${(s*dt).toFixed(1)} s · ${valueLabel(get(rows.get(s)))}`;});
    svg.addEventListener('pointerleave',()=>{readout.textContent=`現在 ${valueLabel(current)}`;});
    card.append(heading,svg);container.append(card);
  }
}
