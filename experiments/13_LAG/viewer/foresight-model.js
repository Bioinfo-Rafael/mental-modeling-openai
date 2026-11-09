// The first answer also covers the lead-in; subsequent answers start at their step.
export function foresightState(run, offset, step, lastStep) {
  const events=(run?.events || []).filter(event=>event.offset===offset).sort((a,b)=>a.step-b.step);
  if(!events.length)return null;
  let index=0;
  for(let i=1;i<events.length && events[i].step<=step;i++)index=i;
  const event=events[index];
  return {event,start:index===0?0:event.step,end:index+1<events.length?events[index+1].step-1:lastStep,
    highlighted:step===event.step};
}
