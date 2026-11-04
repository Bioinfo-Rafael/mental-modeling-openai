import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('./commentary-model.js',import.meta.url),'utf8');
const {replayState,narrativeState}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const event=(id,time,status,choice='A',step=50)=>({id,repeat:1,choice_count:4,stage:'goal',aircraft_id:'A0100',event_sim_s:time,send_sim_s:10,ready_sim_s:time,source_end_step:step,status,choice,label:choice});
test('arrival time, not observation time, controls display; rewind reconstructs state',()=>{
  const replay={events:[event('a',12,'displayed'),event('b',13,'same_goal','A',55),event('old',14,'stale','B',49),event('c',15,'displayed','B',60)]};
  assert.equal(replayState(replay,1,4,11).current.A0100,undefined);
  assert.ok(replayState(replay,1,4,11).pending.includes('A0100'));
  assert.equal(replayState(replay,1,4,12).current.A0100.id,'a');
  assert.equal(replayState(replay,1,4,14).current.A0100.id,'b');
  assert.equal(replayState(replay,1,4,14).history.length,1);
  assert.equal(replayState(replay,1,4,15).history.length,2);
  assert.equal(replayState(replay,1,4,12).current.A0100.id,'a');
  assert.equal(replayState(replay,1,8,15).history.length,0);
  assert.equal(replayState(replay,2,4,15).history.length,0);
  assert.equal(replayState(replay,1,4,15,['A0100']).current.A0100,undefined);
});
test('late counterpart never erases accepted aircraft and expires only at deadline',()=>{
  const replay={events:[event('a',12,'displayed'),{...event('b',13,'late'),aircraft_id:'A0200'}]};
  assert.equal(replayState(replay,1,4,12).rejected.length,0);
  const result=replayState(replay,1,4,13);
  assert.equal(result.current.A0100.id,'a');assert.equal(result.current.A0200,undefined);
  assert.equal(result.rejected.length,1);
});
test('narrative shows both aircraft at observation time regardless of latency, retaining repeated answers',()=>{
  const events=[
    {id:'a',repeat:1,source_time_s:10,aircraft_id:'A0100',label:'same',latency_s:20},
    {id:'b',repeat:1,source_time_s:10,aircraft_id:'A0200',label:'second',latency_s:20},
    {id:'c',repeat:1,source_time_s:11,aircraft_id:'A0100',label:'same',latency_s:1},
    {id:'d',repeat:2,source_time_s:10,aircraft_id:'A0100',label:'other',latency_s:1}];
  assert.equal(narrativeState({events},1,9.8).history.length,0);
  const at10=narrativeState({events},1,10);
  assert.equal(at10.current.A0100.id,'a');assert.equal(at10.current.A0200.id,'b');
  assert.equal(narrativeState({events},1,11).history.length,3);
  assert.equal(narrativeState({events},1,10).current.A0100.id,'a');
  assert.equal(narrativeState({events},1,11,['A0100']).current.A0100,undefined);
});
