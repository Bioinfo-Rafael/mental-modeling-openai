import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('./foresight-model.js',import.meta.url),'utf8');
const {foresightState}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const run={events:[46,49,83,95,143,243].flatMap(step=>[-1,1,30].map(offset=>({step,offset})))};
test('lead-in, boundaries, final hold and exact one-step emphasis',()=>{
  for(const [step,expected,start,end,highlighted] of [
    [0,46,0,48,false],[45,46,0,48,false],[46,46,0,48,true],
    [47,46,0,48,false],[48,46,0,48,false],[49,49,49,82,true],
    [50,49,49,82,false],[82,49,49,82,false],[83,83,83,94,true],
    [243,243,243,343,true],[343,243,243,343,false],
  ]){
    const state=foresightState(run,30,step,343);
    assert.deepEqual([state.event.step,state.start,state.end,state.highlighted],[expected,start,end,highlighted]);
  }
});
test('seek backwards and changing conditions select the matching answer',()=>{
  assert.equal(foresightState(run,30,100,343).event.step,95);
  assert.equal(foresightState(run,1,48,343).event.step,46);
  assert.equal(foresightState(run,-1,49,343).event.offset,-1);
  assert.equal(foresightState(null,30,0,343),null);
  assert.equal(foresightState({events:[{step:30,offset:1}]},30,0,343),null);
});
test('sliding windows use input-end steps and preserve the first lead-in',()=>{
  const sliding={events:[30,45,60].map(step=>({step,offset:1}))};
  assert.equal(foresightState(sliding,1,0,343).event.step,30);
  assert.equal(foresightState(sliding,1,30,343).highlighted,true);
  assert.equal(foresightState(sliding,1,44,343).event.step,30);
  assert.equal(foresightState(sliding,1,45,343).event.step,45);
});
