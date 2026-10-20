import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
// Import buildless browser modules without changing the repository package type.
const source=readFileSync(new URL('./model.js',import.meta.url),'utf8');
const {videoFrame,derived,numeric}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
test('0.2s data steps select 0.4s recorded frames, including end clamp',()=>{
  assert.equal(videoFrame(46,.2,.4,4,17.1).simulationTime,9.200000000000001);
  assert.equal(videoFrame(47,.2,.4,4,17.1).index,23);
  assert.equal(videoFrame(48,.2,.4,4,17.1).index,24);
  assert.equal(videoFrame(341,.2,.4,4,17.1).index,170);
  assert.ok(videoFrame(341,.2,.4,4,17.1).mediaTime<17.1);
});
test('angular velocity crosses north without a 360-degree spike',()=>{
  const a={step:'0',alive:'1',yaw_deg:'359',vn_mps:'100',ve_mps:'0'};
  const b={step:'1',alive:'1',yaw_deg:'1',vn_mps:'100',ve_mps:'0'};
  assert.equal(derived(b,a,.2).yaw_rate,10);
  assert.equal(derived(b,a,.2).course_rate,0);
  assert.equal(derived({...b,step:'2'},a,.2).yaw_rate,null);
  assert.equal(derived({...b,alive:'0'},a,.2).yaw_rate,null);
  assert.equal(numeric(''),null);
  assert.equal(numeric('0'),0);
});

test('rapid seek back cancels a pending newer frame; terminal step clamps',async()=>{
  const pending=[];
  globalThis.Image=class {set src(value){this.url=value;pending.push(this);}};
  const modelURL='data:text/javascript;base64,'+Buffer.from(source).toString('base64');
  const playbackSource=readFileSync(new URL('./playback.js',import.meta.url),'utf8').replace("'./model.js'",JSON.stringify(modelURL));
  const {Playback}=await import('data:text/javascript;base64,'+Buffer.from(playbackSource).toString('base64'));
  const image={removeAttribute(){},src:''};
  const playback=new Playback(image,()=>{},()=>{},()=>{});
  playback.load({lastStep:341,dt:.2,videoSampleSeconds:.4,videoTimeScale:4,frameCount:171,frameBase:'/frames/'});
  pending.shift().onload();
  assert.equal(image.src,'/frames/00000.jpg');
  playback.seek(4);
  const obsolete=pending.shift();
  playback.seek(0);
  obsolete.onload();
  assert.equal(image.src,'/frames/00000.jpg');
  playback.seek(999);
  pending.shift().onload();
  assert.equal(playback.step,341);
  assert.equal(image.src,'/frames/00170.jpg');
  playback.seek(-1);
  pending.shift().onload();
  assert.equal(playback.step,0);
});
