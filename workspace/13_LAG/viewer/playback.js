import {videoFrame} from './model.js';

export class Playback {
  constructor(video, onStep, onPlay, onVideo) {
    Object.assign(this,{video,onStep,onPlay,onVideo,step:0,speed:1,playing:false,token:0,desired:null,shown:null});
  }
  load(data) {
    this.pause(); this.data=data; this.step=0; this.shown=null; this.desired=null; this.seekingFrame=null;
    this.frameRequest=(this.frameRequest||0)+1; this.video.removeAttribute("src"); this.onStep(0); this.syncVideo();
  }
  setStep(value) {
    if(!this.data || !Number.isFinite(Number(value))) return;
    this.step=Math.max(0,Math.min(this.data.lastStep,Math.round(Number(value))));
    this.onStep(this.step); this.syncVideo();
  }
  seek(value) { this.pause(); this.setStep(value); }
  setSpeed(value) { const was=this.playing; this.pause(); this.speed=Number(value); if(was)this.play(); }
  toggle() { this.playing ? this.pause() : this.play(); }
  pause() { this.playing=false; this.token++; clearTimeout(this.timer); this.onPlay(false); }
  play() {
    if(!this.data || this.playing)return;
    if(this.step>=this.data.lastStep)this.setStep(0);
    this.playing=true; this.onPlay(true);
    const token=++this.token, start=performance.now(), first=this.step;
    const tick=()=>{
      const now=performance.now();
      if(token!==this.token)return;
      const target=Math.min(this.data.lastStep,first+Math.floor((now-start)*this.speed/(1000*this.data.dt)));
      if(target!==this.step)this.setStep(target);
      if(this.step>=this.data.lastStep){this.pause();return;}
      this.timer=setTimeout(tick,25);
    };
    this.timer=setTimeout(tick,25);
  }
  reportVideo() {
    this.onVideo(this.shown, !this.shown || this.shown.index!==this.desired?.index);
  }
  syncVideo() {
    if(!this.data)return;
    this.desired=videoFrame(this.step,this.data.dt,this.data.videoSampleSeconds,this.data.videoTimeScale,this.data.frameCount*.1);
    this.reportVideo();
    const request=++this.frameRequest;
    if(this.shown?.index===this.desired.index)return;
    const target={...this.desired};
    const src=this.data.frameBase+String(target.index).padStart(5,'0')+'.jpg';
    const image=new Image();
    image.onload=()=>{
      if(request!==this.frameRequest)return;
      this.video.src=src; this.shown=target; this.reportVideo();
    };
    image.onerror=()=>{if(request===this.frameRequest)this.onVideo(this.shown,true,'録画フレームを読み込めません。サーバーを確認してください。');};
    image.src=src;
  }
}
