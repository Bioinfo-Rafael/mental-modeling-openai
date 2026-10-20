export const stateFields = [
  ['lon_deg','経度','°'], ['lat_deg','緯度','°'], ['alt_m','高度','m'],
  ['roll_deg','ロール','°'], ['pitch_deg','ピッチ','°'], ['yaw_deg','機首方位（ヨー）','°'],
  ['tas_mps','真対気速度','m/s'], ['vn_mps','北方向速度','m/s'], ['ve_mps','東方向速度','m/s'], ['vu_mps','上方向速度','m/s'],
  ['alive','生存フラグ','0 / 1'], ['health','Health',''], ['missiles_remaining','残弾数','発'],
];
export const actionFields = [
  ['altitude_action','高度指令',''], ['heading_action','方位指令',''], ['speed_action','速度指令',''], ['shoot_action','発射要求',''],
];
export const derivedFields = [['yaw_rate','機首方位の変化率（計算）','°/s'], ['course_rate','水平進行方向の変化率（計算）','°/s']];
export const defaults = ['roll_deg','yaw_rate','course_rate','heading_action','shoot_action'];
export const labels = {up:'上げる',hold:'維持',down:'下げる',left30:'左30°',left15:'左15°',right15:'右15°',right30:'右30°',no:'なし',yes:'要求あり'};
export const chartOrders = {altitude_action:['down','hold','up'],heading_action:['left30','left15','hold','right15','right30'],speed_action:['down','hold','up'],shoot_action:['no','yes']};
export function numeric(value) { return value === '' || value === null || value === undefined ? null : Number(value); }
export function wrap(value) { return ((value + 180) % 360 + 360) % 360 - 180; }
export function derived(row, prev, dt) {
  const out = {yaw_rate:null,course_rate:null};
  if (!row || !prev || Number(row.step) - Number(prev.step) !== 1 || row.alive !== '1' || prev.alive !== '1') return out;
  const a=numeric(row.yaw_deg), b=numeric(prev.yaw_deg);
  if(a!==null && b!==null) out.yaw_rate = wrap(a-b)/dt;
  const velocity = r => {
    const n=numeric(r.vn_mps), e=numeric(r.ve_mps);
    return n===null || e===null || Math.hypot(n,e)<1e-6 ? null : Math.atan2(e,n)*180/Math.PI;
  };
  const c=velocity(row), p=velocity(prev);
  if(c!==null && p!==null) out.course_rate=wrap(c-p)/dt;
  return out;
}
export function videoFrame(step, dt, sample, scale, duration) {
  // Seek inside the frame, never exactly on a floating-point frame boundary.
  const index=Math.floor((step*dt + 1e-8)/sample);
  const last=Number.isFinite(duration) ? Math.max(0,Math.ceil(duration/(sample/scale)-1e-6)-1) : index;
  const frame=Math.min(index,last);
  return {index:frame, simulationTime:frame*sample, mediaTime:(frame*sample+sample/2)/scale};
}
