// Pure replay state: only events available by the current simulation clock.
export function replayState(replay, repeat, count, now, inactive = []) {
  const eligible = (replay?.events || []).filter(e => e.repeat === repeat &&
    (e.choice_count === null || e.choice_count === count));
  const visible = eligible.filter(e => e.event_sim_s <= now + 1e-9);
  const current = {}, history = [];
  for (const e of visible) {
    if (e.status === 'displayed' || e.status === 'same_goal') {
      current[e.aircraft_id] = e;
      if (e.status === 'displayed') history.push(e);
    }
  }
  for (const uid of inactive) delete current[uid];
  // Completed durations must not leak the answer before receipt.
  const pending = eligible.filter(e => e.stage === 'goal' &&
    e.send_sim_s <= now && e.event_sim_s > now).map(e => e.aircraft_id);
  return {current, history: history.reverse(), pending,
    rejected: visible.filter(e => !['displayed','same_goal','gate_trigger','gate_none'].includes(e.status)),
    latestGate: visible.filter(e => e.stage === 'gate').at(-1)};
}

// Prose comparison: observed step controls display; measured API latency is metadata only.
export function narrativeState(replay, repeat, now, inactive = []) {
  const history=(replay?.events || []).filter(e=>e.repeat===repeat && e.source_time_s<=now+1e-9);
  const current={};
  for(const event of history)current[event.aircraft_id]=event;
  for(const uid of inactive)delete current[uid];
  return {current,history:history.slice().reverse()};
}
