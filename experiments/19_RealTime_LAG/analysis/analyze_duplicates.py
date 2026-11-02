"""Audit recorded decisions without API calls or changes to replay policy."""
import argparse
from collections import Counter
import csv
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
w = import_module('experiments.19_RealTime_LAG.realtime_experiment.workflow')


def analyze(directory):
    manifest = w.verify(directory)
    records = w.records_for(directory)
    catalog = w.read_json(directory / 'choices/catalog.json')
    replay = w.build_replay(manifest, records, catalog)
    events = replay['events']
    gates = sorted((e for e in events if e['stage'] == 'gate'),
                   key=lambda e: (e['repeat'], e['source_end_step']))
    goals = [e for e in events if e['stage'] == 'goal']
    by_gate = {(e['repeat'], e['source_end_step']): e for e in gates}
    streaks = []
    for repeat in range(1, manifest['repeats'] + 1):
        streak = []
        for e in [g for g in gates if g['repeat'] == repeat]:
            if e['status'] == 'gate_trigger':
                streak.append(e['source_end_step'])
            elif streak:
                streaks.append({'repeat': repeat, 'steps': streak})
                streak = []
        if streak:
            streaks.append({'repeat': repeat, 'steps': streak})

    result = {
        'run_id': manifest['run_id'], 'complete': replay['complete'],
        'manifest_sha256': w.storage.file_hash(directory / 'manifest.json'),
        'responses_digest': w.digest(records),
        'definitions': {
            'same_goal': 'In arrival order, equals latest accepted choice for same repeat/K/aircraft; stale excluded.',
            'displayed': 'First accepted goal or a changed accepted label; includes two initial labels per condition.',
            'stale': 'Older source window arrives after a newer accepted window for same repeat/K/aircraft.',
            'short_return': 'Three consecutive displayed labels X,Y,X within 3 simulated seconds, counting overlapping triples.',
            'pending_overlap': 'At simulated goal send time an earlier goal of same repeat/K/aircraft is still pending; not observed live concurrency.',
            'accuracy': 'Not measured: no human event IDs or goal truth labels.',
        },
        'gate_choices': dict(Counter(e['choice'] for e in gates)),
        'positive_streaks': streaks, 'goal_statuses': dict(Counter(e['status'] for e in goals)),
        'conditions': {}, 'short_returns': [],
    }
    unflagged = [e for e in goals if by_gate[(e['repeat'], e['source_end_step'])]['choice'] ==
                 {'A0100': 'A2', 'A0200': 'A1'}[e['aircraft_id']]]
    result['goals_for_unflagged_aircraft'] = dict(Counter(e['status'] for e in unflagged))
    for k in (4, 8, 32):
        subset = [e for e in goals if e['choice_count'] == k]
        group = {'total': len(subset), 'statuses': dict(Counter(e['status'] for e in subset)),
                 'aircraft': {}, 'simulated_pending_overlap_count': 0}
        for repeat in range(1, manifest['repeats'] + 1):
            for uid in w.UIDS:
                series = [e for e in subset if e['repeat'] == repeat and e['aircraft_id'] == uid]
                displayed = [e for e in series if e['status'] == 'displayed']
                group['aircraft'][f'r{repeat:02d}_{uid}'] = dict(Counter(e['status'] for e in series))
                chronological = sorted(series, key=lambda e: (e['send_sim_s'], e['id']))
                group['simulated_pending_overlap_count'] += sum(
                    any(old['event_sim_s'] > e['send_sim_s'] for old in chronological[:i])
                    for i, e in enumerate(chronological))
                for a, b, c in zip(displayed, displayed[1:], displayed[2:]):
                    if a['choice'] == c['choice'] and c['event_sim_s'] - a['event_sim_s'] <= 3:
                        result['short_returns'].append({
                            'choice_count': k, 'repeat': repeat, 'aircraft_id': uid,
                            'query_ids': [v['id'] for v in (a, b, c)],
                            'choices': [v['choice'] for v in (a, b, c)],
                            'labels': [v['label'] for v in (a, b, c)],
                            'source_steps': [v['source_end_step'] for v in (a, b, c)],
                            'arrival_times_s': [v['event_sim_s'] for v in (a, b, c)],
                            'span_s': c['event_sim_s'] - a['event_sim_s']})
        group['short_return_count'] = sum(e['choice_count'] == k for e in result['short_returns'])
        result['conditions'][str(k)] = group
    result['cross_condition_broad_agreement'] = {}
    broad = {(e['repeat'], e['source_end_step'], e['aircraft_id']): e['choice']
             for e in goals if e['choice_count'] == 4 and e['api_status'] == 'ok' and e['choice']}
    for k in (8, 32):
        pairs = [(broad[(e['repeat'], e['source_end_step'], e['aircraft_id'])], e['choice'][0])
                 for e in goals if e['choice_count'] == k and e['api_status'] == 'ok' and e['choice']
                 and (e['repeat'], e['source_end_step'], e['aircraft_id']) in broad]
        result['cross_condition_broad_agreement'][str(k)] = {
            'pairs': len(pairs), 'agree': sum(a == b for a, b in pairs)}

    out = directory / 'report/duplicate_audit'
    out.mkdir(exist_ok=True)
    w.write_json(out / 'summary.json', result)
    fields = ['id', 'repeat', 'source_end_step', 'source_time_s', 'stage', 'aircraft_id',
              'choice_count', 'choice', 'label', 'status', 'send_sim_s', 'event_sim_s', 'latency_s']
    with (out / 'events.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(events)
    with (out / 'by_window.csv').open('w', newline='') as f:
        fields = ['repeat', 'step', 'time_s', 'gate'] + [f'{uid}_k{k}' for uid in w.UIDS for k in (4, 8, 32)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for g in gates:
            row = {'repeat': g['repeat'], 'step': g['source_end_step'], 'time_s': g['source_time_s'], 'gate': g['choice']}
            for e in goals:
                if (e['repeat'], e['source_end_step']) == (g['repeat'], g['source_end_step']):
                    row[f"{e['aircraft_id']}_k{e['choice_count']}"] = f"{e['choice']} | {e['label']} | {e['status']}"
            writer.writerow(row)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    analyze(w.resolve_run(args.run))
