"""Exact nested observation windows; labels remain outside messages."""
from copy import deepcopy
from .common import records


def build_samples(frame, annotations, config):
    if frame.duplicated(['step', 'object_id']).any():
        raise ValueError('Duplicate step/object key')
    blocks = records.build_blocks(frame)
    samples, excluded = {}, []
    for event in annotations['events']:
        t = event['step']
        if type(t) is not int or not isinstance(event['id'], str):
            raise ValueError('Invalid annotation identity')
        first, last = t - config['history_before'], t + config['horizon']
        missing = [step for step in range(first, last + 1) if step not in blocks]
        if missing:
            excluded.append(dict(event=event, reason='incomplete_common_reference_window',
                                 required_start=first, required_end=last, missing_steps=missing))
            continue
        for n in config['offsets']:
            cutoff = t + n
            window = deepcopy(records.window(blocks, cutoff, cutoff - first + 1))
            window.update(target_start_step=t, target_end_step=last,
                          target_start_time_s=round(t * .2, 1), target_end_time_s=round(last * .2, 1))
            for block in window['blocks']:
                block['relative_step'] = block['step'] - cutoff
            sid = f"{event['id']}_n{n:+03d}"
            if sid in samples:
                raise ValueError('Duplicate event/sample ID')
            samples[sid] = dict(
                episode_id=annotations['episode_id'], target_id='all', task='foresight',
                observation_scope='all_aircraft_missiles', H=len(window['blocks']),
                sample_number=len(samples), start=first, target_step=t,
                event=deepcopy(event), offset=n,
                condition_kind='hindsight_reference' if n == config['horizon'] else 'forecast',
                target_interval=[t, last], observation_cutoff=cutoff,
                other_events_in_target=[e for e in annotations['events'] if e['id'] != event['id']
                                        and t <= e['step'] <= last],
                window=window)
    if not samples:
        raise ValueError('No complete events shared across all conditions')
    return samples, excluded
