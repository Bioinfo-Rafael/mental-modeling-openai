"""Select exact recorded windows without fabricating actions or ammunition."""
import pandas as pd
from lag_data.definitions import STATE_FEATURES, AIRCRAFT
from .config import ACTION_FIELDS


def nullable(value):
    return None if pd.isna(value) else value


def build_window(frame, center):
    if type(center) is not int or center < 30:
        raise ValueError('Center must admit 30 preceding steps')
    steps = list(range(center - 30, center + 31))
    selected = frame[frame.step.isin(steps) & frame.entity_type.isin(('aircraft', 'missile'))]
    if selected.duplicated(['step', 'object_id']).any():
        raise ValueError('Duplicate step/object key')
    blocks = []
    for step in steps:
        rows = selected[selected.step.eq(step)].sort_values(['entity_type', 'object_id'])
        aircraft = rows[rows.entity_type.eq('aircraft')]
        if set(aircraft.object_id) != set(AIRCRAFT) or not aircraft.has_csv_record.eq(1).all():
            raise ValueError(f'Incomplete CSV time point: {step}')
        entities = []
        for _, row in rows.iterrows():
            obj = {key: nullable(row[key]) for key in (
                'object_id', 'entity_type', 'team', 'launcher_id', 'alive',
                'state_valid', 'acmi_event', 'action_status')}
            obj['state'] = [nullable(row[f]) if row.state_valid == 1 else None for f in STATE_FEATURES]
            if row.entity_type == 'aircraft':
                obj['action'] = [nullable(row[f + '_id']) if row.action_status == 'recorded' else None for f in ACTION_FIELDS]
                obj['missiles_remaining'] = nullable(row.missiles_remaining)
            entities.append(obj)
        blocks.append(dict(step=step, time_s=round(step * .2, 1), relative_step=step-center, entities=entities))
    return dict(episode_alias='episode_01', center_step=center, center_time_s=round(center*.2, 1),
                start_step=steps[0], end_step=steps[-1], start_time_s=round(steps[0]*.2, 1),
                end_time_s=round(steps[-1]*.2, 1), blocks=blocks)
