"""Read the six-component text ACMI records present in this LAG dataset.

Each row is an object update, not an interpolated full-world snapshot.
Unspecified fields remain missing; removals are returned separately.
This intentionally rejects transform layouts other than the observed six fields.
"""
from pathlib import Path
import pandas as pd

TRANSFORM_COLUMNS = [
    'longitude_deg', 'latitude_deg', 'altitude_m',
    'roll_deg', 'pitch_deg', 'yaw_deg',
]
NUMERIC_PROPERTIES = ['TAS', 'Health', 'Missiles', 'Radius']


def load_acmi(path):
    """Return (updates DataFrame, removals DataFrame, header/global metadata)."""
    path = Path(path)
    metadata, rows, removals = {}, [], []
    time_s = None
    for line_number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        if line.startswith(('FileType=', 'FileVersion=')):
            key, value = line.split('=', 1)
            metadata[key] = value
            continue
        if line.startswith('#'):
            time_s = float(line[1:])
            continue
        if time_s is None and not line.startswith('0,'):
            raise ValueError(f'{path.name}:{line_number}: object before timestamp')
        if line.startswith('-'):
            removals.append({'time_s': time_s, 'object_id': line[1:], 'line_number': line_number})
            continue
        object_id, *fields = line.split(',')
        props = {}
        for field in fields:
            if '=' not in field:
                raise ValueError(f'{path.name}:{line_number}: unsupported field {field!r}')
            key, value = field.split('=', 1)
            props[key] = value
        if object_id == '0':
            metadata.update(props)
            continue
        row = {'time_s': time_s, 'object_id': object_id, 'line_number': line_number}
        transform = props.pop('T', None)
        if transform is not None:
            values = transform.split('|')
            if len(values) != 6:
                raise ValueError(f'{path.name}:{line_number}: expected 6 transform components')
            row.update({k: float(v) if v else float('nan') for k, v in zip(TRANSFORM_COLUMNS, values)})
        row.update(props)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in NUMERIC_PROPERTIES:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
    for column in frame.select_dtypes(include=['object', 'str']).columns:
        frame[column] = frame[column].astype('string')
    removed = pd.DataFrame(removals, columns=['time_s', 'object_id', 'line_number'])
    return frame, removed, metadata
