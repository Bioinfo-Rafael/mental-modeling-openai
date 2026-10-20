"""Build disposable viewer data from validated, unmodified episode CSVs."""
import csv
import importlib
import json
from pathlib import Path
import sys
import hashlib
import subprocess
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
lag = importlib.import_module('experiments.13_LAG.lag_data.dataset')
definitions = importlib.import_module('experiments.13_LAG.lag_data.definitions')
OUT = ROOT / 'data/viz/viewer'


def prepare_frames(movie, eid):
    """Decode each recorded frame once; cached by source hash."""
    digest = hashlib.sha256(movie.read_bytes()).hexdigest()
    target = OUT / 'frames' / eid
    metadata_path = target / 'frames.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['sha256'] == digest and all((target / f'{i:05d}.jpg').exists() for i in range(metadata['count'])):
            return metadata
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg and ffprobe are needed to prepare video frames.')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate,nb_frames', '-of', 'json', str(movie)]))['streams'][0]
    if probe['r_frame_rate'] != '10/1':
        raise ValueError(f'Expected supplied 10fps recording: {movie}')
    count = int(probe['nb_frames'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        pending = Path(temporary)
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(movie), '-q:v', '2', '-start_number', '0', str(pending / '%05d.jpg')], check=True)
        if len(list(pending.glob('*.jpg'))) != count:
            raise ValueError('Decoded frame count differs from recording')
        metadata = {'sha256': digest, 'count': count}
        (pending / 'frames.json').write_text(json.dumps(metadata))
        target.mkdir(exist_ok=True)
        for source in pending.iterdir():
            source.replace(target / source.name)
    return metadata


def prepare():
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []
    for episode in manifest['episodes']:
        path = ROOT / 'data/processed' / episode['output']['path']
        frame = lag.load_episode(path)  # schema, hash and unique-key verification
        with path.open(newline='') as handle:
            # Keep original numeric strings for the full-precision inspector.
            rows = [r for r in csv.DictReader(handle) if r['entity_type'] == 'aircraft']
        assert len(rows) == int((frame.entity_type == 'aircraft').sum())
        eid = episode['episode_id']
        movie = ROOT / 'data/raw/20261005_LAG_2v2_ACMI/recordings' / f'{eid}.mp4'
        if not movie.exists():
            raise FileNotFoundError(movie)
        frames = prepare_frames(movie, eid)
        title = ('MHI v1.3' if eid.startswith('mhi') else 'Ours · seed 100100')
        title += ' / IC ' + ('100606' if '100606' in eid else '100601')
        title += ' / ' + ('WIN' if eid.endswith('_win') else 'LOSS')
        item = {'id': eid, 'title': title, 'lastStep': episode['acmi_last_step'],
                'video': '/' + str(movie.relative_to(ROOT)),
                'data': f'/data/viz/viewer/{eid}.json'}
        payload = {**item, 'dt': float(definitions.DT), 'videoSampleSeconds': 0.4,
                   'videoTimeScale': 4, 'frameCount': frames['count'],
                   'frameBase': f'/data/viz/viewer/frames/{eid}/', 'aircraft': definitions.AIRCRAFT,
                   'actions': definitions.ACTION_LABELS, 'columns': list(frame.columns),
                   'launches': episode['launches'], 'rows': rows,
                   'sourceSha256': frame.attrs['csv_sha256']}
        (OUT / f'{eid}.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        catalog.append(item)
    (OUT / 'index.json').write_text(json.dumps(catalog, ensure_ascii=False), encoding='utf-8')
    return catalog


if __name__ == '__main__':
    print(f'Prepared {len(prepare())} episodes in {OUT}')
