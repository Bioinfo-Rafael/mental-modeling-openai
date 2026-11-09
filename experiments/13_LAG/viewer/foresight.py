"""Export saved sol answers and their matching translations for local replay."""
import json
from pathlib import Path

EXPERIMENTS = (
    ('exp20_foresight_short_v1', '20_foresight_LAG', 'foresight_short_v1'),
    ('exp20_targets_v1', '20_foresight_LAG', 'targets_v1'),
    ('exp21_foresight_LAG2', '21_foresight_LAG2', 'sliding_n1_v1'),
)
EPISODE = 'mhi_v1.3_ic100601_win'


def build_foresight(experiments_root: Path):
    runs = []
    for label, experiment, run in EXPERIMENTS:
        report = experiments_root / experiment / 'results' / run / 'report'
        responses_path = report / 'responses.json'
        translations_path = report / 'translations_ja.json'
        responses = json.loads(responses_path.read_text()) if responses_path.exists() else []
        translations = json.loads(translations_path.read_text()) if translations_path.exists() else {}
        translated = {
            (row['query_id'], row.get('aircraft_id', row.get('aircraft'))): row
            for row in translations.get('records', translations.get('entries', []))
        }
        events = []
        identities = set()
        for response in responses:
            if response['model'] != 'gpt-5.6-sol':
                continue
            # Exp20 uses the annotated event; exp21 uses the input window end.
            step = (response['event']['step'] if 'event' in response
                    else response['observation_range'][1])
            identity = (response['offset'], step)
            if identity in identities:
                raise ValueError(f'Duplicate sol result in {label}: {identity}')
            identities.add(identity)
            english = response.get('narratives') or {}
            japanese = {}
            for aircraft, text in english.items():
                translation = translated.get((response['query_id'], aircraft))
                # Never attach a stale translation to a revised answer.
                if translation and translation['english'] == text:
                    japanese[aircraft] = translation['japanese']
            events.append({
                'step': step, 'offset': response['offset'],
                'query_id': response['query_id'], 'status': response['status'],
                'observation_range': response['observation_range'],
                'target_interval': response['target_interval'],
                'en': english, 'ja': japanese,
            })
        runs.append({'id': label, 'episode_id': EPISODE,
                     'offsets': sorted({e['offset'] for e in events}),
                     'events': sorted(events, key=lambda e: (e['offset'], e['step']))})
    return runs
