"""Production English prompts; template files are the runtime source of truth."""
import json
from pathlib import Path
PROMPT_VERSION = 'lag-commentary-v2'
HERE = Path(__file__).resolve().parent


def build_messages(window, variant='commentary'):
    if variant not in ('commentary', 'tactical'):
        raise ValueError('Unknown prompt variant')
    data = '\n'.join(json.dumps(block, ensure_ascii=False, allow_nan=False, separators=(',', ':')) for block in window['blocks'])
    fields = {k: v for k, v in window.items() if k != 'blocks'}
    prefix = 'tactical_' if variant == 'tactical' else ''
    user = (HERE/f'{prefix}user.txt').read_text().format(**fields, window_data=data)
    return [{'role': 'system', 'content': (HERE/f'{prefix}system.txt').read_text()}, {'role': 'user', 'content': user}]
