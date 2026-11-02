"""A self-contained, read-only comparison of fixed-target narratives."""
import json
from .common import HERE


def build_viewer(plan, rows, summary, directory):
    data = dict(plan_id=plan['plan_id'], rows=rows, summary=summary,
                samples=plan['samples'], prompt_spec=plan['prompt_spec'])
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = (HERE / 'templates/review.html').read_text()
    if template.count('__DATA__') != 1:
        raise ValueError('Expected exactly one viewer data placeholder')
    path = directory / 'report/review.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace('__DATA__', serialized), encoding='utf-8')
    return path
