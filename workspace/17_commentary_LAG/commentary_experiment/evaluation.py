"""Export answers for human review; never invent ground-truth scores."""
import json
from . import execution as ex
from .config import REFERENCE_URL


def parse_intentions(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        choice = raw['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Response did not finish normally')
        value = json.loads(choice['message']['content'], object_pairs_hook=unique)
        if not isinstance(value, dict) or set(value) != {'A0100', 'A0200', 'B0100', 'B0200'}:
            raise ValueError('Expected exactly four aircraft fields')
        if any(not isinstance(v, str) or not v.strip() for v in value.values()):
            raise ValueError('Each narrative must be a nonempty string')
        return dict(valid=True, narratives=value, error=None)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return dict(valid=False, narratives=None, error=str(exc))


def evaluate_saved(plan, directory):
    folder = directory / 'evaluation'
    folder.mkdir(exist_ok=True)
    ex.write_json(folder/'reference.json', dict(url=REFERENCE_URL, contents_verified=False, included_in_prompt=False))
    template = folder/'rubric_template.json'
    if not template.exists():
        ex.write_json(template, [dict(query_id=q['query_id'], center_step=plan['samples'][q['sample_id']]['target_step'],
            reviewer=None, supported_claims=None, contradicted_claims=None, unverifiable_claims=None,
            important_facts_coverage=None, intention_score=None, notes='') for q in plan['queries']])
    responses = {r['query_id']: r for r in ex.read_journal(directory/'api/responses.jsonl')}
    output = []
    for q in plan['queries']:
        record = responses.get(q['query_id'], {})
        raw = record.get('raw_response')
        output.append(dict(query_id=q['query_id'], model=q['model'], repeat=q['repeat_id'],
            center_step=plan['samples'][q['sample_id']]['target_step'],
            status='received' if raw is not None else ('error' if record else 'pending'),
            choices=raw.get('choices', []) if raw else [], usage=raw.get('usage') if raw else None,
            error=record.get('error')))
        if plan.get('prompt_variant') == 'tactical':
            output[-1]['intention_format'] = parse_intentions(raw) if raw else None
    ex.write_json(folder/'responses.json', output)
