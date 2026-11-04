"""Production English prompts; template files are the runtime source of truth."""
import json
import difflib
import hashlib
from pathlib import Path
PROMPT_VERSION = 'lag-commentary-v2'
HERE = Path(__file__).resolve().parent
VARIANTS = {
    'commentary': dict(version=PROMPT_VERSION, prefix='', default_output='commentary_batch_v1',
                       output_format='sections', description='場面解説（セクション別の文章回答）'),
    'tactical': dict(version='lag-tactical-v1', prefix='tactical_', default_output='tactical_batch_v1',
                    output_format='aircraft_intentions_json', description='61時点全体の戦術的意図（4機のJSON回答）'),
    'tactical_short_v1': dict(version='lag-tactical-short-v1', prefix='tactical_short_v1_',
                              default_output='tactical_short_batch_v1', output_format='aircraft_intentions_json',
                              description='窓全体の主な行動・変化と理由を短縮（注釈情報なし・各90語以内）',
                              comparison_dir='tactical_short_v1',
                              length=dict(target_words=[60, 90], max_words=90, target_sentences=[2, 3],
                                          allow_shorter=True, inactive_sentences=1)),
    'tactical_short_v2': dict(version='lag-tactical-short-v2', prefix='tactical_short_v2_',
                              default_output='tactical_short_batch_v2', output_format='aircraft_intentions_json',
                              description='目的を中心に行動と根拠を結ぶ（原則1文・各55語以内）',
                              comparison_dirs={name: f'tactical_short_v2/{name}' for name in
                                               ('from_tactical_short_batch_v1', 'from_tactical_batch_v1')},
                              length=dict(target_words=[35, 50], max_words=55, target_sentences=[1, 1],
                                          max_sentences=2, allow_shorter=True, inactive_sentences=1)),

    'tactical_short_counttime_v1': dict(
        version='lag-tactical-short-counttime-v1', prefix='tactical_short_counttime_v1_',
        default_output='tactical_short_CountTime_v1', output_format='aircraft_intentions_json',
        description='luna逐次実行・A0100/A0200のみ説明・応答時間計測',
        transport='synchronous', default_models=['luna'], required_models=['gpt-5.6-luna'],
        aircraft_ids=['A0100', 'A0200'], comparison_dir='tactical_short_counttime_v1',
        length=dict(target_words=[35, 50], max_words=55, target_sentences=[1, 1],
                    max_sentences=2, allow_shorter=True, inactive_sentences=1)),
}


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_prompt_spec(variant):
    """Freeze templates and review artifacts into the plan, before any API calls."""
    if variant not in VARIANTS:
        raise ValueError('Unknown prompt variant')
    config = VARIANTS[variant]
    spec = {k: v for k, v in config.items() if k not in ('prefix', 'comparison_dir', 'comparison_dirs')}
    spec['variant'] = variant
    spec['templates'] = {}
    for role in ('system', 'user'):
        path = HERE / f"{config['prefix']}{role}.txt"
        content = path.read_text(encoding='utf-8')
        spec['templates'][role] = dict(path=str(path.relative_to(HERE.parent)),
                                       sha256=text_sha256(content), content=content)
    def load_comparison(relative):
        folder = HERE / 'prompt_changes' / relative
        comparison = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
        comparison['baseline_templates'] = {}
        comparison['diffs'] = {}
        for role in ('system', 'user'):
            baseline = (folder / f'baseline_{role}.txt').read_text(encoding='utf-8')
            if text_sha256(baseline) != comparison['baseline_sha256'][role]:
                raise ValueError(f'Baseline {role} differs from its recorded hash')
            comparison['baseline_templates'][role] = baseline
            diff = ''.join(difflib.unified_diff(
                baseline.splitlines(keepends=True), spec['templates'][role]['content'].splitlines(keepends=True),
                fromfile=f"{comparison['source_run']}/{role}.txt", tofile=f"{config['prefix']}{role}.txt"))
            if (folder / f'{role}.diff').read_text(encoding='utf-8') != diff:
                raise ValueError(f'Stale {role} diff; version and review changed prompts before running')
            comparison['diffs'][role] = diff
        comparison['change_notes'] = (folder / 'README.md').read_text(encoding='utf-8')
        return comparison
    if config.get('comparison_dir'):
        spec['comparison'] = load_comparison(config['comparison_dir'])
    if config.get('comparison_dirs'):
        spec['comparisons'] = {name: load_comparison(relative)
                               for name, relative in config['comparison_dirs'].items()}
    return spec


def snapshot_artifacts(spec):
    """Render review files from the saved plan, never from current source files."""
    files = {f'templates/{role}.txt': template['content'] for role, template in spec['templates'].items()}
    comparisons = dict(spec.get('comparisons', {}))
    if spec.get('comparison'):
        comparisons[''] = spec['comparison']
    for name, comparison in comparisons.items():
        prefix = f'changes/{name}/' if name else 'changes/'
        for role in ('system', 'user'):
            files[f'{prefix}baseline_{role}.txt'] = comparison['baseline_templates'][role]
            files[f'{prefix}{role}.diff'] = comparison['diffs'][role]
        files[f'{prefix}README.md'] = comparison['change_notes']
        provenance = {k: v for k, v in comparison.items() if k not in ('baseline_templates', 'diffs', 'change_notes')}
        files[f'{prefix}provenance.json'] = json.dumps(provenance, ensure_ascii=False, indent=2) + '\n'
    return files


def build_messages(window, variant='commentary', *, spec=None):
    spec = get_prompt_spec(variant) if spec is None else spec
    data = '\n'.join(json.dumps(block, ensure_ascii=False, allow_nan=False, separators=(',', ':')) for block in window['blocks'])
    fields = {k: v for k, v in window.items() if k != 'blocks'}
    user = spec['templates']['user']['content'].format(**fields, window_data=data)
    return [{'role': 'system', 'content': spec['templates']['system']['content']}, {'role': 'user', 'content': user}]
