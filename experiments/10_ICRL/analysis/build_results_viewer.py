"""Build the Exp.09-style standalone viewer; missing reasoning scores stay unscored."""
import argparse
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
data = import_module('experiments.10_ICRL.analysis.results_data')
HERE = data.HERE


def render_html(rows, hashes, template):
    payload = dict(created_at_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
                   rows=rows, source_sha256=hashes,
                   reasoning_scored=sum(r.get('reasoning_score') is not None for r in rows))
    data.require(template.count('__DATASET__') == 1, 'Expected exactly one HTML data placeholder')
    # Literal </script> inside a saved answer must remain text, never executable markup.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return template.replace('__DATASET__', serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build an offline ICRL input/output viewer without API requests.')
    parser.add_argument('--output', type=Path, default=HERE / 'view_rawdata.html',
                        help='New HTML file; existing files are never overwritten')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('HTML already exists; choose a new --output path')
    rows, hashes = data.load_rows()
    template = (HERE / 'templates/results_viewer_template.html').read_text(encoding='utf-8')
    html = render_html(rows, hashes, template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(html)
    scored = sum(r['reasoning_score'] is not None for r in rows)
    print(f'{len(rows)} queries / {scored} reasoning scores: {args.output}')


if __name__ == '__main__':
    main()
