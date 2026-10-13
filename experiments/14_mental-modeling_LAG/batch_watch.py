"""Read-only local Batch progress monitor; no API calls, locks, or file writes."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
import unicodedata

HERE = Path(__file__).resolve().parent
JST = timezone(timedelta(hours=9))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot(directory, now=None):
    now = time.time() if now is None else now
    folder = directory / 'api/batch'
    state_path = folder / 'state.json'
    state = read(state_path)
    if state.get('scope') != 'rolling-comparison':
        raise ValueError('rolling-comparisonの計画が必要です')
    bootstrap = set(state['bootstrap_jobs'])
    planned, completed, failed = Counter(), Counter(), Counter()
    starts = []
    jobs_by_name = {j['name']: j for j in state['jobs']}
    active = sum(s['admitted'] and any(not jobs_by_name[n].get('collected') for n in s['jobs'])
                 for s in state['sets'])
    for job in state['jobs']:
        if job['name'] in bootstrap:
            continue
        model = job['model']
        total = len(job['query_ids'])
        planned[model] += total
        remote_path = folder / f"{job['name']}.remote.json"
        remote = read(remote_path) if remote_path.exists() else {}
        if remote.get('id') != job.get('batch_id') or not job.get('batch_id'):
            remote = {}
        if remote.get('created_at') is not None:
            starts.append(remote['created_at'])
        if job.get('collected'):
            completed[model] += job['successful']
            failed[model] += total - job['successful']
        else:
            counts = remote.get('request_counts') or {}
            done = max(0, min(total, counts.get('completed', 0)))
            errors = max(0, min(total - done, counts.get('failed', 0)))
            completed[model] += done
            failed[model] += errors
            # Terminal failures can omit per-request counts entirely.
            if remote.get('status') in ('failed', 'cancelled', 'expired'):
                failed[model] += total - done - errors
    start = min(starts) if starts else None
    elapsed = max(0, now-start) if start is not None else 0
    all_complete = bool(planned) and all(completed[m] == total for m, total in planned.items())
    eta_seconds = None
    reason = '各モデルの完了実績がそろうまで計算待ち'
    if all_complete:
        eta_seconds, reason = 0, '全リクエスト成功（ローカル回収・採点が残る場合あり）'
    elif any(failed.values()) or any(j.get('collected') and j['successful'] < len(j['query_ids'])
                                    for j in state['jobs']):
        reason = '失敗・期限切れあり：終了予測を停止（送信側ログを確認）'
    elif elapsed > 0 and all(completed[m] > 0 for m in planned):
        # Separate model queues run in parallel. The slowest queue determines ETA.
        eta_seconds = max(elapsed * (total-completed[m]) / completed[m] for m, total in planned.items())
        reason = '暫定：経過時間とモデル別成功件数から、最も遅いモデルの残り時間を推定'
    report_path = folder / 'comparison_sets.json'
    report = read(report_path) if report_path.exists() else []
    return dict(now=now, start=start, elapsed=elapsed, eta_seconds=eta_seconds, reason=reason,
                planned=dict(planned), completed=dict(completed), failed=dict(failed), active=active,
                total_sets=len(state['sets']), collected_sets=sum(r['responses_complete'] for r in report),
                valid_sets=sum(r['predictions_complete'] for r in report),
                state_age=max(0, now-state_path.stat().st_mtime),
                comparison_age=max(0, now-report_path.stat().st_mtime) if report_path.exists() else None,
                collected=sum(j.get('collected', False) for j in state['jobs']), total_jobs=len(state['jobs']))


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}時間{minutes:02d}分{seconds:02d}秒'


def stamp(epoch):
    return datetime.fromtimestamp(epoch, JST).strftime('%Y-%m-%d %H:%M:%S JST')


def pad(text, width):
    """Align Japanese labels by terminal column width, not character count."""
    text = str(text)
    size = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text)
    return text + ' ' * max(0, width - size)


def clock_duration(seconds):
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def bar(value, total, width=20):
    ratio = min(1, max(0, value / total)) if total else 0
    filled = int(ratio * width)
    return '[' + '#' * filled + '-' * (width-filled) + ']'


def render(s):
    total, completed = sum(s['planned'].values()), sum(s['completed'].values())
    errors = sum(s['failed'].values())
    percent = completed / total * 100 if total else 0
    finished = total > 0 and completed == total
    status = 'API処理完了' if finished else ('要確認' if errors else '処理中')
    eta = s['eta_seconds']
    if finished:
        end_text, remaining = '完了', '--:--:--'
    elif eta is not None:
        end_text = datetime.fromtimestamp(s['now'] + eta, JST).strftime('%m/%d %H:%M JST')
        remaining = clock_duration(eta)
    else:
        end_text, remaining = ('算出停止' if errors else '計算待ち'), '--:--:--'
    line = '-' * 68
    lines = [f"LAG / BATCH                                      {status}",
             datetime.fromtimestamp(s['now'], JST).strftime('%Y-%m-%d %H:%M:%S JST'),
             line,
             f"終了予想  {end_text}",
             f"経過      {clock_duration(s['elapsed'])}      残り目安  {remaining}",
             '',
             f"{bar(completed, total, 28)}  {percent:5.1f}%   {completed:,} / {total:,} 件",
             line,
             pad('モデル', 9) + pad('進捗', 30) + pad('成功 / 対象', 18) + '失敗']
    models = sorted(s['planned'], key=lambda m: ({'sol': 0, 'terra': 1, 'luna': 2}.get(m.rsplit('-', 1)[-1], 3), m))
    for model in models:
        count = s['planned'][model]
        done = s['completed'].get(model, 0)
        ratio = done / count * 100 if count else 0
        label = model.rsplit('-', 1)[-1]
        lines.append(pad(label, 9) + pad(f'{bar(done, count)} {ratio:5.1f}%', 30)
                     + pad(f'{done:>5,} / {count:,}', 18) + str(s['failed'].get(model, 0)))
    lines += [line,
              f"比較セット  {s['collected_sets']} / {s['total_sets']} 回収済み     {s['active']} セット稼働中",
              f"予測有効    {s['valid_sets']} セット              Batch回収  {s['collected']} / {s['total_jobs']}",
              '',
              f"ログ更新 {int(s['state_age'])}秒前  |  表示終了: Ctrl+C（送信は継続）",
              '※ 進捗バーは追加Batch分。比較セット数は通常API分も含む。']
    if finished:
        lines.append('※ API処理完了。最後の回収・採点は送信側の表示で確認。')
    elif errors:
        lines.append('! 失敗あり。送信側ログを確認してください。終了予想は停止中。')
    elif eta is None:
        lines.append('※ 各モデルの完了実績がそろうと終了予想を表示します。')
    else:
        lines.append('※ 終了予想は暫定。入力長・待ち時間で変動します。')
    if s['state_age'] > 180:
        lines.append('! 3分以上更新なし。送信側ターミナルを確認してください。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='sliding_window_v1')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error('--interval must be >= 1')
    directory = (HERE / 'results' / args.output).resolve()
    if not directory.is_relative_to((HERE / 'results').resolve()):
        parser.error('--output must be inside results/')
    try:
        while True:
            result = snapshot(directory)
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(render(result), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n表示を終了しました。送信側の処理は継続します。')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
