
- 変なことがいくつか起きてるのでdebugしないといけないな
    - まず、欠損とparsefailureがなんなのか
        - そもそもどの段階で失敗してる？返り値がすでになかった？返り値から処理するところで失敗？
    - なんでstateが二択になってるのか
        - 論文のpromptとは異なるものを送っている？
- queryとして実際に送ったものを保存してるのはどれかな直接みたい。あと失敗した時のlogも
# stateが二択になってる理由
- 


# memo
---
'ff'
コードを理解したいので助けて欲しい。
XPERIMENT = common.Experiment(
name="03_gpt35_history_n30", models=("3.5",),
tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=30,
)
if name == "main":
raise SystemExit(common.run(EXPERIMENT))

ってのは型定義されたEXPERIMENTをargとして

def run(spec, argv=None):
    """Serialize paid executions, including resume, to prevent duplicate sends."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--execute" not in arguments:
        return _run(spec, arguments)
    directory = output_path(EXPERIMENTS / spec.name / "results")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Stopped: another execution holds the results lock", file=sys.stderr)
            return 2
        return _run(spec, arguments)

を実行するんだよね。

--excecuteがテストがない時の実行条件でdirectoryをmkdirする。
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)はなんだ？
で本体は_runに行くのか。
あれ、
def _run(spec, argv=None):
    parser = argparse.ArgumentParser(description=spec.name)
    parser.add_argument("--execute", action="store_true", help="Allow paid execution (also requires confirmation)")
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Exp.3: replay saved responses and send only missing queries")
    parser.add_argument("--retry-uncertain", action="store_true", help="Acknowledge possible duplicate billing for timed-out/interrupted requests")
    parser.add_argument("--retries", type=int, default=0,
                        help="Explicit upstream retries per query; default 0 avoids uncertain duplicate billing")
    args = parser.parse_args(argv)
    if args.resume and spec.name != "03_gpt35_history_n30":
        parser.error("--resume is currently supported for Exp.3 only")

でspecとargsあるけど、EXPERIMENTはどっちだ？specか。pythonを実行する時のコマンドラインがargsとして処理される。
