# mtd-rl-xavier

強化学習エージェントの振る舞いをLLMで「意図」の文章に翻訳し（`translate`）、その翻訳をLLM-as-a-judgeで評価する（`evaluation`）ためのリポジトリです。
`src/` には `src/translate/` と `src/evaluation/` だけがあります。旧LLM-Xavierの研究コードは独立したプロジェクト [`workspace/`](workspace/README.md) にあり、coreとは無関係です。

## 翻訳の流れ

翻訳は1本のループで、何をどう使うかはすべて実行設定（`configs/translate/*.yaml`）で決めます。

```text
入力を選ぶ（inputs:）→ 各入力が時刻tまでの情報だけで文字列を作る → プロンプトを埋める（prompt:）→ 1回呼び出す（model:）
```

- 入力：`instructions`（有効な指示）、`alarms`（警報）、`question`（人の質問）、`recent_messages`（最近の警報と質問）、`previous_translation`（前回の翻訳）、`trajectory`（トラジェクトリ）、`policy_trace`（方策の記録と行動ヘッドの重み）。
- 入力は `View(t)`（時刻t以前の行・メッセージ・翻訳）しか受け取らないので、未来の情報は入りません。
- プロンプトの文面は実行設定の `prompt:` に直接書きます。文面を書き換えたコピーを作るだけでプロンプトのアブレーションができます。

翻訳する時点は3種類です（2026-11-09の定義）。同じstepに重なった時点は、理由をすべて記録した1回の翻訳になります。

- 開始点（`start`）：エピソードの開始（step 0）です。指示が出ていてもいなくても翻訳します。指示がなければ `(no instruction in effect)` と表示します。
- 要求点（`request`）：人の質問がその対象に届いた時点です。指示と警報は翻訳を起こさず、プロンプトと変化点検出器への入力になるだけです。
  - シナリオでは、人のメッセージに指示が含まれていれば `kind: instruction` と書きます。`question` は、人が状況を把握したいだけの質問に使います。指示を含むメッセージは要求点にならず、変化点検出器が判定する指示として扱います。
- 変化点（`change`）：設定の `change:` が決めます。`preset` は前もって書いたstep、`clef_flash` は変化点検出器（ローカルのvLLMで動くclef-flash）が、外的要因や指示によってそれまでの意図では行動を説明できなくなったと判定した時点です。計画どおりの変化は変化点ではありません。

| 設定 | 内容 |
|---|---|
| `configs/translate/preset.yaml` | 変化点を前もって指定（145、245、315）。CPUで実行できます |
| `configs/translate/online.yaml` | clef-flashが変化点を判定（GPU、D49承認後） |
| `configs/translate/policy.yaml` | トラジェクトリではなく方策の記録（obs、gru_out、act_in、logits、actions）と行動ヘッドの重みを読みます |

方策を読む入力は、旧「重みと中間状態」（サンドボックスで全重みを読むcode session）から、記録した中間状態と行動ヘッドの重み（13×128）とヘッド入力 `act_in` に変わりました（2026-11-09、U-c (b)）。

## 評価

Judgeは1つのエンジン（`python -m evaluation`）です。翻訳がどの入力を使ったかに関係なく、人の評価者が見るのと同じ材料（指示、警報、質問、前回の翻訳、時刻tまでのトラジェクトリ。動画は未実装）で評価します。
イベントごとに、その時点の種類（start / change / request）の基準をすべて1回の呼び出しで評価します。プロンプトの文面と基準（rubric `pilot-5`）は `configs/evaluation/judge.yaml` に直接書きます。
人の評価（`import-ratings`）と集計（`report`：基準ごとの平均、重み付きκ、要求点の3.0秒以内の割合）も同じCLIです。

## ディレクトリ

| パス | 内容 |
|---|---|
| `src/translate/` | 翻訳のループ（[README](src/translate/README.md)） |
| `src/evaluation/` | Judge（[README](src/evaluation/README.md)） |
| `configs/translate/` | 実行設定（プロンプトの文面を含む）とシナリオ（`scenarios/`） |
| `configs/evaluation/` | Judgeの設定（プロンプトの文面と基準を含む） |
| `data/mhi/` | MHI v1.3の3試合（CSV・manifest・行動付き動画）。読み取り専用 |
| `slurm/` | すべてのSlurm job と `common.sh`（[slurm/README.md](slurm/README.md)） |
| `scripts/lag/` | 隣の `../mtd-rl-lag` のinterpreterで動くhelper（rollout記録、ACMI再生、CSV変換） |
| `scripts/upgrade_events_v2.py` | schema 2の古い翻訳runを一度だけschema 3に変換します |
| `tests/` | coreのテスト |
| `outputs/`, `private/`, `logs/slurm/` | 実行結果・非公開入力・Slurmログ（いずれもgit管理外） |
| `workspace/` | 独立した自由実験の場（旧LLM-Xavier実験00〜17） |

## セットアップ

このホストはSlurmのlogin nodeです。重い処理はすべてjobとして投げ、login nodeでは編集とgit操作だけを行います。すべてのjobはリポジトリのrootから投げます。

```bash
sbatch slurm/uv_sync.sbatch          # .venv を uv.lock どおりに作る（uv.lock やパッケージ構成が変わったときも）
```

`uv` の既定の場所は `$HOME/.local/bin/uv` です（別の場所なら `UV=<パス>`）。jobは `.venv/bin/python`（3.12）を直接使います。

## OpenAI APIキー

キーはリポジトリ直下の `.env` に置きます。`.env` はgitignoreされていて、意図的にグループ内で共有しています。移動やchmodはしません。
有料jobを投げる直前にだけ読み込み、投げ終えたら環境から消します。

```bash
set -a; . ./.env; set +a
sbatch --export=ALL slurm/translate.sbatch paid --config configs/translate/preset.yaml
unset OPENAI_API_KEY
```

## 実行（Slurm）

有料になりうるjobは、第1引数が `preview`（既定、API呼び出し0件、キーを読まない）か `paid` です。`paid` だけが `--execute --confirm-paid-api` を付けます。2番目以降の引数はPython CLIに渡ります。

| やること | コマンド例 |
|---|---|
| 翻訳（preset、policy） | `sbatch slurm/translate.sbatch preview --config configs/translate/preset.yaml` |
| 翻訳（online、GPU、D49承認後） | `VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=private/<s>.yaml sbatch --export=ALL slurm/translate_online.sbatch probe` |
| Judge | `sbatch slurm/evaluate.sbatch preview --run outputs/translate/<run>` |
| rolloutの記録（act_in付き） | `OUT=outputs/rollouts/mhi-v1.3-act-in SEEDS=100601 sbatch --export=ALL slurm/lag/record_rollouts.sbatch` |
| エピソードの準備 | `sbatch --export=ALL slurm/lag/prepare_episode.sbatch`（環境変数は [slurm/README.md](slurm/README.md)） |

`policy.yaml` は `outputs/rollouts/mhi-v1.3-act-in` を読みます。そのcheckoutにまだなければ、上の「rolloutの記録」を一度実行してください（CPUで約15秒、2026-11-09のjob 5026887）。記録は決定的です。job 5026901で、seed 100601のobs・gru_out・logits・actionsが2026-11-04の記録とビット単位で一致し、`act_in` と行動ヘッドの重みからlogitsを最大誤差9.54e-07で再現できることを確かめました。
人手で作る入力（実シナリオ、rating）は `private/` に置きます。ratingの形式は `src/evaluation/examples/` にあります。

## 出力と再送防止

| 出力先 | 書くもの | 再実行したとき |
|---|---|---|
| `outputs/translate/<run>/` | 翻訳（`run.json`、`events.jsonl`、`requests.jsonl`、`responses.jsonl`） | 既定の名前（設定名とハッシュ）のrunは作り直さず拒否します。別のrunは `--run-name` で |
| `outputs/evaluation/<run>/<name>/` | Judge（要求と応答、結果、監査、rating、report） | output root以下のjournalにある要求は二度と送りません |

- 有料の要求は送る前に `requests.jsonl` へfsyncで記録します。自動の再送はしません。
- Judgeは `outputs/evaluation/` 以下のすべてのjournal（2026-11-05の `outputs/evaluation/assess/<run>/judge_requests.jsonl` を含む）の `request_sha256` を調べ、送信済みの要求を送りません。有料実行中は `outputs/evaluation/.paid-judge.running` を排他的に作り、同時に1つだけ動きます。異常終了でこのファイルが残ったら、judgeが動いていないことを確かめてから手で消します。
- 同じjudgement directoryで評価済みのeventを別のbytesで送ろうとすると拒否します。わざと同じeventを評価し直すときは `rubric_version` かプロンプトを変えます。送信済みなのにこのdirectoryに結果がないevent（判定の形式が不正だった場合など）は自動では再送せず、実行を `failed` にしてそのeventを示します。
- そのeventを送り直す唯一の方法は `--only-event <event_id> --execute --confirm-paid-api` です（利用者が明示したときだけ使います）。このdirectoryから同じbytesで送信済みで、ここに結果がなく、ほかのjournalにもないeventだけを1回送り、journalに利用者の指示による再送として記録します。ほかのeventには触れません。再送分も課金されます。
- 有料のjudgeは、journalのある `outputs/evaluation/` を持つcheckout（mainのcheckout）で実行します。git worktreeからの有料実行は、mainのcheckoutを示して拒否します。設定の `output` を変えても、このリポジトリの `outputs/evaluation/` は必ず調べます。
- バイト列を固定するgolden、manifest、記録済みrunの再生確認は2026-11-09に削除しました（U-b）。2026-11-05の評価をそのまま再開することはできません。

## テスト

```bash
sbatch slurm/validate.sbatch full      # coreの全テスト
sbatch slurm/validate.sbatch static    # ruff、bash -n、repo外からのimport、行数の上限（src、testsとも4000行）
sbatch slurm/validate.sbatch preview   # 両パッケージの全設定のpreview（要求は送らない）
```

どのテストもAPIを呼ばず、jobは `OPENAI_API_KEY` を必ず消してから動きます。`preview` は生成物（seed 100601のエピソード、rollout）を `SOURCE_ROOT`（既定はこのcheckout）から読みます。

## workspace/ について

`workspace/` は、自由に実験してよい独立したuvプロジェクトです。独自の `pyproject.toml`、`uv.lock`、`.venv` を持ちます。
coreは `workspace/` をimportも読み取りもしません。workspaceからcoreへの依存も、`data/mhi/` の読み取りだけです。

## 詳細な仕様（英語）

- [src/translate/README.md](src/translate/README.md)：入力、設定、時点、run directory、安全策
- [src/evaluation/README.md](src/evaluation/README.md)：Judgeの設定、材料、再送防止、rating、report
- [slurm/README.md](slurm/README.md)：job一覧と規約
- [data/mhi/README.md](data/mhi/README.md)：MHI試合データ
