# mtd-rl-xavier

強化学習エージェントの振る舞いをLLMで「意図」の文章に翻訳し（`translate`）、その翻訳をLLM-as-a-judgeで評価する（`evaluation`）ためのリポジトリです。
`src/` には `src/translate/` と `src/evaluation/` の2つだけがあります。旧LLM-Xavierの研究コードは独立したプロジェクト [`workspace/`](workspace/README.md) にあり、coreとは無関係です。

## 翻訳の流れ

翻訳は1本のパイプラインで、何をどう使うかはすべて設定ファイル（`configs/translate/*.yaml`）で決めます。

```text
入力を選ぶ（inputs）→ 入力を加工する → プロンプトを完成させる（prompt）→ 翻訳する
```

1. 入力を選ぶ：`instructions`（指示）、`alarms`（警報）、`question`（人の質問）、`previous_translation`（前回の翻訳）、`trajectory`（トラジェクトリ）、`policy`（方策の重みと中間状態）から選びます。
2. 入力を加工する：時刻tまでの情報だけから、プロンプトに入れる値を作ります。
3. プロンプトを完成させる：プロンプトの文面は設定側のプロンプトファイル（`configs/translate/prompts/`）にあります。文面を書き換えるだけでプロンプトのアブレーションができます。
4. 翻訳する：プロンプトファイルの `protocol` で呼び出し方が決まります。`one_shot` は1回のOpenAI呼び出し、`code_session` はモデルがサンドボックスでPythonを実行しながら方策を読む複数回の対話です。

翻訳する時点は次の3種類です（イベントのschema 3）。

- 開始点（`start`）：対象に最初の指示が出た時点です。
- 変化点（`change`）：clef-flash（変化点検出器）が判定します。新しい戦術を始めたときも、戦術を変えたときも変化点です。
- 要求点（`request`）：人の質問が届いた時点です。

2026-11-07までのrun（schema 2）は `initial`（最初の指示）と検出器の `start` も使っていました。Judgeはこれらを明示した対応表で読みます（評価の節を参照）。

時点の決め方は2通りあります。

- `online`：エピソードを先頭から進めながら、clef-flashが変化点を判定します。開始点と要求点はその時点で必ず翻訳します。
- `preset`：翻訳する時点と種類を、設定ファイルに前もって書きます。

旧「手法A」（トラジェクトリを読む翻訳）と旧「手法B」（重みと中間状態を読む翻訳）は、入力の選び方が違う設定ファイルにすぎません。`seed100601-trajectory*.yaml` と `seed100601-policy*.yaml` が、以前と1バイトも違わない要求を送ります。

## 評価

Judgeは翻訳がどの入力を使ったかに関係なく、同じ方法で評価します。
Judgeには、人の評価者が見るのと同じ材料（シナリオの文面とトラジェクトリ。動画は未実装）が常に渡ります。
Judgeは2種類です。

- `evaluation.assess_cli`：3時点×4観点の評価です。実データで1回の有料実行実績があります。
- `evaluation.cli`：D29のkey-point評価です。まだ実行していません。

Judgeのプロンプトの文面も、翻訳と同じく設定側のプロンプトファイル（`configs/evaluation/prompts/`）にあります。どのファイルを使うかはjudge設定（`configs/evaluation/*.yaml`）が決めます。

- 新しいrunの既定：`assess.yaml` と `judge-text*.yaml`。トラジェクトリを「Translatorが見たもの」ではなく「人が見られるテレメトリ」と書きます。
- 互換用：`assess-legacy.yaml` と `judge-text-legacy.yaml`。以前コードにあったプロンプトと1バイトも違わない要求を送ります。2026-11-05の有料実行は `assess-legacy.yaml` でしか再開できません。

assess judgeの設定は、イベントのschemaごとに、トリガーの種類をどの時点の基準で評価するかを `triggers` に書きます。既定の設定では、schema 2の検出器の `start` を変化点として評価します。

## ディレクトリ

| パス | 内容 |
|---|---|
| `src/translate/` | 翻訳パイプライン（[README](src/translate/README.md)） |
| `src/evaluation/` | Judge 2種、比較用viewer、クリップ作成（[README](src/evaluation/README.md)） |
| `configs/translate/` | 実行設定、プロンプトファイル（`prompts/`）、シナリオ（`scenarios/`）、方策のspec（`policies/`） |
| `configs/evaluation/` | Judgeの設定と、Judgeのプロンプトファイル（`prompts/`） |
| `data/mhi/` | MHI v1.3の3試合（CSV・manifest・行動付き動画）。読み取り専用 |
| `slurm/` | すべてのSlurm job と `common.sh`（[slurm/README.md](slurm/README.md)） |
| `scripts/` | mtd-rl-lag側のinterpreterで動くhelper（rollout記録、エピソード準備、parity）と、互換性の確認script |
| `tests/` | coreのテスト |
| `outputs/`, `private/`, `logs/slurm/` | 実行結果・非公開入力・Slurmログ（いずれもgit管理外） |
| `workspace/` | 独立した自由実験の場（旧LLM-Xavier実験00〜17） |

## セットアップ

このホストはSlurmのlogin nodeです。重い処理はすべてjobとして投げ、login nodeでは編集とgit操作だけを行います。
すべてのjobはリポジトリのrootから投げます。

```bash
sbatch slurm/uv_sync.sbatch          # .venv を uv.lock どおりに作る（uv.lock やパッケージ構成が変わったときも）
```

`sbatch slurm/uv_sync.sbatch` には `uv` が必要です。既定の場所は `$HOME/.local/bin/uv` で、別の場所にある場合は `UV=<uvのパス>` で指定します。
rollout記録・エピソード準備・parity確認の各jobは、隣の `../mtd-rl-lag` チェックアウトを使います。別の場所に置くときは `LAG_ROOT=<パス>` を指定します。

jobは `.venv/bin/python` を直接使い、job内で `uv sync` は行いません。
ログは `logs/slurm/<job名>-<job ID>.out` に出ます。

## OpenAI APIキー

キーはリポジトリ直下の `.env` に置きます。`.env` はgitignoreされていて、意図的にグループ内で共有しています。移動やchmodはしません。
有料jobを投げる直前にだけ読み込み、投げ終えたら環境から消します。キーを画面やコマンド行に出さないでください。

```bash
set -a; . ./.env; set +a
sbatch --export=ALL slurm/translate.sbatch paid --config configs/translate/<設定>.yaml --run-name <名前>
unset OPENAI_API_KEY
```

## 実行（Slurm）

有料になりうるjobは、第1引数が `preview`（既定、API呼び出し0件）か `paid` です。
`paid` だけが `--execute --confirm-paid-api` を付けます。2番目以降の引数は、そのままPython CLIに渡ります。

| やること | コマンド例 |
|---|---|
| 翻訳（preset） | `sbatch slurm/translate.sbatch preview --config configs/translate/seed100601-policy.yaml` |
| 翻訳（online、GPU、D49承認後） | `VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=private/translator/<s>.yaml sbatch --export=ALL slurm/translate_vllm.sbatch probe`（有料は `paid`） |
| rolloutの記録とエピソードの準備 | `sbatch slurm/episode_record_rollouts.sbatch` の後に `sbatch slurm/episode_prepare.sbatch`（引数は [slurm/README.md](slurm/README.md)） |
| 比較viewer | `sbatch slurm/viewer.sbatch --column 軌跡=outputs/translate/<run> --column 方策=outputs/translate/<run> --out <html>` |
| Judge（3時点×4観点） | `sbatch slurm/judge_assess.sbatch preview --translator-run outputs/translate/<run>` |
| Judge（key point） | `sbatch slurm/judge_keypoints.sbatch preview --translator-run outputs/translate/<run> --key-points private/evaluation/<kp>.yaml` |

有料で実行するときは、上の `preview` を `paid` に替え、「OpenAI APIキー」の手順で投げます。
人手で作る入力（key point、reference point、rating）は `private/evaluation/` に置きます。形式は `src/evaluation/examples/` にあります。

## 出力とresume

| 出力先 | 書くもの | 再実行したとき |
|---|---|---|
| `outputs/translate/<run>/` | 翻訳（`events.jsonl`、`cues.jsonl`、要求と応答、code sessionの記録） | 毎回新しいrun。resumeなし |
| `outputs/evaluation/assess/<run>/` | assess judge | `judge_requests.jsonl` にある要求は再送しない |
| `outputs/evaluation/<UTC>_judge_*/` | key-point judge | resumeなし。再実行すると全件を再送する |

以前の出力（`outputs/translator/`、`outputs/intent_compare/`、`outputs/weightreading/`）はそのまま残り、Judgeとviewerで読めます。
旧手法Bの `--resume` は廃止しました。記録済みの3 runはすべて完了しているので、続きから実行するものはありません。

resumeとバイト列の保証は次のとおりです。

- assess judgeは `request_sha256` で送信済みかを判定します。プロンプトファイル、packet、judge設定の値が1 byteでも変わると別のrequestになるため、同じassessment directoryで判定済みのeventを別のbytesで送ろうとすると実行を拒否します。2026-11-05の支払い済みrunは `configs/evaluation/assess-legacy.yaml` で再開します。
- 出荷している設定のプロンプトと要求のバイト列はgoldenで固定してあります。記録済みの旧手法A・Bのrunは、このパイプラインで再実行するとすべての要求が一致します（`scripts/check_request_replay.py`）。
- 確認は `sbatch --export=ALL,SOURCE_ROOT=<出力のあるcheckout> slurm/validate.sbatch golden` で行います。APIは呼びません。

## テスト

```bash
sbatch slurm/validate.sbatch subset             # translate・policy・sandbox・LLM・LAG・Judge
sbatch slurm/validate.sbatch full               # coreの全テスト
sbatch slurm/validate.sbatch static             # ruff、bash -n、repo外からのimport、古いパスの検出
sbatch slurm/validate.sbatch translate-preview  # 出荷しているすべての設定のpreview（要求は送らない）
```

どのテストもAPIを呼ばず、jobは `OPENAI_API_KEY` を必ず消してから動きます。

## workspace/ について

`workspace/` は、自由に実験してよい独立したuvプロジェクトです。独自の `pyproject.toml`、`uv.lock`、`.venv` を持ちます。
coreは `workspace/` をimportも読み取りもしません。workspaceからcoreへの依存も、`data/mhi/` の読み取りだけです。
使い方は [workspace/README.md](workspace/README.md) を見てください。

## 詳細な仕様（英語）

- [src/translate/README.md](src/translate/README.md): パイプライン、設定、入力、プロンプトファイル、`events.jsonl` の契約
- [src/evaluation/README.md](src/evaluation/README.md): 2種類のJudge、入力、出力、安全策、viewer
- [src/translate/llm/README.md](src/translate/llm/README.md): LLM client、vLLM、課金gate、D49
- [src/translate/lag/README.md](src/translate/lag/README.md): LAGデータの読込と描画
- [data/mhi/README.md](data/mhi/README.md): MHI試合データ
- [slurm/README.md](slurm/README.md): job一覧と規約
