# mtd-rl-xavier

強化学習エージェントの振る舞いをLLMで「意図」の文章に翻訳し（A: Translator）、その翻訳をLLM-as-a-judgeで評価する（B: Judge）ためのリポジトリです。
coreはこの2つだけを扱います。旧LLM-Xavierの研究コードは独立したプロジェクト [`workspace/`](workspace/README.md) にあり、coreとは無関係です。

## 全体像

```text
入力: data/mhi/ の試合、rollout、方策の重み
  M1   2層Translator ─────────────────────────────→ Translator v2 run ─→ Judge
  手法A 固定stepで翻訳層を呼ぶ ─→ to_translator_run ─→ (outputs/translator/)  (outputs/evaluation/)
  手法B 重みとactivationを読む ─→ （Judgeへ渡す変換はまだない）
```

- M1（`mtd_rl_xavier.translator`）はJudgeの入力形式をそのまま書き出します。ただし実データでの実行は、vLLM環境とGPU probeの承認待ち（D49）です。
- 手法A（`mtd_rl_xavier.intent_compare`）は、`python -m mtd_rl_xavier.intent_compare.to_translator_run` で1回分の呼び出しをTranslator v2 runに変換すると、Judgeに渡せます。
- 手法B（`weightreading`）の出力をJudgeへ渡す変換は、まだありません。
- Judgeは2種類です。`evaluation.assess_cli` は3時点×4観点の評価で、実データで1回の有料実行実績があります。`evaluation.cli` はD29のkey-point評価で、まだ実行していません。

## ディレクトリ

| パス | 内容 |
|---|---|
| `src/mtd_rl_xavier/` | M1 Translator（`translator/`）、手法A・比較用viewer・変換（`intent_compare/`）、LLM client（`llm/`）、LAGデータの読込と描画（`lag/`） |
| `src/weightreading/` | 手法B（`sandbox-selftest`、`intent-preview`、`intent-run`） |
| `evaluation/` | Judge 2種と設定（`config/`）、入力例（`examples/`） |
| `configs/` | Translator・手法A/B の設定。`compare*.yaml` と `intent_b.yaml` はmethod Bのresumeがバイト列を照合するので編集しない。中の古いコメント（`intent_compare/compare*.yaml`、`weightreading/intent_b.yaml`）も意図的にそのまま残している |
| `data/mhi/` | MHI v1.3の3試合（CSV・manifest・行動付き動画）。読み取り専用 |
| `slurm/` | すべてのSlurm job と `common.sh`（[slurm/README.md](slurm/README.md)） |
| `scripts/` | mtd-rl-lag側のinterpreterで動くhelperと、resume互換性の確認script |
| `tests/` | coreのテスト |
| `outputs/`, `private/`, `logs/slurm/` | 実行結果・非公開入力・Slurmログ（いずれもgit管理外） |
| `workspace/` | 独立した自由実験の場（旧LLM-Xavier実験00〜17） |

## セットアップ

このホストはSlurmのlogin nodeです。重い処理はすべてjobとして投げ、login nodeでは編集とgit操作だけを行います。
すべてのjobはリポジトリのrootから投げます。

```bash
sbatch slurm/uv_sync.sbatch          # .venv を uv.lock どおりに作る（uv.lock が変わったときも）
```

`sbatch slurm/uv_sync.sbatch` には `uv` が必要です。既定の場所は `$HOME/.local/bin/uv` で、別の場所にある場合は `UV=<uvのパス>` で指定します。
エピソード準備・rollout記録・method Bのparity確認の各jobは、隣の `../mtd-rl-lag` チェックアウトを使います。別の場所に置くときは `LAG_ROOT=<パス>` を指定します。

jobは `.venv/bin/python` を直接使い、job内で `uv sync` は行いません。
ログは `logs/slurm/<job名>-<job ID>.out` に出ます。

## OpenAI APIキー

キーはリポジトリ直下の `.env` に置きます。`.env` はgitignoreされていて、意図的にグループ内で共有しています。移動やchmodはしません。
有料jobを投げる直前にだけ読み込み、投げ終えたら環境から消します。キーを画面やコマンド行に出さないでください。

```bash
set -a; . ./.env; set +a
sbatch --export=ALL slurm/judge_assess.sbatch paid --translator-run outputs/translator/<run>
unset OPENAI_API_KEY
```

## 実行（Slurm）

有料になりうるjobは、第1引数が `preview`（既定、API呼び出し0件）か `paid` です。
`paid` だけが `--execute --confirm-paid-api` を付けます。2番目以降の引数は、そのままPython CLIに渡ります。

| やること | コマンド例 |
|---|---|
| M1 Translator（GPU、D49承認後） | `VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=private/translator/<s>.yaml sbatch --export=ALL slurm/translator_vllm.sbatch probe`（有料は `paid`） |
| 手法A | `sbatch slurm/translator_method_a.sbatch preview --config configs/intent_compare/compare.ja.yaml` |
| 手法B | `sbatch slurm/translator_method_b.sbatch preview --compare configs/intent_compare/compare.ja.yaml` |
| A/B用エピソードの準備 | `sbatch slurm/translator_record_rollouts.sbatch` の後に `sbatch slurm/translator_prepare_episode.sbatch` |
| Judge（3時点×4観点） | `sbatch slurm/judge_assess.sbatch preview --translator-run outputs/translator/<run>` |
| Judge（key point） | `sbatch slurm/judge_keypoints.sbatch preview --translator-run outputs/translator/<run> --key-points private/evaluation/<kp>.yaml` |

有料で実行するときは、上の `preview` を `paid` に替え、「OpenAI APIキー」の手順で投げます。
手法Aの結果をJudgeに渡すには、まず `python -m mtd_rl_xavier.intent_compare.to_translator_run --help` で、変換する呼び出しを1つ選びます。
人手で作る入力（key point、reference point、rating）は `private/evaluation/` に置きます。形式は `evaluation/examples/` にあります。

## 出力とresume

| 出力先 | 書くもの | 再実行したとき |
|---|---|---|
| `outputs/translator/<run>/` | M1、`to_translator_run` | M1は毎回新しいrun。resumeなし |
| `outputs/intent_compare/<run_id>/` | 手法Aの結果、`cues_A/B.jsonl`、viewer | 手法Aは同じrun IDを上書きしない。resumeなし |
| `outputs/weightreading/intent/runs/<run_id>/` | 手法B | `paid ... --run-id <id> --resume` で続きから |
| `outputs/evaluation/assess/<run>/` | assess judge | `judge_requests.jsonl` にある要求は再送しない |
| `outputs/evaluation/<UTC>_judge_*/` | key-point judge | resumeなし。再実行すると全件を再送する |

resumeが成り立つ条件は次のとおりです。

- assess judgeは `request_sha256` で送信済みかを判定します。プロンプト、packet、`evaluation/config/assess.yaml` の値が1 byteでも変わると、支払い済みの呼び出しを再送します。
- 手法Bの `--resume` は、記録されたcompare・config・specのパス文字列と内容を照合します。これらのファイルを動かしたり編集したり、別のcheckoutから再開したりすると、拒否されます。
- 既存の結果が今のコードで再開できるかは、`sbatch slurm/validate.sbatch golden` で確認できます。APIは呼びません。

## テスト

```bash
sbatch slurm/validate.sbatch subset     # Translator・LLM・LAG・手法A/B・Judge
sbatch slurm/validate.sbatch full       # coreの全テスト
sbatch slurm/validate.sbatch static     # ruff、bash -n、repo外からのimport、古いパスの検出
```

どのテストもAPIを呼ばず、jobは `OPENAI_API_KEY` を必ず消してから動きます。

## workspace/ について

`workspace/` は、自由に実験してよい独立したuvプロジェクトです。独自の `pyproject.toml`、`uv.lock`、`.venv` を持ちます。
coreは `workspace/` をimportも読み取りもしません。workspaceからcoreへの依存も、`data/mhi/` の読み取りだけです。
使い方は [workspace/README.md](workspace/README.md) を見てください。

## 詳細な仕様（英語）

- [src/mtd_rl_xavier/translator/README.md](src/mtd_rl_xavier/translator/README.md): M1と `events.jsonl` v2 の契約
- [evaluation/README.md](evaluation/README.md): 2種類のJudge、入力、出力、安全策
- [src/mtd_rl_xavier/llm/README.md](src/mtd_rl_xavier/llm/README.md): LLM client、vLLM、課金gate、D49
- [src/mtd_rl_xavier/lag/README.md](src/mtd_rl_xavier/lag/README.md): LAGデータの読込と描画
- [src/weightreading/README.md](src/weightreading/README.md): 手法B
- [data/mhi/README.md](data/mhi/README.md): MHI試合データ
- [slurm/README.md](slurm/README.md): job一覧と規約
