# このrunのディレクトリについて

ここは、短縮版プロンプト `tactical_short_v1` を使う実験の計画・入力・回答をまとめる保存先です。
比較元の `../tactical_batch_v1/` とは別の実験条件です。

| 場所 | 意味 |
|---|---|
| `manifest.json` | このrunの条件を記録したファイル。モデル、窓、プロンプト本文・バージョン・ハッシュ、リクエストIDなどを保存 |
| [prompts/](prompts/README.md) | 今回使うプロンプト、窓ごとの入力全文、元条件とのdiff。中身の案内はリンク先を参照 |
| `windows/` | 各窓の61時点の入力データをJSONで保存 |
| `api/batch/` | APIへ渡すBatch入力JSONLと、各Batchの準備・送信・回収状態 |
| `api/identity.json` | このAPI処理がどの実行計画に属するかの識別情報 |
| `api/requests.jsonl` | 送信処理で作成するリクエスト記録。dry runだけなら存在しない |
| `api/responses.jsonl` | 回収したAPI応答の記録。dry runだけなら存在しない |
| `evaluation/` | 回答のレビュー用データ、形式検証・語数集計、未記入の評価票など。dry run時は回答待ちの状態 |

dry runでも入力や評価票は作成されます。ファイルがあることとモデル回答があることは区別してください。
実行状況は `api/batch/state.json` で確認できます。`prepared` は入力の準備済み、まだBatch未送信という意味です。

```sh
# リポジトリのルートで実行。ローカルの状況確認のみ
.venv/bin/python experiments/17_commentary_LAG/run.py --output tactical_short_batch_v1 --status
```
