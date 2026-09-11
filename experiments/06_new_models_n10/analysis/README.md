# Exp.06の解析

## 4モデルを同じ図で比較（PNGのみ）

新しい入口は`compare_models.py`。GPT-3.5の03_1 N10 subsetとTerra/Luna/Solを同じ図で比較し、全図のモデル色を固定する。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/06_new_models_n10/analysis/compare_models.py --allow-api-failures
```

`comparison_4models/`へ65 PNG、CSV、抽出ID、README、RESULTSを新規保存する。`--allow-api-failures`を明示したときだけ、API失敗を欠損として保持して有効応答で評価する。元結果は上書きしない。
図の分割・混同行列の番号・入力/出力・関数・欠損方針は[比較README](COMPARISON_README.md)を参照。

## 従来のモデル別解析

06の全分割実行が完了してから、repository rootで実行する。

```bash
source .venv/bin/activate
python experiments/06_new_models_n10/analysis/analyze.py
```

03_1と同じ[共通評価コード](../../common_analysis/README.md)を使い、**N=10だけ**を生成する。
`sol/`、`terra/`、`luna/`の各directoryに`cross_task_10/`、`pendulum_10/`、`diagnostics_10/`、`tables/`を出す。
各モデル22図×PNG/SVGと12 CSV。モデルを混ぜて集計しない。
分割結果がある場合、解析の前に05と全batchをオフライン統合し、新規`results/merged/<識別子>/`を作る。
不足・重複・未完了batchがあれば停止する。手動の`run.py --merge`を先に実行する必要はない。

全query試行済みだがAPIエラーを含む`complete_with_errors` batchは、失敗行・出典も含めて統合できる。ただし従来の`analyze.py`は全応答が必要なので、統合後に`failed_queries.jsonl`の確認を求めて停止する。API失敗を0点扱いしたり、黙って解析対象から除外したりしない。欠損を明示した4モデル比較は上記の新しい入口を使う。
05の40 query/modelは統合済みrecordsに含まれるので、05をさらに結合しない。
実行時に05の再利用内容もオフライン照合する。API呼び出し・元batch/05結果の上書きはしない。

既存の解析出力を保持して別保存先にする場合は、`--output-dir experiments/06_new_models_n10/analysis/<名前>`。
派生ファイルだけを同名で再生成する場合は`--overwrite-derived`を明示する。
定義・図・表の一覧は共通評価README、実験の送信手順は[実験README](../README.md)を参照。
