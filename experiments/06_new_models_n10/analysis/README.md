# Exp.06の解析

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
05の40 query/modelは統合済みrecordsに含まれるので、05をさらに結合しない。
実行時に05の再利用内容もオフライン照合する。API呼び出し・元batch/05結果の上書きはしない。

既存の解析出力を保持して別保存先にする場合は、`--output-dir experiments/06_new_models_n10/analysis/<名前>`。
派生ファイルだけを同名で再生成する場合は`--overwrite-derived`を明示する。
定義・図・表の一覧は共通評価README、実験の送信手順は[実験README](../README.md)を参照。
