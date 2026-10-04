# 11_ICRL_n10 — numeric history reward ablation

10_ICRLの **ICR-FQI K=5** instructionを固定し、trajectory historyのstep-wise numeric reward observationの有無によるNext Action予測性能の差を調べる。

- **40 calls = 2 models (`terra`, `luna`) × 2 reward modes × N10**。
- `Pendulum-v1` / `next-action` / `H=20` / `E=1`（追加referenceなし）。
- N10は異なる10 target query。`episode_0.npz` の `ordinal=0..9`、`history_start=0..9`、`history_end_exclusive=query_index=20..29`。例えばordinal 1はsteps 1..20を見てs21からa21を予測する。同じqueryの反復ではない。
- 05のTerra/Luna H20、06が05から再利用したH20、03_1のH20先頭10件と同じwindow。実際の05/03_1 `results/manifest.csv` と06 `results/merged/*/manifest.csv` をテストで照合し、episode SHA256、window、system・元target promptのSHA256まで確認する。実行時にはこれらのmanifestを読まず、path順の先頭episodeから既存 `build_prompt_queries()` の有効query先頭10件を生成する。

`with_reward` は10の `icrfqi_k5`, E=1と同じ構成。`without_reward` はtarget historyの20本の `  reward: <numeric value>` 行だけを削除する。system、task説明、state/action、history範囲、question、Joint answer format、API parametersは同一。

rewardなしでもsystem内のreward function説明とICR-FQI内の `(s_t, a_t, r_t, s_{t+1})`、Bellman式は残す。rewardを0にする・推定する等の指示は足さない。したがってreward情報を完全にゼロにする実験ではなく、**明示されたnumeric history rewardだけのablation**。K=5、gamma=0.99、Q_0=0、既存10 action binsによるFQI・argmaxとbin内continuous予測は10のpromptをそのまま使用する。

実装は `run.py → runner.py` の浅い構成。再利用元は以下。

- **10_ICRL**: `prompts.compose_user / METHOD_K / ICRFQI`、`runner.scoring`、`runner.summarize`。集計の分母・token/time集計を維持し、11側でreward metadataとpaired identity、N10の説明を補う。
- **09_ablation_reward**: `runner.prompts.without_rewards`、`runner.execution.execute`（key入力、retry=0、timeout=180秒、journaling、resume）。
- **共通**: `build_prompt_queries`、`JOINT_QUESTIONS`、`common.expected_api_request`（`reasoning_effort=medium`）、`execution_lock`。採点は10と同じ既存Pendulum Joint scorer/parser。

リポジトリrootから実行（必要なら先に `.venv` を有効化）。

```bash
python experiments/11_ICRL_n10/run.py --dry-run
python experiments/11_ICRL_n10/run.py
python experiments/11_ICRL_n10/run.py --resume
```

dry-runはAPI clientを作成・送信せず、`results/dry_run/manifest.json`・`manifest.csv` と40件の `prompts/*__query00.txt` 等を保存する。manifest内の `request` は送信予定パラメータの辞書。変更された既存previewや既存API runの無指定上書きは拒否する。resumeは未試行queryのみ送信し、送信済み失敗queryは自動再送しない。

実API実行後は `results/api/` にmanifest、requests/responses/recordsのJSONL、records.csv、実行summaryを保存。`results/analysis/conditions.csv`・`summary.json` は4条件それぞれN10のbin accuracy、continuous action MAE、parse数、token数・時間を集計する。accuracyはbinをparseできた回答、MAEはnumeric actionをparseできた回答を分母とし、未試行・未解決件数も残す。

`analysis/per_query.csv` とrecordsにはground truth/prediction（continuous・bin）、absolute error、bin match (`status`)、回答、token数、時間、query identityを保持する。parse successは `analysis/per_query.csv` に保存。`pair_id` は同じmodel/ordinal/episode/windowのrewardあり・なしを結び、analysisにも保持する。10 windowsは同一episode内で重なっている。

```bash
python -m pytest -q experiments/11_ICRL_n10/tests experiments/10_ICRL/tests/test_icrl.py experiments/09_ablation_reward/tests/test_ablation.py tests/test_joint_modern_experiments.py tests/test_raw_preprocessing.py
```

実装確認はoffline unit/regression testsとdry-runのみ。テスト内のexecution/resume検証もmock clientを使用する。

## 回答閲覧用HTML

[view_rawdata.html](view_rawdata.html) は40回答を埋め込んだ単一のオフラインHTML。10_ICRLのテンプレート・安全なJSON埋め込みを再利用し、モデル／reward有無／query番号／採点結果での絞り込み、全文検索、回答一覧、system/user/回答/正解/時間/tokenの詳細、回答JSON保存に対応する。詳細画面では同じmodel・queryのrewardあり（左）／なし（右）を並べる。推論レベルは未評価として表示する。

生成前にmanifest・request/response/recordの一致と20 rewardペアのprompt差分を検証する。API送信は行わず、既存HTMLも上書きしない。再生成は別名を指定する。

```bash
.venv/bin/python experiments/11_ICRL_n10/build_results_viewer.py --output experiments/11_ICRL_n10/view_rawdata_v2.html
```
