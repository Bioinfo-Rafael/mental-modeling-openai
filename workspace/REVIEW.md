# 静的レビュー（2026-09-10）

今回の検証はソース読解、AST構文解析、Git差分確認のみです。実験・dry-run・テストでの動作確認はしていません。以下はコード構造の確認であり、モデル応答や実promptの実測結果ではありません。

## 共通処理と公式関数

- 各 `run.py` は条件を宣言して `common.run()` を呼びます。04のみ `common.analysis_main()` → `analysis.main()`。
- plan/preview：既存 `discover_episodes()` → `build_prompt_queries()` → 公式 `Episode.load()`、`_query_indices()`、`_history_range()`、`system_prompt()`、`render_question()`、`user_prompt()` → `Episode.history_text()`。
- 有料評価：`common.invoke_cli()` → **公式 `llm_x.cli.main()` → `_evaluate()` → `EvaluationConfig` → `_backend()` → `evaluate_episode()` → Recording subclassの `complete()` → `super().complete()`**。
- `super().complete()` 内のSDK `create()`だけをwrapして記録し、実response objectをそのまま返します。評価は公式 `_score_response()` と `_summarize()` に戻ります。Exp.6のreuseもSDK responseを復元して同じ `super().complete()` / scorerへ渡します。
- 追加集計・Exp.4のsubset評価は保存済みの公式scoreを `_summarize()` に渡します。prompt・ground truth・parser・bin range・scoringを再実装していません。

## 依頼された13項目

| # | 確認項目 | 静的確認・根拠 |
| --- | --- | --- |
| 1 | upstreamを変更しない | 編集対象外。CLI内のbackend名だけを実行中patchし、終了時に復元。pycache書込も無効化 |
| 2 | dataを変更しない | reader経由の読込のみ。output_safety＋results限定のpath検証 |
| 3 | paid API 0件 | 今回API呼び出しなし。既定経路はplanのみ、二重guard後にのみclientを構築 |
| 4 | Exp.1/2＝8条件 | 1 model × 2 tasks × 4 metrics × pilot H × N1 |
| 5 | Exp.3＝960件 | 1 × 2 × 4 × 4＝32条件、各N30。件数はplanから集計 |
| 6 | Exp.4＝API 0 | 保存済み03のrecordsをsubset化するだけ。client/実行関数を呼ばない |
| 7 | Exp.5＝12件 | 3 models × Pendulum × H20 × 4 metrics × N1 |
| 8 | Exp.6＝316/model、948合計 | 32×10−4×1＝316/model。3モデルで948新規、12再利用、最終960 |
| 9 | 実履歴H＝5/10/20/30 | `history_size(H)` でH−1、wrapperが公式rangeを利用。range差と履歴のStep行数を実行時assert/検査。実測は未実行 |
| 10 | 正しいquestion | `QUESTIONS` をtask別に明示。MountainCar actionはdiscrete、Pendulum actionはcontinuous_bins、stateは指定の公式question |
| 11 | raw/usage/timing保存 | create kwargsとmodel_dumpをquery/attempt IDで対応。UTC時刻・各試行・query・condition・experimentの時間を記録 |
| 12 | credential非保存 | OPENAI_API_KEYは環境のみ。credential field/既知key/Bearerをredact、headerを保存せずSDKのdebug loggingを抑制 |
| 13 | Exp.5を重複送信せずreuse | 全12件を送信開始前に検証。missing/mismatch時は停止。SDK create前にもkwargs一致確認。reuse時のAPI fallbackなし |

## 意図的な実装上の差・未検証事項

- history parameterだけをH−1へ変換し、promptの文面/serialization・採点仕様は保持します。
- 可視化されないSDK内部retryを無効にし、upstream retryは設定可能・安全側で既定0。upstreamのretry実装自体はそのままです。未知の課金を避けるため自動resumeはありません。
- `api_attempts` はSDK呼び出し回数。ネットワーク切断時に課金が発生したかをこのコードだけで断定しません。追加usage fieldはraw responseに保持し、統計は最終responseのusageを使用します。
- APIエラーは途中停止。取得済みprefixの採点回収はAPIを使わないCLI replayです。中断/ディスク障害時は未採点分を成功扱いせず、summaryに未完了を示します。
- Exp.4のnested prefixはquery順を検証し、別sampleに置換しません。queryは同一episode由来で相関し得るため、図の標準偏差を信頼区間とは呼びません。
- modelの利用権限、Chat Completions/temperature互換性、SDK responseの実payload、図の描画結果は未検証です。05がその確認用pilotです。
- Loggingのpatchはプロセス内で一時的に適用するため、同じPythonプロセス内で別評価を並行実行する用途にはしません。
