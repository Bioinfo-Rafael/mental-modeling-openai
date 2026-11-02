# 19_RealTime_LAG — 目的選択と録画の遅延付きリプレイ

作成日: 2026-10-26 JST。

**送信CLI・機体別並列計測・3秒判定・録画ビューア統合を実装済み。実APIの実行状況は各runのsummary.jsonで確認する。**

5stepごとにDecisions APIで `NONE / A1 / A2 / BOTH` を判定する。
`NONE` 以外なら、最初の回答がA1だけでもA0100とA0200の両方について目的を判定する。
機体別の2リクエストは並列に送り、4・8・32択の3条件を比較する。
解説は選択された目的の定型文だけを表示する。各API呼び出しが**3秒を超えた場合はその結果を棄却**する。
2段階の合計を3秒に制限する実験ではない。期限の解釈はユーザー確認済み。

APIは録画再生とは別に実行し、実測遅延を保存する。その後、13の録画フレームと目的の到着イベントをHTMLで同期再生する。
ライブ送信・MP4への書き出しは初回実装の範囲に含めない。

- [実験計画](PLAN.md): 測定、3秒棄却、時刻計算、表示順、比較指標、実装段階。
- [選択肢一覧](choices/README.md): 合意した4・8・32択。
- [分類の正本](choices/catalog.json): 表示文、定義、親分類、共通判定規則。
- [4択](choices/4.json) / [8択](choices/8.json) / [32択](choices/32.json): APIの `choices` に渡す `value / description` 配列。
- [実験設定](configs/experiment.json): 実行時にrunへ固定する設定。

選択肢は優先順位や強弱ではなく、名義カテゴリとして扱う。IDのA/B/C/Dは目的分類であり、機体のチーム名ではない。
ゲートの `A1` と、8択目的の `A1` は異なる名前空間なので、保存・表示では `gate_choice` と `goal_choice` を必ず分ける。

参照元:

- `../13_LAG/viewer/`: 録画フレーム、step時計、localhostサーバー。
- `../18_desicion_LAG/`: H=40、stride=5の入力構築、v003ゲート、応答・実測時間の記録。
- `../17_commentary_LAG/`: 目的推定の環境説明・欠損値・根拠の扱い。自由文出力契約は流用しない。

## 実行方法

リポジトリ直下から実行する。planとexportはAPIに送信しない。

```sh
# 62窓、1反復。ゲート62件＋有効なゲート陽性ごとに最大6件。
.venv/bin/python experiments/19_RealTime_LAG/run.py plan --run pilot_20261026_v2 --repeats 1

# 実API送信。環境変数がなければターミナルでAPIキーを非表示入力する。
.venv/bin/python experiments/19_RealTime_LAG/run.py execute --run pilot_20261026_v2

# 同じ起動コマンドで、右側の「推定目的」から4/8/32択を切り替える。
.venv/bin/python experiments/13_LAG/viewer/serve.py
```

planは同名runの上書きを拒否する。保存済みrunを使う場合はplanを再実行しない。
実行中もゲート・機体ペアごとに結果を公開する。開いたビューアでは「結果を更新」で再読込できる。
各機体は返答到着時刻から表示される。各APIで3秒超の回答は表示しないが、生応答と実測時間は保存する。
HTTP打切りは30秒、SDKの自動リトライは0。30秒打切りを正常な応答時間には数えない。

```sh
# 状態確認と再集計（API送信なし）
.venv/bin/python experiments/19_RealTime_LAG/run.py status --run pilot_20261026_v2

# 中断後は、開始済みリクエストを再送せず未送信分だけ続ける
.venv/bin/python experiments/19_RealTime_LAG/run.py execute --run pilot_20261026_v2 --resume

# 閲覧キャッシュを再生成（API送信なし）
.venv/bin/python experiments/19_RealTime_LAG/run.py export --run pilot_20261026_v2

# 保存した送信本文を確認（API送信なし）
.venv/bin/python experiments/19_RealTime_LAG/run.py request --run pilot_20261026_v2
```

目的は強制選択による推定。正解ラベルは自動生成しない。
`report/review_template.csv` をコピーして人手評価を記入できる。
`report/summary.json`、`report/latency.csv`、`responses/`、`requests.jsonl` に記録を残す。
費用は18の料金スナップショットに基づくusage推定で、請求書との照合値ではない。

実測遅延を仮想の同時進行へ貼り付けた再生であり、毎秒の実送信時の混雑や待ち行列を実測したものではない。
目的条件は1つずつ計測し、同じ条件内の2機だけを並列送信する。ゲートは3条件で共用する。

## GPT-5.6 solによるA1・A2の自由文解説

`sol_tactical_short_v2_20261026` を準備済み。`pilot_20261026_v2` のゲートがA1/A2/BOTHだった43窓を使用し、ゲートAPIは再送しない。
1窓につき1リクエストでA0100・A0200の2つの文字列を返す。全43呼び出し・86機分の説明を対象にする。
選択式目的の重複対策は今回追加しない。

```sh
# 準備済みrunを実行（APIキーを設定したターミナルで）
.venv/bin/python experiments/19_RealTime_LAG/run_narrative.py execute --run sol_tactical_short_v2_20261026

# 中断後に未送信分を続ける。開始済みリクエストは再送しない。
.venv/bin/python experiments/19_RealTime_LAG/run_narrative.py execute --run sol_tactical_short_v2_20261026 --resume

# 記録済みの失敗も1回ずつ再試行し、未送信分を続ける（成功済みは再送しない）
.venv/bin/python experiments/19_RealTime_LAG/run_narrative.py execute --run sol_tactical_short_v2_20261026 --resume --retry-failed

# 保存済みの進捗確認（API送信なし）
.venv/bin/python experiments/19_RealTime_LAG/run_narrative.py status --run sol_tactical_short_v2_20261026
```

プロンプトの出典は17の `results/tactical_short_batch_v2/prompts/templates/`。目的・行動・根拠を短く結ぶ本文、英語、原則1文・各55語以内、`reasoning_effort=medium` を維持する。
入力は19の検知時点までの40時点に変更し、元の前後61時点の記述を修正。出力フィールドはA0100/A0200だけにする。敵機・ミサイルは入力の背景情報として維持する。
原本と変更差分は新runの `prompts/source/`、`prompts/*.diff` に保存。元の17のプロンプトと結果は変更しない。

17と同じChat Completionsで `gpt-5.6-sol` を指定する。Batch APIは使わず、各呼び出しの送信直前から完全な応答受信までを `perf_counter()` で測る。最初のトークンまでの時間ではない。2機が同じ応答に入るため応答時間は共通。
SDK自動リトライは0、通信打切りは180秒。通信エラー・無効JSONも生応答とともに記録する。正常なJSONを得られなかった呼び出しは成功件数に含めない。
通常の`--resume`は記録済みの失敗もスキップする。`--resume --retry-failed`を明示すると、通信エラー・タイムアウト・無効出力だけを各起動で1回ずつ再試行する。過去の記録は`attempts/<query_id>/attempt_NNN.json`と追記ログに残し、`responses/`には最新の試行結果を保存する。成功済み・応答不明の開始済み呼び出しは再送しない。通信断ではサーバー側の処理完了有無は分からず、再送時に再度処理される可能性がある。時間の統計には成功試行の応答時間を使い、失敗までの待ち時間は各試行ログで確認する。

**ユーザー指定により、この自由文条件では応答時間を表示時刻の計算や3秒棄却に使わない。** 窓末尾の観測時刻からA1/A2の説明を表示し、カードと履歴にAPI応答時間を併記する。同じ説明も各窓の回答として履歴に残す。
ビューアの「推定目的」タブで実験runから **GPT-5.6 sol · A1/A2解説** を選ぶ。このrunでは4/8/32択欄を隠し、既存runに戻すと再表示する。実行後は「結果を更新」、コード更新後に開きっぱなしの画面ではブラウザを再読み込みする。

保存先はrun内の `responses/`（生応答・解説・実測時間）、`requests.jsonl`（開始記録）、`report/answers.csv`（86行、各機体の本文と共有応答時間）、`report/summary.json`（成功数・中央値・p95）、`replay.json`。
キーはファイルへ保存しない。こちらの実行環境にはキーがないため、準備時点で実API送信は未実施。

モデル名とChat Completions対応は[公式モデルページ](https://developers.openai.com/api/docs/models/gpt-5.6-sol)で2026-10-26に確認。

## 検証コマンド

```sh
.venv/bin/python -m pytest experiments/19_RealTime_LAG/tests -q
node --test experiments/13_LAG/viewer/test_model.mjs experiments/13_LAG/viewer/test_commentary.mjs
```

13へは目的タブを追加した。17・18の既存コードと結果は変更していない。
