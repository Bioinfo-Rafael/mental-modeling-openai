# tactical_short_CountTime_v1

v2を出発点に、説明対象だけをA0100/A0200へ変更する。A1/A2はこの2機を指す。
入力は同じ全4機・全ミサイルの61時点。中心stepは45、85、145、245、310。
各窓について2機の説明を一つのJSONで求める。1機ずつ送るのではなく、1窓ずつ5リクエストを順番に送る。
1機あたり目安35〜50語・上限55語、原則1文の方針はv2を維持する。
system.diff/user.diffはtactical_short_batch_v2からの差分。baselineは比較元で、送信本文ではない。

実行条件はgpt-5.6-luna、reasoning_effort=medium、通常Chat Completions API、非ストリーミング、同時実行数1。
1件の回答を受け取って保存してから次を送る。自動リトライは0、SDKのtimeout設定は180秒。
計測はtime.perf_counterでSDK呼び出し直前から戻った直後まで。通信、サーバーの待機、推論・生成、SDKの応答解析を含み、入力ログ保存・出力ファイル保存・model_dumpは含めない。純粋な推論時間や最初のトークンまでの時間ではない。
SDKクライアントを共用するため初回と後続で接続再利用条件が違う可能性があり、入力キャッシュの有無も速度へ影響しうる。usageのcached_tokensとreasoning_tokensも保存する。
Batch時間との単純比較や、5件だけからの一般的な性能評価は行わない。4機→2機という回答量の変更も含む条件である。

結果はresults/tactical_short_CountTime_v1に保存する。
api/responses.jsonlに生回答とelapsed_seconds、evaluation/responses.jsonに2機形式の検証結果、evaluation/timing.jsonに各件の秒数と成功回答の平均・中央値・最小・最大を保存する。
APIエラーはその所要時間とエラーを保存して停止する。同じコマンドを再実行すると未試行分だけ再開し、成功済み・失敗済み・結果不明のリクエストは自動再送しない。失敗時間は成功回答の平均へ混ぜない。

dry run:
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical_short_counttime_v1 --dry-run

API実行（順番に待つので--waitは不要）:
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical_short_counttime_v1 --execute

APIキーはOPENAI_API_KEYまたは端末の非表示入力で渡す。プロンプトや記録ファイルに保存しない。
公式SDKのリトライ・timeout設定：https://developers.openai.com/api/reference/python
