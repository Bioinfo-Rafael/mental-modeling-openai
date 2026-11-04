# tactical_short_CountTime_v1

2026-10-26にdry runで5窓分を準備。APIはまだ送信していない。実行後の状態は [計測結果](evaluation/timing.json) を確認する。

- モデル：gpt-5.6-lunaのみ、reasoning_effort=medium。
- 送信順：中心step 45 → 85 → 145 → 245 → 310。
- 通常APIを1件ずつ実行し、前の回答を受信・保存してから次を送る。
- 各回答はA0100・A0200の2機を説明。入力は全4機とミサイルの61時点を維持。
- 文体と長さはshort v2のまま。原則1文、1機あたり合計55語以内。

## 実行

リポジトリルートで：

```sh
.venv/bin/python experiments/17_commentary_LAG/run.py \
  --prompt-variant tactical_short_counttime_v1 --execute
```

APIキーを端末で入力する。入力は非表示。`--wait`は不要。
各件の完了時に`Received center_XXXX: XX.XXX seconds`を表示し、最後に平均・中央値・最小・最大も表示する。

## 保存物

| 内容 | 保存先 |
|---|---|
| 実行条件と凍結した入力 | manifest.json |
| 今回のsystem / user | [system](prompts/templates/system.txt)、[user](prompts/templates/user.txt) |
| v2からの変更点 | [system diff](prompts/changes/system.diff)、[user diff](prompts/changes/user.diff) |
| 各リクエストの生回答・秒数 | api/responses.jsonl（実行後に作成） |
| 説明の出力・2機JSONの検証 | [responses.json](evaluation/responses.json) |
| 各件の秒数・トークン数・集計 | [timing.json](evaluation/timing.json) |
| 計測の定義と再開方法 | [詳細](prompts/changes/README.md) |

`elapsed_seconds`は、非ストリーミングのSDK呼び出し直前から完全な回答が返るまで。通信・サーバー待機・推論・生成・SDK応答解析を含む。ファイル保存時間、APIキー入力時間、クライアント作成時間、model_dumpは除く。純粋な推論時間や最初のトークンまでの時間ではない。

自動リトライなし。エラー時は失敗時間を記録して停止。同じコマンドで未試行分だけ再開する。成功済み・失敗済み・結果不明のリクエストは再送しないため、失敗があれば全件成功にはならない。結果を削除して再送しないこと。
Batchの待ち時間と今回の秒数は測っているものが違う。さらに出力を4機から2機に変更しているため、前回からの速度改善をモデルの純粋な性能差としては解釈しない。
