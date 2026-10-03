# Exp.09: Pendulum Next Action reward ablation

推論レベル評価の準備は [reasoning_score_analysis/README.md](reasoning_score_analysis/README.md)。08と同じ採点基準の [Prompt.md](reasoning_score_analysis/Prompt.md) と、96件の [採点用CSV](reasoning_for_scoring.csv) を用意した。入力のscoreは空欄で保持し、96件の [採点済みCSV](reasoning_for_scoring_scored_astra_high.csv) を別途保存した。

評価対象の履歴からrewardを除いたときの性能変化を調べる。
Terra / Luna × H=5 / 20 × 履歴開始t=0 / 100 × 0-shot / 4-shot × rewardあり / なし × 3反復 = **96 API calls**。
対象は `episode_9_seed3407.npz` のみ。32条件、各3反復。同じ質問への独立したAPI呼び出しであり、異なる3問ではない。

## 実行

リポジトリの `.venv` を使用する。コマンド実行後にAPIキーの入力を求める（入力文字は表示しない）。
環境変数の設定は不要。設定済みでも対話入力したキーを使用し、キーをファイルへ保存しない。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python experiments/09_ablation_reward/run.py --dry-run
.venv/bin/python experiments/09_ablation_reward/run.py
```

**引数なしで96件を実行する（有料API送信）。** `--dry-run` は送信予定プロンプトだけを生成する。
実験ディレクトリから `../../.venv/bin/python run.py` でも同じ条件・出力先になる。
APIキーが空なら送信前に停止。`--dry-run` と送信不要の完了済み再開では入力を求めない。既存のAPI結果があれば引数なしでの再実行は拒否する。
途中再開は `--resume`。成功済み・送信済みは再送せず、未送信のqueryだけを送る。
保存済みresponseはオフラインで採点復元できる。エラー・結果不明queryはunresolvedとして残し、自動再送しない。
SDK retry=0、timeout=180秒。通信エラーで停止し、ログ・途中集計を残す。

## プロンプトと時点

| H | 開始 | 提示履歴 | 予測action |
|---|---|---|---|
| 5 | 0 | 0–4 | 5 |
| 20 | 0 | 0–19 | 20 |
| 5 | 100 | 100–104 | 105 |
| 20 | 100 | 100–119 | 120 |

予測対象時点のstateは既存Next Action質問に従って提示する。正解actionとその時点のrewardは提示しない。
Hを変えると予測対象も変わる。reward有無の比較は同じmodel/H/開始位置/shot/反復で対応付ける。
rewardなしでは評価対象の履歴の `reward:` 行だけを削除する。state/action・system・質問文は維持する。
**few-shot内のrewardは残る**ため、「評価対象の履歴に含まれるreward」のablationであり、全入力からreward情報を消す実験ではない。

few-shotは08の保存済み `shots=4 / pattern1` の接頭辞をそのまま固定。
正解2例・不正解2例、出典はepisode_0。last-actionの例も含め、08の選択・回答・ラベル・順序を変更しない。
`data_prep/fewshot_p1_k4.json` に全文・例ID・出典manifestとhash・prefix hashを保存している。
08が後から更新されてもこのsnapshotを使うため、実験条件は変わらない。
モデルIDとAPI引数は `experiments.common` を再利用（reasoning_effort=medium、temperature未指定）。
rewardあり→なしを各反復で隣接して実行する。反復番号はquery IDに含める。

## 構成・出力

- `run.py`: 条件と起動入口。
- `runner/planning.py`: 問題選択・96件の計画・プロンプト保存。
- `runner/prompts.py`: 評価履歴だけのreward削除。
- `runner/execution.py`: 08の実行処理を09用に適用。採点は08のscoringを直接再利用。
- `analysis/summarize.py`: 保存済み結果から再集計可能。
- `results/dry_run/`: manifest JSON/CSV、96件のsystem/userテキスト。
- `results/api/`: manifest、送受信JSONL、records JSONL/CSV、実行summary。
- `results/analysis/`: 条件別accuracy/MAE、rewardなし−ありの差分CSV/JSON。

bin accuracyはbin解析成功件数、action MAEは連続action解析成功件数を分母とする。
各分母・未送信・未回収・解析不能件数を明示。全予定件数を分母とするmatch rateも別記する。
差分は両側の全3反復が該当指標で有効な場合だけ算出する。
3反復は同一問題の繰り返しで、episode間の一般化性能や独立な3サンプルを意味しない。

```bash
.venv/bin/python -m pytest experiments/09_ablation_reward/tests -q
.venv/bin/python experiments/09_ablation_reward/analysis/summarize.py
```

### 作成に利用したプロンプト

```text
まずは実装する計画を立てて



09\_ablation\_reward\
&#x20;ってdirを/Users/cls-lab/Git/Matsuo/mental-modeling-openai/experimentsに作って以下を実装してほしい。\
&#x20;目的：rewardをなくしたときに性能が変わるかどうかを検証する

固定する条件：

- TaskはPendulumのNext Action

比較する条件：

- モデルはTerra, Luna
- 0 shotとfewshotを試す。fewshotはexperiments/08のk=4, P1のものを利用
- 履歴長H=5, 20
- timepoint:&#x20;
  - t=0始まりでepisode\_1\~episode\_9の9こをそれぞれ3回ずつ実施
  - t=100始まりでepisode\_1\~episode\_9の9こをそれぞれ3回ずつ実施

これでモデル２つ, H 2つ, timepoint２通りで9種類, n=3なので2*2*18\*3=212回分の実行になるはず

実験詳細：

実装についはeperiments/06, 05を参考にして。エントリポイントrun pyを用意して、キーワードなしで実行したときにしそのまま上記の実験条件がそのまま再現できるようになるってことと05,06のようにrun pyの可読性が高いこと。

fewshotについては08のものをそのまま再利用して
```
```text
episode\_9だけでいい。0 shot fewshot reward有無は全部やる。
すると９６回になるかな？
では実装して
```
```text
比較したい軸は
1. timepoint t=0と100の間で結果が変わるのか
→rewardの変化がどのくらい影響してるか
2. reward あり・なしの間でどのくらい結果が変わるか
→rewardがそもそも影響するのか
あとはこれを細かく分解していくだけ。
作図するものは以下について正解率・誤差・推論レベルをそれぞれ描写してほしい
(1) 棒グラフでtimepointの開始点がt=0のものとt=100のものを比較するやつ。modelは２系列で比較。左にt=0で右にt=100で、左側の中で２モデルに対応する棒グラフを並べる感じで右も同様。これで棒１つあたり96/4個分の値が出るはず
(2) (1)についてrewardあり・なしで同じことをする。左がrewardありで右がなし。
(3) (1)と(2)を組み合わせた比較
これはモデルに対応して１つずつ作る感じでいい。subplotsで縦二行にして上がTerraでしたlunaって感じ。
で、そのsubplots１つについて以下のようにする
左側はrewardありで右がrewardなし。左のなかでさらに左右で分けて開始の位置がt=0を左でt=100を右。

96/８個分が１つの棒に対応するはず

(4)あとは色付きのmatrixを描きたい
まず行については階層型で上側が0 shot or 4shotで下側がH=5 or 20で四行分になる。
列について上側がrewardの有無で下側が開始点がt=0,100
これについてモデルをpoolしたものをまず作る。なので１マスあたり96/16通りが対応してるはず。
そして、モデルごとに分けてそれぞれ作る。これで１マスあたりn=3なのでこれ以上分けようがないね。

まずはどう作成するかについて計画を立てて
```