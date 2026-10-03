# @Rafaメモ
- Fewshotで実験。fewshotに使う例は07で表示されているものから選択。144件のAPI送信
- run.pyがrunner/fewshot_runner.pyの中のrunを呼び出し、そこで実行。make_plans関数が条件ごとにfewshotのプロンプトを作成しcompose_user関数でデフォルトのuser promptの上に追加する

# Exp.8 — Next Action few-shot、Terra / Luna、N=3

07で選択した正解例・不正解例を、同じTaskのNext Action質問の前に入れる。
例数とReasoning scoreの組み合わせを変え、送信予定のsystem/user全文を確認する。
**`python run.py` はdry-run。`--execute --confirm-paid-api` を付けた場合だけAPIへ送信する。**

## ファイル構成と役割

`run.py`は実験条件を設定する入口。処理本体は`runner/`、結果の集計・表示は`analysis/`、実験のテストは`tests/`に置く。`data_prep/`は候補データの準備を担当する。

```text
08_fewshot/
├── README.md                    # 構成・実行手順・実験条件
├── run.py                       # 実験条件・起動入口
├── runner/                      # 実験計画・API実行
│   ├── fewshot_runner.py
│   ├── api_execution.py
│   └── scoring.py
├── analysis/                    # 保存済み結果の集計・表示用コード
│   ├── rescore_results.py
│   ├── summarize_results.py
│   ├── plot_results.py
│   ├── plot_pooled_comparisons.py
│   ├── plot_previous_comparison.py
│   └── build_results_viewer.py
├── tests/
│   └── test_fewshot_runner.py
├── data_prep/                    # 元データ・候補の準備
│   ├── build_examples.py
│   ├── select_by_id.py
│   ├── prepare_prompt.py
│   ├── report_fixed_pools.py
│   └── test_data_prep.py
├── view_rawdata.html             # 入出力・採点結果の閲覧画面
├── templates/
│   └── results_viewer_template.html # 回答ビューアの生成用テンプレート
├── reasoning_for_scoring.csv     # Reasoning採点用データ
├── reasoning_score_analysis/     # Reasoning score分析用の資料
└── results/                      # 生成された実験結果・表・グラフ
    ├── dry_run/                  # 送信計画・プロンプト・プレビュー
    │   ├── preview_template.html # プレビュー生成用テンプレート（再生成時も保持）
    │   └── preview.html          # 送信前の入力を確認する画面
    ├── api/                      # API送受信記録・回答・採点結果
    ├── scoring_backups/          # 再採点前のrecords・summaryの原本
    └── analysis/                 # 正解率表・グラフ
```

上のツリーは主要ファイルとディレクトリの構成メモ（2026-09-30更新）。`data_prep/`内のJSON・CSV・説明資料と、自動生成される`__pycache__/`は省略している。`analysis/`はコードの置き場で、生成した表・グラフは`results/analysis/`に保存する。

| ファイル | 役割・入出力 |
|---|---|
| [runner/fewshot_runner.py](runner/fewshot_runner.py) | 実験全体を組み立てる本体。候補JSONの検証、few-shot例と評価問題の選択、プロンプト生成を行い、`results/dry_run/`へ送信計画・確認用HTMLを保存する。実行フラグがある場合はAPI実行処理を呼ぶ。 |
| [runner/api_execution.py](runner/api_execution.py) | dry-runとの一致を確認してAPIへ送信し、回答・正誤採点・トークン数・エラーを`results/api/`へ保存する。途中再開と二重送信防止も担当。Reasoning scoreの再採点は別工程。 |
| [runner/scoring.py](runner/scoring.py) | 08共通の採点。最終action/bin IDを正解IDと比較し、連続トルク予測と絶対誤差を別フィールドに保存する。API実行・オフライン再採点・pool集計で共有する。 |
| [analysis/rescore_results.py](analysis/rescore_results.py) | 保存済み回答をAPI再送なしで再採点し、records JSONL/CSVとsummaryを更新する。旧ファイルは`results/scoring_backups/`へ保存する。 |
| [analysis/summarize_results.py](analysis/summarize_results.py) | 保存済みAPI結果をTask・H・Scoreパターン・例数・モデル別に集計し、正解率表を`results/analysis/accuracy_tables.*`へHTML・Markdown・JSONで保存する。 |
| [analysis/plot_results.py](analysis/plot_results.py) | 正解率表を更新してから、モデル・例数ごとの比較グラフを`results/analysis/figures/`へPNGで保存する。 |
| [analysis/plot_pooled_comparisons.py](analysis/plot_pooled_comparisons.py) | Task・モデルごとに他の2条件をpoolして、Scoreパターン・例数・H別の正解率、平均絶対誤差、平均推論スコアを比較する。最終action/bin IDを再解析し、`results/analysis/pooled_comparisons/`へ図・集計CSV・検証情報を保存する。 |
| [analysis/plot_previous_comparison.py](analysis/plot_previous_comparison.py) | 既存07の各条件n=10と08のn=3を比較する。Hをpoolし、全条件・例数pool・パターンpool・Terra/Luna全poolの4種類についてTask別一覧図と拡大PNGを作る。追加APIなし。 |
| [analysis/build_results_viewer.py](analysis/build_results_viewer.py) | 送信計画・回答・採点・送信状況をまとめ、1件ごとの質問・回答を確認できる`view_rawdata.html`を生成する。 |
| [data_prep/build_examples.py](data_prep/build_examples.py) | 03_1・06の過去結果とReasoning scoreのCSVを照合し、質問・回答・正解・スコア・出典を含む`data_prep/examples.json`と検証情報を生成する。07側の`view_rawdata.html`も再生成する。この段階では採用区分は未選択。 |
| [data_prep/select_by_id.py](data_prep/select_by_id.py) | MountainCarの行動ID／Pendulumのbin IDの一致で正解・不正解を分類し、Task・モデル・Scoreごとの規則で候補を自動選択する。候補JSONと不足数などのレポートを生成する。 |
| [data_prep/prepare_prompt.py](data_prep/prepare_prompt.py) | 選択例を`Question / Answer / Label`形式へ整形する。単独実行では条件を指定して例を抽出しTXT・JSONを生成でき、runnerも整形関数を利用する。 |
| [data_prep/report_fixed_pools.py](data_prep/report_fixed_pools.py) | dry-runの`pattern_examples.json`から実際に選ばれた例の内訳を集計し、`fixed_pattern_examples.json`・`fixed_pattern_counts.csv`と、このREADMEの内訳表を更新する。 |
| [tests/test_fewshot_runner.py](tests/test_fewshot_runner.py) | 実験条件、例の選択、プロンプト、入力検証、API結果の保存・再開・二重送信防止を検証する。API部分は疑似応答を使う。 |
| [data_prep/test_data_prep.py](data_prep/test_data_prep.py) | 元データの件数・構造、正解値、解析失敗の扱い、例の抽出・プロンプト整形を検証する。 |

処理の流れは、過去結果 → `build_examples.py` → 候補選択（07のビューアまたは`select_by_id.py`）→ `run.py` → `runner/fewshot_runner.py` → dry-run確認 → `runner/api_execution.py` → 結果の集計・表示。
`select_by_id.py`が作るのは候補集で、各条件に使う4・8・12例を決めるのは`fewshot_runner.py`。

通常は08のディレクトリから以下を実行する。集計・グラフ・ビューア生成は保存済みデータを使い、API送信は行わない。

```bash
python run.py
python analysis/summarize_results.py
python analysis/plot_results.py
python analysis/build_results_viewer.py
```

入力ファイルの相対パスは引き続き`08_fewshot/`基準。テンプレート・データ・結果の保存先も同じ基準で解決する。

## 1. 条件・予定件数

| 軸 | 値 |
|---|---|
| 評価Task | MountainCar-v0 / Pendulum-v1 |
| 評価対象 | **next-actionのみ** |
| 評価モデル | terra → luna |
| 評価履歴H | 5 / 20（実際の履歴step数） |
| few-shot総数 `2k` | 4 / 8 / 12 |
| 正解例：不正解例 | 1：1。各2 / 4 / 6件 |
| Scoreパターン1 | 正解例Score 1のみ、不正解例Score 3以上（3・4） |
| Scoreパターン2 | 正解例Score 2以下（1・2）、不正解例Score 3以下（1・2・3） |
| N | 各条件で異なる3問 |
| 例の選択 | モデル配分を固定し、next-actionを優先（balanced_next） |

2 Task × 2 Model × 2 H × 3例数 × 2パターン = **48条件**。
48条件 × N3 = **144件の送信予定プロンプト**。1 Taskなら72件。
dry-run時のAPI呼び出し数は0。有料実行は既定で144件を順番に送信する。

条件は [run.py](run.py) の `EXPERIMENT` に集約している。06と同じ短い入口から、この実験専用の [fewshot_runner.py](runner/fewshot_runner.py) を呼ぶ。
Scoreの範囲は境界値を含む。`run.py` の `ScorePattern.correct_scores` / `incorrect_scores` で対象Scoreを列挙し、各区分の候補をその集合で絞り込む。Scoreごとの均等割当は行わない。パターン間で範囲が重なるので、同じ例を両パターンで利用してよい。

## 2. 実行方法

既存の `.venv` を使う。初回のみrepository rootで環境を有効化してから、08へ移動する。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
cd experiments/08_fewshot
python run.py
```

以後08のディレクトリでは **`python run.py` だけ**で実行できる。repository rootから `python experiments/08_fewshot/run.py` でも同じ結果になる。`--dry-run` は明示してもよい。

実行のたびに `results/dry_run/` 直下へ出力し、最後に `preview.html` の絶対パスを表示する。生成完了後、このディレクトリの生成物を置き換える。`preview_template.html`は編集用の原本として保持する。

### APIへ送信する（手動実行）

プレビュー確認後、以下をターミナル（zsh）で実行する。`OPENAI_API_KEY` が未設定の場合だけ、非表示入力でキーを尋ねる。既存のモデルID・reasoning_effortを維持し、Terra → Lunaの順で144件を1件ずつ送信する。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
cd experiments/08_fewshot
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rs "OPENAI_API_KEY?OpenAI API key: "
  echo
  export OPENAI_API_KEY
fi
python run.py --execute --confirm-paid-api
```

[api_execution.py](runner/api_execution.py) が `results/api/` に保存する。`dry_run/` の再生成・削除でAPI結果は消えない。

- `manifest.json`: 確認済みの送信計画、モデル・条件・例ID・プロンプト全文。
- `requests.jsonl` / `responses.jsonl`: 送信前の記録、API応答全文・時間・エラー。1件ごとにディスクへ確定保存。
- `records.jsonl` / `records.csv`: 回答、usage/token数、`runner/scoring.py`によるprediction・ground_truth・match/mismatch/ignored。Pendulumのpredictionは最終bin ID、連続トルクはaction_value_predictionとして別に保存。
- `summary.json`: 保存済み・未送信・結果未回収件数と正誤件数。Reasoning scoreの再評価は別工程。

SDK自動retryは0、timeoutは180秒。APIエラー・Ctrl-Cで停止し、自動再送しない。途中停止後は同じ送信コマンドの末尾へ `--resume` を付けると、保存済み応答を再利用して**未送信のqueryのみ**続ける。送信記録があるエラー・結果不明queryも再送しないため、その分は `unresolved_queries` に残る。全件完了済みなら再送は0件。`--resume` なしの二重実行は拒否する。

入力や設定がプレビューと変わった場合は送信前に停止する。その場合は `python run.py` で再生成し、確認してから送信する。API実行開始後は同一の計画でのみ再開できる。

### 選択済みJSONを配置する

07の [view_rawdata.html](../07_view_rawdata/view_rawdata.html) で例を選び、**出力 → 候補集JSON（全選択）**を [data_prep/](data_prep/) に保存する。未選択の `examples.json` は自動選択には使わず、export内容の照合にだけ使用する。

- `kind=candidate_pool` のJSONが1つなら、ファイル名に依存せず自動で読む。
- 複数あれば自動で最新ファイルを選ばず、一覧を表示して停止する。`run.py` の `INPUT_FILES` に使うファイルを指定すれば、以後も `python run.py` だけで実行できる。
- 複数ファイルを明示した場合、同じquery_idで区分・メモが一致する例は1件にまとめる。区分・メモが競合すれば停止する。
- `fewshot_subset` 形式は自動検出しないが、`INPUT_FILES` または `--input` で明示すれば利用できる。
- 入力・回答・Task・予測対象・モデル・H・正解・Score・出典等を共通データ `examples.json` と照合する。07で指定した正解／不正解とメモはそのまま使う。回答なし、未選択、未知ID、内容不一致は拒否する。

指定例（パスは08のディレクトリを基準とする）：

```python
INPUT_FILES = ('data_prep/selected_examples_....json',)
```

一度だけの指定や範囲の限定も可能：

```bash
python run.py --input data_prep/selected_examples_....json
python run.py --task Pendulum-v1
python run.py --model terra --history 5 --shots 4
```

`--input` は絶対パスも使える。`--task` / `--model` / `--history` / `--shots` は複数値を指定可能。その他の条件はrun.pyで変更する。

行動IDによる自動選択JSONも利用できる。ユーザー指定の規則で生成した `data_prep/selected_examples_by_id.json`（74件）が現在の候補集。選択方法と不足の内訳は [data_prepの選択結果](data_prep/selected_examples_by_id.md) を参照。元のScore・正解は保持し、区分はaction ID／bin IDの一致で決めている。

### JSON未配置／候補不足の場合

ランナー自体は未選択データから勝手に例を補充したり、正解／不正解のラベルを付けたりしない。予定条件・不足表・理由を保存し、終了コード2で終了する。

| status | 意味 |
|---|---|
| `input_required` | 選択済みJSONが未配置。送信予定プロンプトは0件 |
| `insufficient_candidates` | Task・パターン・区分のいずれかの例数が不足。全条件の生成を保留 |
| `invalid_input` | 入力の不一致・複数候補ファイル未指定・評価問題不足等。理由を保存 |
| `ready` | 全条件の送信予定プロンプトを生成。終了コード0。**送信済みではない** |

既定の12例条件を満たすには、Taskごとに次の各候補群が6件以上必要（範囲内の合計で数える）。

| パターン | 正解例の候補群 | 不正解例の候補群 |
|---|---|---|
| 1 | Score 1が6件以上 | Score 3・4を合わせて6件以上 |
| 2 | Score 1・2を合わせて6件以上 | Score 1・2・3を合わせて6件以上 |

候補群は重複するため、別々に24件を用意する必要はない。例えばTaskごとに正解Score 1を6件、不正解Score 3を6件選べば両パターンを満たすが、この場合は両パターンの例集合が同じになる。1回に入れる例数は4 / 8 / 12件。
不足表は `candidate_inventory.csv` と `preview.html` に表示する。

## 3. few-shot例と評価問題の選択

### few-shot例

- **Taskだけを評価問題と一致**させる。例の予測対象、元モデル、Hは制限しない。
- 例の正解／不正解区分は07での手動選択を使用する。元の旧parserのstatusで分類し直さない。
- パターンごとに、正解例・不正解例を指定Scoreの集合で絞り込む。
- パターンごとに正解・不正解を各6件へ固定する。モデル配分はパターン1の正解がsol/terra/luna各2件、それ以外はsol 1・terra 1・luna 2・GPT-3.5 2件。
- 各モデルの中でnext-actionを優先し、同順位はH・episode・index・ordinal・ID順。パターン1の不正解にS4、パターン2の正解にS2を少なくとも1件残し、条件を満たす範囲でlast-actionを最小にする。
- モデルをsol→terra→luna→3.5で巡回し、その固定順から先頭k件ずつ使う。元の74件は残し、今回使わない候補は使用集合から除く。
- 順番は「正解例、不正解例」の交互。4例は8例の先頭部分、8例は12例の先頭部分になる。
- 同じTask・Scoreパターン・例数の例集合と順序は、評価モデル・H・3問の間で固定する。入力ファイル順序が変わっても固定順は変わらない。seedは従来のseeded方式用に設定を残しているが、現在のbalanced_next方式では使わない。
- query_idが異なるモデル別の回答は別例として数える。同一の物理的な問題への別回答を1問にまとめる処理はしない。

### 評価問題

- 保存済み回答を再利用せず、公式raw episodeから元のLLM-Xavier経路で質問を生成する。
- 読み込んだ例候補で使われているepisodeを全て除外する。パスだけでなく同一ファイルSHAも除外する。
- 残るepisodeをpath順に調べ、最大Hでも有効な先頭3問を採用する。足りなければ次のepisodeへ進む。historyがepisode境界をまたぐことはない。
- 同じTaskでは、モデル・H・例数・Scoreパターンを変えても**同じepisode/indexの3問**を使う。Hを変えると履歴範囲だけ変わる。
- 現在の03_1・05/06の候補はepisode_0由来なので、通常はepisode_1以降が評価対象になる。固定ファイル名に依存せず候補の出典から除外する。

## 4. プロンプトとAPI設定の計画

systemと元のuserは `preprocessing.llmx_original.build_prompt_queries()` → 同梱LLM-Xavierの既存関数を使う。Task別のJoint next-action質問は03_1・05・06と同じ `JOINT_QUESTIONS` を使う。Hから上流history_sizeへの変換は `common.history_size()` を利用する。

**systemは変更しない。** userの次の文より前にfew-shotブロックを追加する。

```text
The following are past Question / Answer pairs and evaluation Labels for those Answers.

Use Correct examples as references for your answer.
Use Incorrect examples to avoid mistakes, and do not imitate their answers or reasoning.
Labels are evaluation information and must not be included in your output.

Answer only the final New Question, and follow the output format specified in that question.

Question:
例のuser入力全文（systemなし）

Answer:
例のモデル回答全文

Label:
例の区分、Reasoning score、正解、比較結果

……指定数の例……

New Question:

Given the following snippet of an episode generated by a trained reinforcement-learning agent:

今回の履歴・質問（元のuser全文）
```

Question/Answer/Labelは既存 `data_prep/prepare_prompt.py` を再利用。モデル回答は不正解例でも編集しない。例の正解はLabelに含めるが、**今回回答する問題の正解は入力に含めない**。今回の質問は別の「user全文」として二重送信するのではなく、few-shot付きの1つのuserメッセージにする。

`manifest.json` の各queryには送信予定の `request` を保存する。06と同じモデルID・Chat Completionsのmessages形式・`reasoning_effort="medium"` を用い、temperatureは指定しない。dry-runではclientを作らない。`--execute --confirm-paid-api` 指定時のみ、保存済みdry-runとの一致を確認して、そのrequestを送信する。

ローカルtoken数は `o200k_base` によるsystem + 最終userのcontent計測。APIのmessage付加分を含まず、実際のserver usage・課金見積もりではない。モデル可用性のAPI確認は行わない。

## 5. 保存内容・確認方法

```text
results/dry_run/
├── README.md
├── preview.html
├── manifest.json
├── manifest.csv
├── conditions.csv
├── candidate_inventory.csv
├── selected_candidates.json
└── prompts/<condition_id>/query_01.txt ... query_03.txt
```

**最初に `preview.html` を開く。** APIや外部通信は不要。readyならTask・next-action・モデル・H・例数・Scoreパターン・評価問題を切り替えられる。

- 今回使う例の一覧：区分・Score・予測対象・元モデル・H・IDを表示
- 入力1：system全文
- 入力2：**few-shot込みのuser全文**。改行をそのまま表示し、長文を折り返す
- 今回の質問だけ：user末尾を再表示する確認用の欄。追加の送信メッセージではない
- TXTリンク：同じ送信予定system/user全文を開く
- 条件・API設定・各種ID・入力content token数

`manifest.json`（schema_version=2）は `correct_scores` / `incorrect_scores` を配列で保存する。不足表の `allowed_scores` も配列であり、その集合の合計候補数を表示する。全queryの入力全文・最終request・例IDの順序・評価episode/index・元user・SHAを保持する。few-shot後のuser hashと独自query_idを記録するため、06のzero-shotのqueryとは区別する。元入力ファイル・評価episodeのSHA、コードSHA、seed、除外方針も記録する。
`selected_candidates.json` は読み込んだ候補のスナップショット。各queryが実際に使う例は `example_ids` で特定する。
入力・raw・03_1/05/06の結果・07のHTMLを変更しない。

固定した使用集合は各dry-runの `pattern_examples.json` に保存する。最新の整理済み集合は `data_prep/fixed_pattern_examples.json`、内訳は本README末尾と `data_prep/fixed_pattern_counts.csv` に記録する。

## 6. 検証

repository rootから：

```bash
.venv/bin/python -B -m pytest experiments/08_fewshot/tests/test_fewshot_runner.py experiments/08_fewshot/data_prep/test_data_prep.py -q -p no:cacheprovider
```

144件の条件数、next-action限定、Task・Score・1:1、4/8/12例の入れ子、H間での問題一致、別episode選択、元user全文の保持、例のsystem除外、入力改変の拒否、候補不足、入力未配置、API client未生成を確認する。検証用の選択ラベルとpreviewは一時ディレクトリにのみ作り、ユーザーの選択済みデータや実験結果として保存しない。

<!-- FIXED_POOL_REPORT_START -->
## 固定した例の内訳（2k=12）

各Task×パターンで正解6件・不正解6件。元の74件は削除せず、パターン別の使用候補を絞った。
4群で延べ48件、パターン間の重複を除くと38件。元候補のうち36件は今回使わない。
以下は例の件数であり、144件の評価リクエスト数ではない。モデルは例を回答した元モデル。
パターン1の正解はsol/terra/luna各2件、GPT-3.5は候補0件。それ以外はsol 1・terra 1・luna 2・GPT-3.5 2件。
モデル配分とScore条件を満たす範囲でlast-actionを最小化した。パターン1の不正解にS4、パターン2の正解にS2を少なくとも1件含める。
結果としてパターン2の正解は全件S2。Score 1も許容する条件は維持し、next-action優先によってこの構成になった。

| Task | パターン | 正解 | 不正解 | next-action | last-action | 合計 |
|---|---|---:|---:|---:|---:|---:|
| MountainCar-v0 | pattern1 | 6 | 6 | 6 | 6 | 12 |
| MountainCar-v0 | pattern2 | 6 | 6 | 12 | 0 | 12 |
| Pendulum-v1 | pattern1 | 6 | 6 | 6 | 6 | 12 |
| Pendulum-v1 | pattern2 | 6 | 6 | 10 | 2 | 12 |

### MountainCar-v0 / pattern1

| 区分 | 元モデル | S1 | S2 | S3 | S4 | next-action | last-action | 合計 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 正解 | sol | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | terra | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | luna | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 不正解 | sol | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | terra | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | luna | 0 | 0 | 1 | 1 | 2 | 0 | 2 |
| 不正解 | 3.5 | 0 | 0 | 2 | 0 | 2 | 0 | 2 |

### MountainCar-v0 / pattern2

| 区分 | 元モデル | S1 | S2 | S3 | S4 | next-action | last-action | 合計 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 正解 | sol | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| 正解 | terra | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| 正解 | luna | 0 | 2 | 0 | 0 | 2 | 0 | 2 |
| 正解 | 3.5 | 0 | 2 | 0 | 0 | 2 | 0 | 2 |
| 不正解 | sol | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | terra | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | luna | 0 | 0 | 2 | 0 | 2 | 0 | 2 |
| 不正解 | 3.5 | 0 | 0 | 2 | 0 | 2 | 0 | 2 |

### Pendulum-v1 / pattern1

| 区分 | 元モデル | S1 | S2 | S3 | S4 | next-action | last-action | 合計 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 正解 | sol | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | terra | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | luna | 2 | 0 | 0 | 0 | 0 | 2 | 2 |
| 正解 | 3.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 不正解 | sol | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | terra | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | luna | 0 | 0 | 2 | 0 | 2 | 0 | 2 |
| 不正解 | 3.5 | 0 | 0 | 1 | 1 | 2 | 0 | 2 |

### Pendulum-v1 / pattern2

| 区分 | 元モデル | S1 | S2 | S3 | S4 | next-action | last-action | 合計 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 正解 | sol | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| 正解 | terra | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| 正解 | luna | 0 | 2 | 0 | 0 | 2 | 0 | 2 |
| 正解 | 3.5 | 0 | 2 | 0 | 0 | 0 | 2 | 2 |
| 不正解 | sol | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | terra | 0 | 0 | 1 | 0 | 1 | 0 | 1 |
| 不正解 | luna | 0 | 0 | 2 | 0 | 2 | 0 | 2 |
| 不正解 | 3.5 | 0 | 0 | 2 | 0 | 2 | 0 | 2 |

パターン1の正解S1はnext-action候補がないため、両Taskとも6件全てlast-action。Pendulumのパターン2正解はGPT-3.5の2件だけlast-actionで、それ以外はnext-action。

4例・8例はこの固定集合から正解／不正解を各2件・各4件取る。元モデルをsol→terra→luna→3.5の順で巡回するので、先頭4件（片側）では可能なモデルを1件ずつ含める。

固定した全例と順序: [fixed_pattern_examples.json](data_prep/fixed_pattern_examples.json)。Score×next/lastまでの完全なクロス集計（0件セルを含む）: [fixed_pattern_counts.csv](data_prep/fixed_pattern_counts.csv)。

[更新済みプレビュー](results/dry_run/preview.html)。元の74件JSONは保持し、実行時は同じ規則で固定集合を再現する。
<!-- FIXED_POOL_REPORT_END -->

## Task別の正解率表

[HTMLで表を開く](results/analysis/accuracy_tables.html) / [Markdown](results/analysis/accuracy_tables.md) / [集計JSON](results/analysis/accuracy_tables.json)。

MountainCar-v0とPendulum-v1を別表とし、行をH・Scoreパターン・総例数2k、列をTerra/Lunaとする。各条件n=3の正誤（1/0）を平均して正解率を計算し、正解数/回答数と解析不可件数を併記する。解析不可・空応答は正解には含めず、分母には含める。未回収分は分母から除外し、未完了と表示する。

08のディレクトリから `python analysis/summarize_results.py` で更新できる。APIには送信せず、`results/api/`を読み取り、`results/analysis/`へ出力する。

### 正解率グラフ

- [MountainCar-v0（PNG）](results/analysis/figures/MountainCar-v0_accuracy.png)
- [Pendulum-v1（PNG）](results/analysis/figures/Pendulum-v1_accuracy.png)

各Taskに1枚、行がScoreパターン1/2、列がH=5/20の4パネル。各パネルにTerra/Lunaを並べ、各モデルの棒を総例数2k=4/8/12で色分けする。高さはn=3の正解率、ラベルは正解率と正解数/回答数。0%もラベルを明示する。図の採点・分母は上の表と同じ。

08のディレクトリで `python analysis/plot_results.py` を実行すると、表とグラフを保存済みAPI結果から再生成する（API送信なし）。

### 入出力ビューア

[view_rawdata.htmlを開く](view_rawdata.html)。07と同様に入力1（system）、入力2（few-shot入りuser全文）、assistantの回答、正解と比較結果、条件・ID・時間・tokenを閲覧できる。今回の質問のみ・APIレスポンス全文は折りたたみ表示。

モデル、Task、予測対象、H、総例数2k、Scoreパターン、評価問題、採点結果で絞り込める。ID・入力・回答の検索、前後移動、1回答のJSON保存に対応。Reasoning scoreは今回の出力については未評価であり、使用例のScore条件と区別して表示する。ID差は物理量の誤差ではない。

08のディレクトリで `python analysis/build_results_viewer.py` を実行すると保存済みAPI結果から更新する。HTMLはデータを内蔵しており、サーバー・通信なしで開ける。実行中に生成した場合は、その時点のスナップショットとなる。API送信や元結果の変更は行わない。

### 08のReasoning採点用データ

[採点用CSV](reasoning_for_scoring.csv)（144件、score空欄）と[採点依頼文](reasoning_score_analysis/Prompt.md)を用意した。今回の採点は1＝一般的推測、2＝policy傾向、3＝具体的根拠、4＝定量的推論で、07の番号の逆向き。実行済みfew-shot条件の旧Scoreは変更していない。[詳細](reasoning_score_analysis/README.md)。

## 条件をpoolした比較図

[図と指標の定義](results/analysis/pooled_comparisons/README.md)。Taskごとに、Scoreパターン・few-shot総数・Hのうち1つを比較軸にし、残る2条件をpoolする。Terra/Lunaを同じ図に表示する。Scoreパターン・Hは並列棒グラフ、few-shot総数は2系列の折れ線。

- [MountainCar一覧図](results/analysis/pooled_comparisons/MountainCar-v0_overview.png)
- [Pendulum一覧図](results/analysis/pooled_comparisons/Pendulum-v1_overview.png)

08から `python analysis/plot_pooled_comparisons.py` で再生成できる。採点済みCSVをquery_idで結合し、144回答を集計する。Scoreパターン・Hは各モデル・群18回答、few-shot総数は12回答。ただし物理的な評価問題は同じ3問を共有する。

**採点修正済み（2026-09-30）:** 旧parserは`predictions`のトルク値をbin IDとみなし、Pendulumを0/72正解と記録していた。現在は`runner/scoring.py`で最終action/bin IDを採点する方式に統一し、保存済み144回答の再採点と、正解率表・通常図・pool図・ビューアの再生成を完了した。Pendulumは59/72、MountainCarは65/72正解。旧records・summaryは`results/scoring_backups/`に保存し、各recordの`legacy_evaluation`にも旧採点を保持する。API応答全文・リクエスト・manifest・プロンプト・Reasoning採点CSVは変更していない。

修正範囲は08内の採点と派生結果。共有の上流`llm_x.evaluation`は他実験でも利用するため変更していない。03_1・05・06の過去の保存済みstatusには同種の問題があり得るが、別途監査が必要。共通Joint分析側はもともと最終IDを別に解析している。

再採点・再生成は08から次の順で実行する。既に現行バージョンで採点済みなら再採点は何もしない。API送信は行わない。

```bash
python analysis/rescore_results.py
python analysis/plot_results.py
python analysis/build_results_viewer.py
python analysis/plot_pooled_comparisons.py
```

## 07の既存結果との比較

[図一覧・集計方法](results/analysis/previous_comparison/README.md)。07はnext-action・H=5/20・4モデルの各条件先頭10件、08は既存各条件3件を使用する。Hは等重みでpoolし、07の旧推論scoreだけを`5 − score`に変換する。追加API取得は行わない。

- [MountainCar一覧図](results/analysis/previous_comparison/MountainCar-v0_overview.png)
- [Pendulum一覧図](results/analysis/previous_comparison/Pendulum-v1_overview.png)

各一覧は4行（全条件／例数pool／パターンpool／Terra・Luna全pool）×3列（正解率／MAE／推論レベル）。一覧2枚と拡大24枚のPNG、集計CSV、使用回答一覧を保存する。07のMountainCar/Lunaには解析不可6件があるため、MAEの有効件数を明示する。episode・時点が違う既存結果の比較であり、同一問題の対応付き比較ではない。

08から `python analysis/plot_previous_comparison.py` で再生成できる。追加取得用に誤って作ったn=10のコード・テスト・送信計画は削除した。
