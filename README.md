# Mental Modeling / LLM-Xavier 再評価環境

作業先: `/Users/cls-lab/Git/Matsuo/mental-modeling-openai`。raw readerと前処理を分離済み。**local token countは実行済み、OpenAI API request=0**。LLM精度評価・RL学習・simulation・paid pilotは実行していません。

## すぐ使う

```bash
source .venv/bin/activate
python tools/list_datasets.py
python tools/describe_dataset.py --dataset MountainCar-v0
python tools/show_data.py --dataset MountainCar-v0 --episode 0 --index 25
python tools/show_data.py --dataset f16capstone --sequence 0 --index 10
python tools/show_data.py --dataset f16capstone --sequence 0 --index 10 --processed --preprocessing-config configs/preprocessing/f16capstone_default.yaml
python tools/show_data.py --dataset trajair --sequence 100 --index 20
python tools/show_data.py --dataset aircombat_wez --row 100
python tools/count_input_tokens.py --dataset MountainCar-v0 --history-size 1 --model gpt-4o
python tools/count_input_tokens.py --all --history-sizes 0 1 2 3 5 10 --model gpt-4o
python tools/count_input_tokens.py --external-all --prompt-mode external_generic --history-size 1 --model gpt-4o
```

結果は [dataset目次](outputs/dataset_inventory/datasets.csv)、[token比較表](outputs/token_counts/token_inventory.csv)、[query別CSV](outputs/token_counts/queries.csv)、[実施報告](outputs/token_counts/REPORT.md)。各count実行は同じoutput-dirの結果を置き換えるため、比較実験は `--output-dir outputs/token_counts/experiment_name` で分けてください。

## データを目視確認する

調査Notebookを公式用と外部用に分離しています。どちらも単独で上から実行でき、他方のrawデータやinventoryは読みません。

- [01: 公式データ](notebooks/01_data_and_token_inspection.ipynb)：`create_inventory(data/llmx_data)` で公式rawを検査し、全task・全episode・全NPZ keyを表示します。
- [02: 外部データ](notebooks/02_external_data_and_token_inspection.ipynb)：`scan(data/candidate_datasets)` で外部rawを検査し、file/schema/column/arrayを表示します。大きな表は全行をDataFrameに保持してページ表示します。scanには数分かかる場合があります。

どちらもread-onlyのlive scanを主とし、保存済みinventoryは後から検算にだけ使用します。

```bash
# workspace直下で、初回のみ依存を追加
.venv/bin/python -m pip install -e '.[data,notebook,test]'
.venv/bin/python -m ipykernel install --prefix .venv --name mental-modeling --display-name 'Python (mental-modeling)'
.venv/bin/jupyter lab notebooks/01_data_and_token_inspection.ipynb
```

各Notebookの「drill-down selection」でdataset/episode/sequence/row/indexを選び、raw値→既存preprocessing→history→prompt全文→選択queryのtoken数を追跡します。公式20 taskはraw閲覧、元promptは対応11 taskのみ。外部用の初期選択はF16Capstoneで、`PREPROCESSING_CONFIG='configs/preprocessing/f16capstone_default.yaml'` を明示しています。外部preprocessingは本projectで定義するLLM入力用example / baseline変換であり、raw datasetそのものではありません。

「LIVE TOKEN COUNT」は選択recordとは独立に、公式用は `build_token_inventory(datasets='all', ...)`、外部用は `datasets='external-all'` を呼びます。公式は全valid query、外部は既存YAML設定に従い100,000件超のpoolを既定2,000件・seed=0でsample（全件扱いしない）。Hは既定 `[1]`、staticは0です。各Notebookの対象範囲のqueryを `all_token_queries_df`、集計を `token_by_dataset_df`、scope/skip理由もDataFrameで保持し、mode/前処理別の分布と上位・下位queryを表示します。最後の保存済みbatch比較も各Notebookの対象datasetに限定します。

Notebookはdata/・保存済みCSV/JSONを変更せず、OpenAI APIも呼びません。CLIとNotebookは同じ `tools/token_inventory.py:build_token_inventory` を共有し、保存処理はCLI側の明示的writerへ分離しています。検証は `.venv/bin/python tools/check_inspection_notebook.py`（全live scanと全token計測を実行）です。

## データ一覧

| Dataset | データ単位 | 件数 | State | Action | Reward | 形式 | 使い方 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Acrobot-v1 | step / episode | 399 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset Acrobot-v1` |
| BipedalWalker-v3 | step / episode | 144 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset BipedalWalker-v3` |
| CartPole-v1 | step / episode | 2500 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset CartPole-v1` |
| FetchPickAndPlace-v2 | step / episode | 500 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset FetchPickAndPlace-v2` |
| FetchPush-v2 | step / episode | 500 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset FetchPush-v2` |
| FetchSlide-v2 | step / episode | 500 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset FetchSlide-v2` |
| HalfCheetah-v4 | step / episode | 3000 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset HalfCheetah-v4` |
| InvertedDoublePendulum-v4 | step / episode | 538 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset InvertedDoublePendulum-v4` |
| InvertedPendulum-v4 | step / episode | 1419 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset InvertedPendulum-v4` |
| LunarLander-v2 | step / episode | 744 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset LunarLander-v2` |
| MiniGrid-DoorKey-5x5-v0 | step / episode | 226 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-DoorKey-5x5-v0` |
| MiniGrid-Empty-Random-5x5-v0 | step / episode | 30 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-Empty-Random-5x5-v0` |
| MiniGrid-Fetch-5x5-N2-v0 | step / episode | 210 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-Fetch-5x5-N2-v0` |
| MiniGrid-GoToDoor-5x5-v0 | step / episode | 307 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-GoToDoor-5x5-v0` |
| MiniGrid-KeyCorridorS3R1-v0 | step / episode | 162 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-KeyCorridorS3R1-v0` |
| MiniGrid-Unlock-v0 | step / episode | 130 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MiniGrid-Unlock-v0` |
| MountainCar-v0 | step / episode | 1041 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset MountainCar-v0` |
| Pendulum-v1 | step / episode | 2000 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset Pendulum-v1` |
| Pusher-v4 | step / episode | 600 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset Pusher-v4` |
| Reacher-v4 | step / episode | 250 | あり | あり | あり | NPZ | [schema](data/README.md) / `--dataset Reacher-v4` |
| f16capstone | reader内record¹ | 836572 | あり | あり | なし | CSV/MAT/ULog | [schema](data/candidate_datasets/f16capstone/SCHEMA.md) / `--dataset f16capstone` |
| trajair | reader内record¹ | 12130491 | あり | なし | なし | CSV/TXT | [schema](data/candidate_datasets/trajair/SCHEMA.md) / `--dataset trajair` |
| aircombat_wez | static / reader内record¹ | 4050 | あり | なし | なし | CSV | [schema](data/candidate_datasets/aircombat_wez/SCHEMA.md) / `--dataset aircombat_wez` |
| calculated_moves | static / reader内record¹ | 5298468 | なし | なし | なし | CSV/TXT | [schema](data/candidate_datasets/calculated_moves/SCHEMA.md) / `--dataset calculated_moves` |
| baidu_fighter_jet | static / reader内record¹ | 未取得 | 未確認 | 未確認 | 未確認 | — | [schema](data/candidate_datasets/baidu_fighter_jet/NOT_ACQUIRED.md) / `--dataset baidu_fighter_jet` |

¹ 件数はadapterがアクセスできるrecordの合計で、異種のMAT変数・ULog topic・元CSV・提供元processedファイル・script/configを含みます。同一飛行の別exportもあり、独立した物理trajectoryの数ではありません。F16は338 stream、TrajAirは6,544 file×aircraft track＋112 raw/weather表。詳細なunitと元ファイルは `describe_dataset.py --dataset ID --include-units` で確認できます。外部state/action/rewardの固定shapeは未定義（CSV目次ではnull）。全columnの型・意味・shapeはdescribe出力のraw_schema_groupsと個別schema参照。

## Raw・reader・前処理・結果の境界

```text
data/                         取得元ファイルをそのまま保持、read-only
dataset_adapters/             全raw column/arrayへのreader、feature選択なし
preprocessing/llmx_original.py 元のLLM-X関数を呼ぶwrapper
preprocessing/prompting.py     外部baselineの選択・history・文字列化
configs/preprocessing/        feature-selection / formatting
tools/                        raw表示・processed表示・local token count
outputs/                      全derived results、必要なcacheはoutputs/cache/
```

data/に以前からある調査README/SCHEMAは今回変更せず保存しています。今後のschema文書生成先は `outputs/docs/` に変更しました。derived writerはdata/・upstream/への出力（symlink経由も）を拒否します。raw変更検証は `python tools/raw_integrity.py --verify`。今回のbaselineは既存文書も含む34,227ファイルのSHA256です。

**このpreprocessingはraw datasetそのものではなく、LLM入力用に本projectで定義した変換である。** 外部はexample / baselineであり、original Mental Modeling benchmarkや精度評価ではありません。

- F16: 3 CSV（43,044 records）の7 state列＋rc_channel_0〜3＋時刻。controlの物理的な舵面対応は決めつけません。MAT/ULogへのアクセスも可能ですが、既定countでは重複exportや時刻同期の研究判断を避けるため除外。
- TrajAir: 提供元が配布するprocessed TXTの5位置/風列＋frame/aircraft ID。これは取得物そのもので、本projectがrawディレクトリへ加工保存したものではありません。全raw CSV/weatherもreaderで参照可能。frame gapは維持し、補間・0埋め・速度/action/rewardの生成はしません。
- WEZ: Data/の4表3,864 recordsをstatic表示。うち3表は関連する別target exportで、独立3,000場面とは扱いません。
- Calculated Moves: agent/validation CSVの2,804,268 recordsをstatic表示。rule weight / encounter fitnessはflight action / timestep rewardへ置換しません。script・configuration・winnerもraw readerでアクセス可能。
- Baidu: 未取得。schema・capabilityは不明のまま。

CSV raw readerは元の文字列表現を保持するため、raw dtypeはUnicode文字列です。数値解釈したdtypeはdescribeのschemaに別記します。外部baselineもlexical表現のJSONで文字列化し、任意の丸めやnormalizationを暗黙には行いません。`state_columns/action_columns/reward_column/fields` は設定で変更可能。存在しない列はエラー。欠損値は補完しません。

番号は0始まり。公式はdataset_id＋episode_id＋index、軌跡はdataset_id＋sequence_id＋index、staticはdataset_id＋row_id。安定したsource_file/sequence_keyも記録します。static row_idは全reader unitの累積offsetなので、token count対象だけの連番とは限りません。

## Token countの範囲と再現性

`raw → adapter → preprocessing → prompt construction → tokenizer` の実文字列を計測します。公式wrapperは上流Episode・history_text・render_question・system_prompt・user_promptを変更せず使用。外部ではモデルへ渡す列をadapterでなくYAMLで決定します。

公式は11 taskの全valid queryをH=0,1,2,3,5,10で計測。外部genericはH=1、staticはH=0のみ（requested Hを静的データへ無理に適用しない）。外部genericのhistoryはquery indexを含むH+1観測で、公式next-actionのH+1過去tuple＋現在stateとは意味が異なります。

外部のeligible poolが100,000 query超なら既定2,000件をseed=0でdeterministic sample。今回F16/WEZは選択scope全件、TrajAir/Calculated Movesはsampleです。`--full-count` で明示的に全件化できますが、数百万prompt・大きなCSVが必要です。`source_globs` と全reader件数・eligible pool・sample sizeを結果に記録し、sample総tokenを全dataset総tokenへ外挿しません。

`queries.csv` に元ファイルSHA256、全ID、history範囲、prompt mode、metric/question、実効前処理JSON＋SHA256、prompt文字列SHA256、system/history/question/user/content/API概算tokenを保存。設定JSONの同一copyは `preprocessing_configs/<sha256>.json`。同じqueryはdataset/source/sequence/query_indexでjoinしH間比較できますが、Hで有効query集合が変わるため総和の比と同一queryの増加率は別です。

`summary_by_sequence.csv`、`summary_by_task.csv`、`summary_by_dataset.csv`、`token_inventory.csv` はmode・H・設定SHAごとに分離します。history/questionの単独tokenはBPE境界のためuser_tokensに単純加算できません。contentは実測、API入力はcontent+9の近似でserver usageではありません。今回のGPT-4o tokenizerはo200k_base。異なるHやsample scopeの総和を全dataset費用へ変換しないようcost toolもガードします。

今回と同じ一括実行:

```bash
python tools/count_input_tokens.py --everything --history-sizes 0 1 2 3 5 10 --external-history-sizes 1 --model gpt-4o
```

## ディレクトリ

```text
mental-modeling-openai/
├── README.md
├── pyproject.toml
├── upstream/LLM-Xavier/          # 原典checkout、変更なし
├── data/
│   ├── README.md                 # 全20 task・全key・実step例・dimension意味
│   ├── llmx_data/                # 固定revisionの公式snapshot
│   └── candidate_datasets/
│       ├── README.md             # 取得状況・互換性比較
│       └── 各dataset/SCHEMA.md   # 全ファイルschema群・解釈
├── tools/
│   ├── inventory_llmx_data.py
│   ├── inventory_candidates.py
│   ├── audit_dataset_relationships.py
│   ├── render_schema_docs.py
│   ├── llmx_adapter.py
│   ├── count_input_tokens.py
│   ├── pilot_output_tokens.py
│   └── summarize_token_costs.py
├── configs/                      # 設定例・取得元・revision
├── tests/
└── outputs/
    ├── dataset_inventory/
    │   ├── llmx_files.csv
    │   ├── llmx_tasks.csv
    │   ├── llmx_schema.json
    │   ├── candidate_files.csv
    │   ├── candidate_columns.csv
    │   └── candidate_schema.json
    ├── token_counts/
    └── pilot/                    # 過去dry-run計画。paid結果ではない
```

downloadデータ・仮想環境・生成CSV/JSONはgitignore対象。`upstream/LLM-Xavier`は公式コードをコピーした通常ディレクトリとして、このリポジトリで直接管理する。コードの編集・commit・pushは本体と同じ手順で行える。旧 `outputs/dataset_inventory.{json,csv}` も互換出力する。過去smoke出力はsynthetic fixtureで実データ評価ではない。

通常の`git clone`で原典コードも取得できる。submoduleの初期化は不要。取得元・元commitは`configs/sources.json`の`llm_xavier`に記録し、原典の`LICENSE`・`NOTICE`・`CITATION.cff`を保持する。公式コードの更新を取り込む際は、こちら側の変更を確認してから差分を統合する。

token見積もりのスクリプトは`notebooks/01_token_estimation.py`、Excelは`notebooks/01data/token_estimation.xlsx`。再生成には`.[estimation]`の追加依存関係を使用する。

環境再作成（source取得済みが前提）:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e 'upstream/LLM-Xavier[openai,test]' -e '.[data,test]'
```

移動に伴う仮想環境の旧workspace参照は更新済み。明示的に `.venv/bin/python` を使うと解釈器が明確になる。

## 原典と公式データ

[LLM-Xavier](https://github.com/LukasWill/LLM-Xavier) commit `441c5644f8eb93d6aee9c53631feb5fa5bf75b3c` は変更せず利用。[llmx_data](https://huggingface.co/datasets/lerrhoo/llmx_data) revisionは `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`。取得情報は [configs/sources.json](configs/sources.json)、全20 task表・license・各keyの全dtype/shape/ndim/要素数は [data/README.md](data/README.md)。

全183ファイル（56,861,285 bytes、cache除外）のうち172 NPZがepisode。11件のcheckpoint/TensorBoard等はhistoryではなく、metadataのみ検査する。NPZは `allow_pickle=False` で読む。

vector taskのrawは通常 `states[T,1,D]`、continuous `actions[T,1,A]` またはdiscrete `actions[T,1]`、`rewards[T,1]`。stateはfloat32/float64、continuous actionはfloat32、discreteはint64。MiniGridは `states[T,7,7,3] uint8` と方向field、Fetchは26次元stateと3次元goal field。`episodic_return` はepisode単位で、step rewardとはshapeも役割も異なる。

T個のtupleが並ぶが、rawにnext_states、timestamp、terminated/truncatedはない。隣接stateは同一episode内でのみ次状態として扱い、最終行の次状態を補完しない。T=1の3件はinventoryに含めるが上流Episodeの最低2step条件に適合しない。

## Rawからprompt、送信内容まで

```text
NPZのndarray
  → Episode.load: 数値型/長さ検証、SHA256
  → _query_indices / _history_range: 質問位置とhistory範囲
  → data.py: history文字列
  → questions.py: 質問にstate/action等を挿入
  → prompting.py: system=task説明、user=history＋質問
  → 将来paid flagで許可された場合のみ文字列を送信
```

preprocessing/llmx_original.py は上流の `Episode`, `system_prompt`, `user_prompt`, `render_question` とquery planning helperを直接利用する。Mental Modeling promptを独自再実装しない。質問原文は `llm_x/feedback.py`、task/state説明は `llm_x/task.py` にある。

### 数値表示とrawの違い

`data.py` と `questions.py` は `np.array2string(value, precision=4, separator=", ")`。小数点以下4桁固定ではなくNumPyの表示設定。dtype metadataやNPZ bytes自体は入力に入らず、ブラケット階層は維持する。linewidth/thresholdは指定されず、NumPy設定による改行・大きい配列の `...` があり得る。

公式MountainCar `episode_0.npz` の実先頭2stepを同じformatterで表示:

```text
Step 0:
  state: [[-0.4976,  0.    ]]
  action: [2]
  reward: [-1.]

Step 1:
  state: [[-0.4968,  0.0008]]
  action: [2]
  reward: [-1.]
```

元float32値はNPZに保持する。indexed historyはstep間を空行で区切る。unindexedはslice全体を `states:/actions:/rewards:` として表示するがwrapperはindexed=True。Fetchは最後のfeatureを落とす（26→25）。26番目の意味は一意に確認できず、時刻などと推測しない。optional fieldsとepisodic_returnは保持・検証するがhistoryへ追加しない。

`next-state` 質問内rewardはPython float/listへ変換され、historyのarray表記と異なる。metricによって現在/次stateやactionを質問へ挿入する。採点用のflattenとprompt表現も区別する。

### History規約

H=history_size、L=episode length、i=query index。上流規約では **H+1 tuple** を使う。

| metric | query index | history半開区間 |
| --- | --- | --- |
| next-action / argue-action | H+1 ≤ i < L | [i-H-1, i) |
| last-action / last-state | H+1 ≤ i < L-1 | [i-H-1, i) |
| next-state | H ≤ i < L-1 | [i-H, i+1) |

H=1のnext-actionは最初i=2、historyにstep 0/1、質問にstate 2が入る。

### 最終送信対象

systemにはtask名、task description、observation/action/reward space、transition dynamics、initial state、termination。userには固定導入文＋history＋質問。ローカルpilotは将来許可時のみ次のpayloadを構成する（**説明のみ、未実行**）:

```python
{
    "model": selected_model,
    "input": [
        {"role": "system", "content": system_text},
        {"role": "user", "content": history_and_question_text},
    ],
    "max_output_tokens": configured_limit,
}
```

NPZやCSVを添付する実装ではない。pilotはResponses API、上流backendはChat Completionsで経路が異なり、同じ文字列でもserver token usageの一致を仮定しない。message inputとusageは [OpenAI公式Responses reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) を確認した。

## 対応task・既定質問

`--all` は全NPZを発見するが、上流registryのデータ付きtaskは11個のみ。9個（Unlock以外のMiniGrid 5個、HalfCheetah、Pusher、Reacher、BipedalWalker）は理由付きskip。CliffWalkingはregistryにあるがepisodeなし。20 taskのschema調査完了と20 taskでprompt実験可能は同義ではない。

| tasks | 既定metric / question |
| --- | --- |
| MountainCar / LunarLander / CartPole / Acrobot / MiniGrid-Unlock | next-action / next_action_prediction |
| Pendulum | next-action / next_action_prediction_continuous_bins、[-2,2] |
| InvertedPendulum / InvertedDoublePendulum | next-action / next_action_prediction_continuous_bins_mjpen、[-3,3] |
| Fetch 3 task | next-action / next_action_prediction_continuous_bins_fetch、[-1,1] |

上流state-direction variantsも登録taskで選択可能。discrete/argue-actionとcontinuous配列の不整合等は拒否し、明示metric/question pairをregistryで照合する。Fetch finger左右等のtask文章の誤記やwrapper不明点はdata READMEへ記録し、今回は評価仕様を変更していない。

## Inventory再実行

```bash
.venv/bin/python tools/inventory_llmx_data.py
.venv/bin/python tools/inventory_candidates.py
.venv/bin/python tools/audit_dataset_relationships.py
.venv/bin/python tools/render_schema_docs.py
```

CSVはchunk、processedはscene、NPZはepisode単位。全nested ZIP entryと全展開dataを検査。MATはtop-level variableを選び巨大MCOS metadataの一括実体化を避ける。CSVは行数をschema fingerprintへ混ぜず列順/型/構造を比較する。source size/mtimeのscan前後不変性も確認。未取得Baiduのschemaは推測しない。

## Input-token estimator（local実行済み）

`count_input_tokens.py` はAPI clientを持たずローカルtiktokenで文字列を数える。初回に公開tokenizer assetを取得する可能性はあるが、OpenAI inference APIを呼ばない。query別system/user/content-only/estimated inputと、dataset/task別mean、母標準偏差(ddof=0)、min/max/median/nearest-rank p95/totalを記録する。

model名がtiktoken未登録なら既定cl100k_baseへfallbackし、警告とmetadataを残す。chat framingの既定3 tokens/message＋3 primingは近似で、role文字列等を含むserver厳密値ではない。将来pilotのactual inputと比較する。

出力は `outputs/token_counts/queries.csv`, `summary_by_dataset.csv`, `summary_by_task.csv`, `summary.json`。再実行する例:

```bash
.venv/bin/python tools/count_input_tokens.py --all --history-size 1 --model gpt-4o
```

単一taskなら `--all` の代わりに `--task MountainCar-v0 --metric next-action --question-name next_action_prediction`。

## 5-query pilot・費用集計（将来用）

sample_size=5、seed=0でtaskの全episode中の有効queryから再現可能に抽出する。既定dry-runは `dry_run_plan.json` のみ作成し、`DRY RUN / Would send N queries. / No API request was made.` と表示。今回は追加pilotも実行していない。

```bash
.venv/bin/python tools/pilot_output_tokens.py --all --history-size 1 --model gpt-4o --sample-size 5
```

将来別途paid実行を決めたときだけ `--execute-paid-api` を付ける。その分岐のみが環境変数OPENAI_API_KEYを読みclientを作る。keyをCLI引数・文書・ログへ保存しない。今回key/.envは読んでいない。

記録はepisode/query/history範囲/model/timestamp/parameters/SHA256/予測文字列/actual input-output-total usage/visible output tokens/reasoning tokens（提供時）。raw API response objectは保存しない。visible outputと課金対象outputは区別する。統計stdはddof=0。

token countとpilotが揃った後、単価を実行時点で確認して明示指定:

```bash
.venv/bin/python tools/summarize_token_costs.py --pilot-summary outputs/pilot/gpt-4o/token_summary.json --input-price-per-million INPUT_PRICE --output-price-per-million OUTPUT_PRICE
```

## 外部候補・検証

[候補比較](data/candidate_datasets/README.md): Calculated Moves=C、AirCombat-WEZ=C、F16Capstone=B、TrajAir=B。BaiduはNOT_ACQUIRED、判定保留。判断根拠と未確認事項は各SCHEMA.mdに分離。

外部34,036ファイル、公式と合わせ34,219ファイルを走査。support/archive/textも含む外部分類82群、構造化data/empty schemaのみ54群。公式はtask別20群（T可変）、task間の同型を重複除外すると13群。

```bash
.venv/bin/python -m pytest tests upstream/LLM-Xavier/tests -q
```

inventory testsはNPZ/CSV/MAT extraction、dtype/shape、grouping、chunk、欠損、empty、track分割、ZIP、script、source不変性を確認。既存prompt/token工具はsyntheticデータ・fake clientでregression testし、新規testは実raw reader、前処理設定、元prompt一致、ネットワーク禁止下のlocal countも検証します。LLM評価は実行しません。学習とsimulationも実行対象外。

2026-09-09: 48 tests passed。全件token計測もsocket接続を禁止して再実行し、API送信は0件。OpenAI Docsスキルはtokenizer/API usageの区別を確認するために使用しました。
