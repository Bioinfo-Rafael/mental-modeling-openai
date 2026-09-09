#!/usr/bin/env python3
"""Render Japanese schema documentation from exhaustive inventories (offline only)."""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inventory_llmx_data import _atomic_text


def table(headers, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]) + "\n\n"


def block(text, lang="text"):
    return f"```{lang}\n{text}\n```\n\n"


def jblock(obj):
    return block(json.dumps(obj, ensure_ascii=False, indent=2), "json")


STATE_MEANINGS = {
"MountainCar-v0": "`s[0]`=x位置、`s[1]`=速度。action 0/1/2 は左加速/無加速/右加速。",
"CartPole-v1": "`s[0]`=台車位置、`s[1]`=台車速度、`s[2]`=棒の角度(rad)、`s[3]`=棒の角速度。action 0/1 は左/右。",
"LunarLander-v2": "`s[0]`=横位置、`s[1]`=縦位置、`s[2]`=横速度、`s[3]`=縦速度、`s[4]`=角度、`s[5]`=角速度、`s[6]`=第1脚接地、`s[7]`=第2脚接地。action 0=無操作、1=左姿勢エンジン、2=主エンジン、3=右姿勢エンジン。",
"Pendulum-v1": "`s[0]`=cos(theta)、`s[1]`=sin(theta)、`s[2]`=角速度。actionは1成分の連続トルク。",
"Acrobot-v1": "`s[0]`=cos(theta1)、`s[1]`=sin(theta1)、`s[2]`=cos(theta2)、`s[3]`=sin(theta2)、`s[4]`=theta1角速度、`s[5]`=theta2角速度。theta2は第1リンク相対。action 0/1/2 はトルク -1/0/+1。",
"InvertedPendulum-v4": "`s[0]`=台車位置、`s[1]`=棒の鉛直からの角度(rad)、`s[2]`=台車速度、`s[3]`=棒の角速度。actionは連続1成分。",
"InvertedDoublePendulum-v4": "`s[0]`=台車位置、`s[1]`=sin(theta1)、`s[2]`=sin(theta2)、`s[3]`=cos(theta1)、`s[4]`=cos(theta2)、`s[5]`=台車速度、`s[6]`=theta1角速度、`s[7]`=theta2角速度、`s[8:11]`=3自由度のconstraint force。theta1は台車と第1棒、theta2は棒間の角度。actionは連続1成分。",
"HalfCheetah-v4": "`s[0]`=rootz、`s[1]`=rooty角、`s[2:8]`=bthigh,bshin,bfoot,fthigh,fshin,ffootの角度（この順）。`s[8]`=rootx速度、`s[9]`=rootz速度、`s[10]`=rooty角速度、`s[11:17]`=上記6関節角速度。actionも上記6関節順。標準v4の `_get_obs()` と実測17次元は整合するが、収集時wrapperの完全な構成は未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/half_cheetah_v4.py)。",
"Pusher-v4": "`s[0:7]`=shoulder pan, shoulder lift, upper arm roll, elbow flex, forearm roll, wrist flex, wrist rollの角度（この順）。`s[7:14]`=同順角速度、`s[14:17]`=fingertip xyz、`s[17:20]`=object xyz、`s[20:23]`=goal xyz。actionは7関節入力。標準v4と実測23次元が整合。収集時wrapperは未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/pusher_v4.py)。",
"Reacher-v4": "`s[0:2]`=2関節cos、`s[2:4]`=2関節sin、`s[4:6]`=target xy、`s[6:8]`=2関節角速度、`s[8:11]`=fingertip-target xyz。actionは2関節トルク。標準v4は11次元であり、現行v5の10次元説明で置き換えない。収集時wrapperは未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/reacher_v4.py)。",
"BipedalWalker-v3": "`s[0]`=胴体角、`s[1]`=正規化胴体角速度、`s[2:4]`=正規化水平/鉛直速度。`s[4:9]`=第1脚の股角/正規化股角速度/膝角+1/正規化膝角速度/接地、`s[9:14]`=第2脚の同5値、`s[14:24]`=10本のlidar fraction。位置座標は含まない。actionは股・膝・股・膝の4入力。標準コードと実測24次元が整合。収集時wrapperは未確認。[実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)。",
}

FETCH_MEANING = """`s[0:3]`=gripper xyz、`s[3:6]`=object xyz、`s[6:9]`=object-gripper相対xyz、`s[9:11]`=左右finger変位、`s[11:14]`=object XYZ Euler角、`s[14:17]`=objectのgripper相対速度xyz、`s[17:20]`=object角速度xyz、`s[20:23]`=gripper速度xyz、`s[23:25]`=finger速度2値。これは `task.py` の25次元説明との対応。`s[25]`（rawに追加された26番目）は**意味を一意に確認できず**。時刻等と決めつけない。`task.py` は23と24をともにRightと記しており、左右の厳密な割当はその説明だけでは一意に特定できない。promptでは26番目を除外する。`achieved_goals`=現在object xyz、`desired_goals`=目標object xyz。actionは4成分（xyz変位指令とgripper開閉指令）。"""

MINIGRID_MEANING = """1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。"""


def official_doc(inv):
    s = """# 公式 llmx_data の詳細schema

## 出典・調査範囲

[Hugging Face](https://huggingface.co/datasets/lerrhoo/llmx_data) の固定revision `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`。全172 NPZを `allow_pickle=False` で読み、全183ファイルを列挙した。非cacheサイズは56,861,285 bytes。上流の記載ではCC BY 4.0。policy checkpoint・TensorBoard等の11ファイルはパス/形式/bytesを記録し、危険なpickle等のdeserializeはしない。

再取得手順（今回は再downloadしていない）:

```bash
hf download lerrhoo/llmx_data --repo-type dataset --revision dc2b798f72bc02f7285949ccfcdcb42e5ff326fd --local-dir data/llmx_data
```

## 全20 task

`T` は各episodeの長さ。shapeの先頭以外の `1` もrawに実在し、説明上勝手に削除しない。すべてのkeyのPython型は `numpy.ndarray`、object arrayはなし。action typeは実配列・上流task定義に基づく。rewardは全taskで1成分。

"""
    s += f"NPZ container合計={inv['npz_container_bytes']} bytes、数値ndarray payload合計={inv['ndarray_payload_bytes']} bytes。NPZ内の各NPY entryのcompressed/uncompressed bytesとcompression methodもJSONへ保存（ZIP containerが常に圧縮されているとは限らない）。task間で共有するschemaを重複除外すると{inv['unique_schema_group_count']}群。\n\n"
    rows = []
    for t in inv["tasks"]:
        a = t["files"][0]["arrays"]
        def shape(k): return str(["T"]+a[k]["shape"][1:]).replace("'", "")
        rows.append([t["task"], t["number_of_episodes"], t["total_timesteps"], shape("states"), a["states"]["dtype"], shape("actions"), a["actions"]["dtype"], "/".join(t["actions"]["kinds"]), shape("rewards"), ", ".join(t["other_keys"]) or "なし"])
    s += table(["Task", "Episodes", "Timesteps", "State shape", "State dtype", "Action shape", "Action dtype", "Action type", "Reward shape", "Optional fields"], rows)
    s += """合計 **20 tasks / 172 episodes / 15,200 timesteps**。各task内はTを可変とするschemaで統一されており、dtype・key・非T軸が異なるepisodeはなし。exact shapeはTが違えば異なる。全filename、T、bytes、dtype、shape、ndim、要素数、8 keyの存在判定を `outputs/dataset_inventory/llmx_schema.json` と `llmx_files.csv` に保存。task集計は `llmx_tasks.csv`。旧 `outputs/dataset_inventory.{json,csv}` も互換出力する。

`T=1` の3件はinventoryには含めるが、上流 `Episode.load` の最低2step条件を満たさない:

"""
    for t in inv["tasks"]:
        for f in t["files"]:
            if f["timesteps"] < 2:
                s += f"- `{t['task']}/{f['filename']}`\n"
    s += "\n## 各taskの1 timestepとdimension\n\n以下の `s` は説明用に `states[t]` をflattenしたindex。表示例自体はrawの階層を保持し、先頭episodeの実測t=0を省略せず示す。`episodic_return` はepisode単位であり、stepごとのrewardと区別する。`(state_t, action_t, reward_t)` の列がT個並ぶ。NPZに `next_states`、`terminated`、`truncated`、timestampは存在しない。次状態は同一episode内の隣接stateで扱えるが、最終行の次状態と収集時のreward時刻規約はデータ単体では確認できない。\n\n"
    for t in inv["tasks"]:
        f = t["files"][0]
        s += f"### {t['task']}\n\n"
        s += f"{t['number_of_episodes']} episodes、{t['total_timesteps']} steps。Tのmin/mean/max={t['timesteps_per_episode']['min']}/{t['timesteps_per_episode']['mean']:.6g}/{t['timesteps_per_episode']['max']}。state/action/reward要素数 per step={f['state_dimension']}/{f['action_dimension']}/{f['reward_dimension']}。\n\n"
        s += (FETCH_MEANING if t["task"].startswith("Fetch") else MINIGRID_MEANING if t["task"].startswith("MiniGrid") else STATE_MEANINGS[t["task"]]) + "\n\n"
        s += "keyの有無: " + "、".join(f"`{k}`={'あり' if v else 'なし'}" for k, v in f["key_presence"].items()) + "。\n\n"
        s += f"実例: `{f['relative_path']}`（{f['raw_size_bytes']} bytes、T={f['timesteps']}）。\n\n"
        s += table(["key", "type", "dtype", "episode shape", "ndim", "要素数"], [[k, v["python_type"], v["dtype"], v["shape"], v["ndim"], v["num_elements"]] for k, v in f["arrays"].items()])
        values = []
        with np.load(ROOT / "data/llmx_data" / f["relative_path"], allow_pickle=False) as z:
            for k in z.files:
                value = z[k] if k == "episodic_return" else z[k][0]
                values.append(f"{k}{'' if k == 'episodic_return' else '[0]'} = " + np.array2string(value, precision=17, threshold=1000, separator=", "))
        s += block("\n".join(values))
        s += "Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。\n\n"
    s += """## Rawとpromptの違い

`llm_x/data.py` の `_array_text` と `questions.py` の `_text` は `np.array2string(..., precision=4, separator=", ")`。4桁固定小数ではなくNumPyの数値表示設定であり、dtype指定そのものは送られない。ブラケットの階層は維持する。indexed historyは `Step t:` に state/action/reward を続け、step間は空行。unindexedではslice全体を `states: / actions: / rewards:` として表示。linewidth/thresholdは明示されていないのでNumPy既定値の改行や `...` 省略が起こり得る。

Fetchだけraw26次元の最後を除き25次元を文字列化する。optional fieldsと `episodic_return` はhistoryへ追加しない。`next-state` 質問内のrewardは `Episode.reward()` がscalarをPython floatへ変換するので、history中のarray表記とは異なる。質問によって現在state、次state、action等を挿入する。

最終的な送信対象はtask説明から作るsystem文字列と、history＋質問から作るuser文字列でありNPZ自体ではない。詳しい実装経路、history範囲、現在の対応taskは[workspace README](../README.md)を参照。
"""
    return s


INTRO = {
"calculated_moves": "[Calculated Moves](https://data.4tu.nl/articles/_/12688547/1)、DOI `10.4121/uuid:3521e3e6-a05a-4b9c-9151-c269c15b7f30`。CC BY 4.0。元の4tu mirror URLは取得不能だったため、同名資料のDOIから公式archiveを取得済み。data.zipのMD5は `8c1c24e77633c4d6f5c7f0fa4cfb0b4e`（取得時一致）。全nested ZIP展開済み。",
"aircombat_wez": "[AirCombat-WEZ](https://github.com/andrekuros/AirCombat-WEZ)、commit `5ae42f4e56362e4df2bf29c5c01cbfb50c5795ed`。CC0 1.0。4 input CSVに加えて8 Paper_Results CSVも全件解析した。",
"f16capstone": "[F16Capstone](https://github.com/camdeno/F16Capstone)、commit `acedfe4b2401b600a89f14f771dbcdf88c803b14`。repositoryはMIT。実飛行、SITL、FlightAxis/RealFlight、モデル・測定資料を含む。機種の同一性を含め、異なる記録を同一flightとしてまとめない。",
"trajair": "[TrajAir公式配布](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251)、DOI `10.1184/R1/14866251.v1`。CC BY 4.0。111_days.zipとweather_data.zipを取得済み。MD5は順に `50dc9f4d271da435b6f1be2d13e70202`、`279aa9141d77ec9b86d952e9a1448570`（取得時一致）。7days1〜4は111日版に含まれる部分集合のため未取得。",
}

NARRATIVE = {
"calculated_moves": """# 1 record / timestep の意味

`blue1.csv` 等の行はrule weightと `fitness`, `avgfitness` の推移。**飛行simulationのtimestep-level trajectoryではない**。runはtimestamp付きdirectory、encounterはscript内 `Encounter n` とwinHistoryの項目順で識別する。1runのagent CSVには `maxEncounters+1` 行があり、初期値と更新後の行を含む。最初のfitness=0行を「第1encounterの結果」と誤認しない。原READMEの「nth encounter開始時のweight」とreward更新の厳密なoffsetは生成コードなしでは一意に特定できない。

`scripts-blue*.txt` は `[weight] [priority] rule name` をEncounterブロックにまとめた構造化text。全行をparseして異常行数を記録。空scriptもあり、原READMEは固定script制御の場合があると説明する。`winHistory.csv` はheaderなし横1行のblue/red列（物理shape=[1,100/150/300]、論理的には同数のencounter outcomes）。`Validation.csv` は **semicolon区切り468回答×10列** のATACC質問票で、飛行軌跡ではない。`flight` は閲覧した映像番号、`tactical` は匿名回答者ID。

# trajectory構造

4,200 learning runs。Chapter3はcent/decent/tacticのcoordination、Chapter4はreward条件、Chapter5は2v1-transfer-sources / 2v2-no-transfer / 2v2-transfer-with-mixed-as-source。blue1/blue2=lead/wingman、redまたはred1/red2=敵側。filenameをagent ID、pathをrun/scenario IDとして利用できる。trajectory episodeとlearning run/encounterは別物。

# state候補

rule weight配列はencounter-level学習状態の候補だが、飛行状態の代用品にはならない。

| 調査field | 実ファイルでの結論 |
| --- | --- |
| scenario ID / episode ID | 条件path / timestamp付きrun directoryとEncounter番号。独立したflight episode ID列はなし |
| timestep / simulation time | 飛行timestep・simulation timeはなし。configのspeedはsimulation設定であり時刻ではない |
| aircraft / agent ID / team | blue1,blue2,red,red1,red2のfilenameとblue/red outcome |
| position x/y/z・latitude/longitude・altitude | なし |
| velocity・acceleration・heading・pitch・roll・attitude | なし |
| control input | なし |
| selected action / maneuver | encounter開始時の選択rule scriptあり。実行されたmaneuverの時刻列はなし |
| reward / cumulative reward | fitnessとavgfitnessあり。cumulative reward列はなし（avgfitnessは累積和ではない） |
| target aircraft | default-target等のrule名はあるが、対象機ID時系列はなし |
| weapon state / missile / fire event | rule名にfire/missileがある。weapon state値・実発射eventログはなし |
| hit / kill event / terminal state | winHistoryの勝利teamはあるが、hit/kill時刻・terminal state vectorはなし |
| 1v1 / 2v1 / 2v2 | 2v1・2v2条件pathあり。独立1v1条件pathなし |

# action候補

encounter単位の選択ruleとweight/priority。flight control inputやstep actionは存在しない。

# reward候補

`fitness`, `avgfitness` はencounter-level。具体的reward関数や更新時刻の完全復元は未確認。winHistoryは勝利teamの結果ラベル。

# LLM-Xavierとの互換性

**Cを維持**。飛行state/actionの時系列が存在せず、flight-level Mental Modeling historyを復元できない。encounter-level dynamic scriptingを別のtaskとして定義する研究には使えるが、同じ再評価実験ではない。

# 不明点

原READMEに列挙された `fitness.csv`, `fitness2.csv`, `scripts-red.txt` は実配布ファイルには存在しない。原READMEのValidationData.csvに対し実名はValidation.csv。ruleの厳密な動作定義はthesis参照が必要。飛行kinematicsをrule名から推測していない。

""",
"aircombat_wez": """# 1 record / timestep の意味

Data内の1行は独立した射手(BL)と標的(RD)の初期交戦条件に対する射程計算case。時刻順のflight sampleではない。`Case` はcase番号、`Unnamed: 0` はjoined CSVの保存indexでありtimestampではない。Paper_Resultsの1行はregressor・前処理・学習条件別の誤差/実行時間の集計で、航空機状態ではない。

# trajectory構造

FactorialExperimentの864行はfactorial cases。3個のRandomExperiment_1000_* はそれぞれ1,000行。NEZ/RMAX/WEZは関連する実験表なので独立した3,000 flightと数えない。Caseと7入力の対応検査は後掲。12 CSVすべてにflight timestampや永続agent ID、episode境界はない。

# state候補

input variables=`BL_Speed, RD_Speed, rad, BL_Hdg, RD_Hdg, BL_Alt, RD_Alt`。両機の速度・heading・altitudeと相対幾何の初期条件。raw headerは単位を明記しない。repository READMEは速度NM/hour・角度degree・高度ftと説明するが、factorialのBL_Alt実値は304.8/7620/13716でありREADMEの高度範囲1,000〜45,000 ftとスケールが異なる。raw高度の単位を一意に確認できず、今回勝手に換算していない。連続flightの状態列は存在しない。

# action候補

なし。操舵、throttle、時系列maneuverは記録されていない。

# reward候補

なし。`maxRange`, `RMax`, `RNez` は射程等のtarget/output、`minErrorTry` は計算結果の補助値。RL rewardではない。Paper_ResultsのMAE/RMSEは学習器の誤差指標でありflight rewardではない。

# LLM-Xavierとの互換性

**C**。時系列のstate-action-reward historyは作れない。静的なWEZ回帰taskのデータとして利用できる。

# 不明点

Case順を飛行時刻順に読み替える根拠は存在しない。制御・rewardを復元することはできない。重複やcase対応の実測値は後掲に保存。

""",
"f16capstone": """# 1 record / timestep の意味

3 CSVの1行はFlightAxisの同一physics timeで得た55状態/RC値。空行はrecordとして数えない。2 clipped MATの1行も `vals` の55値。CSVとMATを全て別flight数として合算しない（clipped segmentや同名exportが重複する可能性）。3 ULogはtopicごとに異なるtimestampを持ち、1topic messageは同時刻の全機状態ではない。

# trajectory構造

3 CSV recording blocks、2 numeric MAT clipped segments、2 opaque full MAT tables、3 ULog sessions。`Chirp.mat` はtimeseries object、`Doublet.mat` はSimulink.SimulationOutputで、SciPyでは内部時系列を一意に復元できない。`Control_Inputs.mat` と `time.mat` は補助配列で独立trajectoryではない。拡張子のないRacetrack Log/Rascal Flight Test CamはPX4ログへのURL1行だけで、追加のdownload済みULogではない。

ULogの `FlightData/FlightTest1` 系pathは実飛行、`SITL/Practice SITL Log Files` はsimulation練習。FlightAxis CSV/MATはRealFlight simulator観測で、実飛行sensorとは区別する。XLSXは機材/慣性モーメント/trim等の資料であり、flight時系列と自動的にみなさない。

# state候補

CSV列のm-をm_に置換した順序が `Mathematical Model/System Identification/prep_flight_data.m` の55-column headerと一致する。位置x/y、ASL/AGL高度、対気/対地速度、world/body速度・加速度、wind、角速度、方位/姿勢、quaternion、機体statusなど。センサ由来とsimulation真値はsource区分を保持する。inclinationはAoAを直接表すとは限らない（同コード内にも注記）。55列の全名・dtypeは後掲。

# action候補

同一CSV行の `rc_channel_0/1/2/3` がaileron/elevator/throttle/rudder入力（prep_flight_data.mのコメント）。RC指令は実舵面角のsensor measurementとは異なる。残るRC channelの用途は確定できるもの以外を推測しない。ULogの `actuator_controls_0.control[*]`, `actuator_outputs.output[*]`, `input_rc`, `manual_control_setpoint` は指令/出力の異なる階層であり交換可能ではない。

# reward候補

flightデータのrewardは存在しない。安全性や追従誤差から新たにrewardを設計するなら別途定義が必要。

# LLM-Xavierとの互換性

**B**。単調なphysics timeを持つCSVまたはnumeric clipped MATでは、同じ行から `state_t` と `control_t`、次行から `state_{t+1}` を選び **(state, action, next_state) は構成可能、rewardは存在しない**。不等間隔なのでdelta_tも保持する。opaque MAT単体は内部未復元のためこの判定対象から除く。

ULogではtimestampを基準にstate topicとcontrol topicを同期する必要がある。将来値を使う補間はリークになるため、たとえば過去向きas-of join＋許容遅延＋欠損maskを明示する。episode境界、再sampling、入力遅延、trimや物理単位、機種と実飛行/SITLの区分は別途決定する。今回変換・simulationは実行していない。

# 不明点

2 opaque full MATのrow数、Chirp/Doublet内部trajectoryのshapeは未確認。MATLAB内部metadataは観測変数ではない。巨大MCOS内部workspaceは展開せず、既知のtop-level変数だけ選択して読む。収録全体の独立flight総数・MAT/CSV間の完全同一性は一意に特定できない。ネット上のULog URL先は今回は取得していない。

""",
"trajair": """# 1 record / timestep の意味

processed TXTはheaderなし、空白区切り7列。1行はscene内の1航空機の1frame観測。全列順は `frame_number, aircraft_id, x_km, y_km, z_km, wind_x_mps, wind_y_mps`。位置は空港基準座標(km)、風はm/s。単位と1Hzは同梱README.source.txtを根拠とする。rawはADS-BのID/Date/Timeと位置等、weatherは観測所とvalid時刻の気象観測。

# trajectory構造

`(train/testを含む相対filename, aircraft_id)` をtrackキー、`frame_number` を時刻順序とする。異なるsceneの同じIDを無条件に結合しない。同じscene・同じframeの異なるIDは同時航空機。欠測gapを分割しないtrack数と、delta_frame=1だけで連続とするsegment数を区別する。これは実flight離着陸境界を同定した数ではない。

```text
scene relative_path
  aircraft_id
    frame f:     x, y, z, wind_x, wind_y
    frame f + 1: x, y, z, wind_x, wind_y
    frame f + k: gap（k>1なら連続segmentの切れ目候補）
```

rawでは日別file＋ID＋Date/Timeで記録をまとめられるが、1日同一IDに複数flightが含まれ得る。rawとprocessedのtrack数が異なるのは定義/抽出が異なるため。raw timestampの順序差分も全ファイルで集計し、規則的1Hzと決めつけない。

# state候補

processedのxyz、wind xy。rawのLat/Lon、Altitude、Speed、Headingも候補。processedに速度/headingそのものの列はなし（位置差分から導く量は派生特徴と明記）。rawは13列と14列があり `AltisGNSS` の有無が異なる。flight/trajectory ID列はなし。weatherのwind contextはstation/validを基準に時刻・空間対応を定義する必要がある。位置と風の単位を混ぜない。

# action候補

action/controlは存在しない。位置差分や速度変化を作っても真の操縦指令ではなくderived motion proxy。

# reward候補

存在しない。研究目的に応じたreward定義が必要。

# LLM-Xavierとの互換性

**B**。sequentialなstate/next_stateは作れる。actionとrewardは未提供なので、公式NPZへ直結はできない。自然な用途は複数航空機trajectory prediction。実controlが必要な評価taskへの転用は制約がある。

# 不明点

同梱READMEの2,731,256行に対し実ファイルは2,731,255行（差1）。原因は一意に特定できない。元のREADMEで空ファイルを1件とした記述は誤りで、今回全走査ではprocessed7件とraw2件。収集日は2020-09-12〜2021-04-27と記載されるが、全期間が毎日連続ではない。raw day directory数・実Date値数・非空file数を区別する。licenseは公式配布のCC BY 4.0を採用し、別論文のCC0との記述に置き換えない。

""",
}


def candidate_doc(name, result):
    files = [r for r in result["files"] if r["dataset"] == name]
    groups = {k: g for k, g in result["groups"].items() if g["dataset"] == name}
    total = result["datasets"][name]
    s = f"# 概要\n\n{INTRO[name]}\n\n2026-09-09のdownload済み実ファイル全走査。元データは変更していない。\n\n# ファイル構成\n\n"
    s += table(["format", "file数"], sorted(total["format_counts"].items()))
    s += "全filenameとbytesは `outputs/dataset_inventory/candidate_files.csv`（datasetでfilter）に記録。schema、file別missing/unique/type・shape、archive treeは `candidate_schema.json`。support file（code/model/media/PDF）はmetadata inventoryのみで、flight dataとして実行/deserializeしていない。上流READMEは原文保存し、このSCHEMA.mdを日本語の調査結果とする。\n\n# データ量\n\n"
    s += table(["量", "値"], total.items())
    s += "bytesはファイル内容のサイズ合計で、filesystem allocation（du）ではない。`.git`, `.cache` と自作wrapper文書を除外。top-level ZIP、nested ZIP、展開物を区別しており、ZIPの展開後サイズを現在のdisk bytesへ重複加算しない。Git取得でZIPが0なのは圧縮配布物を保存していない意味で、Git pack size=0の主張ではない。\n\n# Schema\n\n"
    s += "schema fingerprintは列名順序＋reader dtype＋delimiter/header等の構造からSHA256の先頭16桁を計算。CSVの行数・missing数をfingerprintへ混ぜない。MATはvariable shapeを含む。dtypeはCSVにnative保存された型ではなくreaderの推定結果。空fileは未知schemaとして独立。文字列/全欠損/数値型の差も分離する。\n\n"
    s += table(["schema group", "file数", "物理rows合計", "kind", "例filename"], [[k, g["file_count"], g["rows"], g["schema"]["kind"], g["example_file"]] for k, g in groups.items()])
    s += NARRATIVE[name]
    if name == "trajair":
        t = result["trajair"]
        s += "## 全日・全sceneの実測集計\n\n"
        s += table(["項目", "値"], [[k, v] for k, v in t.items() if not k.endswith("counts") and k not in {"raw_dates", "empty_files"}])
        s += f"raw day directory数=111、raw非空file数={sum(bool(r.get('rows')) for r in files if '/raw_data/' in r['relative_path'])}、実Date値数={len(t['raw_dates'])}。processed train=2,230、test=858。\n\n"
        for field in ("frame_delta_counts", "raw_time_delta_ms_counts"):
            counts = t[field]
            s += f"`{field}`: 異なるdelta={len(counts)}、min={min(map(float, counts)) if counts else None}、max={max(map(float, counts)) if counts else None}、頻度上位=" + str(sorted(counts.items(), key=lambda x: -x[1])[:8]) + "。完全な頻度はJSON参照。\n\n"
        s += "空processed filename:\n\n" + "\n".join(f"- `{f}`" for f in t["empty_files"]) + "\n\n空raw filename: `12-01-20_adsb/1.csv`, `12-02-20_adsb/1.csv`。\n\n"
    if name == "calculated_moves":
        conf = [r for r in files if "configuration" in r]
        s += "## run設定・script全件集計\n\n"
        s += table(["設定", "run数"], Counter(str(r["configuration"]) for r in conf).items())
        scripts = [r for r in files if "encounter_count" in r]
        s += f"scripts {len(scripts)} files、空script {sum(r['encounter_count']==0 for r in scripts)} files、parseできない非空行 {sum(len(r['unmatched_lines']) for r in scripts)}。\n\n"
        s += table(["basename", "file数"], Counter(Path(r["relative_path"]).name for r in files).items())
    if name in {"aircombat_wez", "f16capstone"}:
        s += "## 全CSVのrows・shape・sampling\n\n"
        s += table(["filename", "bytes", "rows", "columns", "time開始/終了", "median delta (s)"], [[r["relative_path"], r["size_bytes"], r.get("rows"), r.get("columns"), str((r.get("time", {}).get("start"), r.get("time", {}).get("end"))), r.get("time", {}).get("positive_delta", {}).get("median")] for r in files if r["format"] == ".csv"])
    s += "# Schema group別の全column / variable\n\n同じschemaは全columnを一度掲載。少数groupはfilenameも記録し、多数groupの全対応はcandidate_files.csvで検索できる。missingはgroup内合計、uniqueのfile別値はJSON内 `unique`。空文字やNA/NaN等を欠損とし、weatherのみMも欠損扱い。XLSXは全OOXML cellを読み、formulaを実行せずcached typeとtext cellを調査する。\n\n"
    for k, g in groups.items():
        schema = g["schema"]
        if schema["kind"] == "support_file":
            continue
        s += f"## {k}\n\n{g['file_count']} files。\n\n"
        members = [r for r in files if r["schema_group"] == k]
        if len(members) <= 5:
            s += "\n".join(f"- `{r['relative_path']}`" for r in members) + "\n\n"
        if "fields" in schema:
            s += table(["全column/field名（順序保持）", "dtype", "missing合計"], [[f["name"], f["dtype"], g["missing"].get(f["name"], 0)] for f in schema["fields"]])
        elif schema["kind"] == "MAT":
            for r in members:
                s += f"`{r['relative_path']}`\n\n"
                s += table(["variable", "MATLAB class", "NumPy/Python type", "dtype", "shape", "ndim", "num_elements", "struct field"], [[n, v.get("matlab_object_class", v["matlab_class"]), v["python_type"], v["dtype"], v["shape"], v["ndim"], v["num_elements"], v["struct_fields"]] for n, v in r["variables"].items()])
                for n, v in r["variables"].items():
                    if "time" in v:
                        s += f"`{n}` 時間統計:\n\n" + jblock(v["time"])
                s += "opaque objectのshape=[1]はMATLAB tableの1行という意味ではない。内部metadataはtrajectory変数ではなくrow数の根拠にしない。\n\n" if any(v["opaque"] for v in r["variables"].values()) else ""
        elif schema["kind"] == "ULog":
            for r in members:
                s += f"duration={r['duration_seconds']} s。全topicのfield dtype/shapeを以下に掲載。\n\n"
                s += table(["topic / multi_id", "rows", "median delta(s)", "全fields: dtype shape"], [[f"{t['name']}/{t['multi_id']}", t["rows"], (t["time_seconds"] or {}).get("positive_delta", {}).get("median"), "; ".join(f"{n}: {a['dtype']} {a['shape']}" for n, a in t["arrays"].items())] for t in r["streams"]])
        elif schema["kind"] == "XLSX":
            for r in members:
                s += table(["sheet", "range", "populated cells", "OOXML cell types", "formula数"], [[t["name"], t["dimension"], t["populated_cells"], t["cell_types"], t["formula_count"]] for t in r["sheets"]])
                s += "全text cell（column labelを含む）はJSONのsheets/text_cellsへcell address付きで保存。フラットflight tableではないので、最初の行を一律headerと推測しない。\n\n"
        elif schema["kind"] == "archive":
            for r in members:
                a = r["archive"]
                s += f"`{r['relative_path']}`: {a['file_count']}内部files、uncompressed={a['uncompressed_bytes']} bytes、formats={a['format_counts']}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。\n\n"
        else:
            s += jblock(schema)
    return s


def main():
    inv = json.loads((ROOT / "outputs/dataset_inventory/llmx_schema.json").read_text())
    result = json.loads((ROOT / "outputs/dataset_inventory/candidate_schema.json").read_text())
    _atomic_text(ROOT / "outputs/docs/llmx_schema.md", official_doc(inv))
    for name in INTRO:
        doc = candidate_doc(name, result)
        relation_path = ROOT / "outputs/dataset_inventory/relationships.json"
        if relation_path.exists():
            relations = json.loads(relation_path.read_text())
            if name in relations:
                doc += "# ファイル間対応の全件照合\n\n`tools/audit_dataset_relationships.py` の実測。sourceコードは実行せず比較のみ。\n\n" + jblock(relations[name])
                if name == "f16capstone":
                    doc += "全1,062行の照合で `control_IN` 列順は **rc2, rc1, rc0, rc3 = throttle, elevator, aileron, rudder**。raw rc0〜3順と異なる。time.matはclipped1のphysics timeと完全一致。clipped1/2の全timestampは同名25_05_2021 CSV内に一致するため、独立flightとして合算しない。\n\n"
                if name == "aircombat_wez":
                    doc += "WEZ/NEZ/RMAXの全1,000 Caseで7入力が一致し、RNez/RMaxのoutputも対応表で全件一致（rtol=1e-9, atol=1e-10）。1,000 random casesの関連exportであり3,000独立caseではない。864 factorialの入力tupleは全件unique。周辺level集合の直積を全組合せ採用したと仮定しない。\n\n"
        _atomic_text(ROOT / f"outputs/docs/{name}/SCHEMA.md", doc)


if __name__ == "__main__":
    main()
