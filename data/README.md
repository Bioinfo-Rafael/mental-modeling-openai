# 公式 llmx_data の詳細schema

## 出典・調査範囲

[Hugging Face](https://huggingface.co/datasets/lerrhoo/llmx_data) の固定revision `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`。全172 NPZを `allow_pickle=False` で読み、全183ファイルを列挙した。非cacheサイズは56,861,285 bytes。上流の記載ではCC BY 4.0。policy checkpoint・TensorBoard等の11ファイルはパス/形式/bytesを記録し、危険なpickle等のdeserializeはしない。

再取得手順（今回は再downloadしていない）:

```bash
hf download lerrhoo/llmx_data --repo-type dataset --revision dc2b798f72bc02f7285949ccfcdcb42e5ff326fd --local-dir data/llmx_data
```

## 全20 task

`T` は各episodeの長さ。shapeの先頭以外の `1` もrawに実在し、説明上勝手に削除しない。すべてのkeyのPython型は `numpy.ndarray`、object arrayはなし。action typeは実配列・上流task定義に基づく。rewardは全taskで1成分。

NPZ container合計=1840635 bytes、数値ndarray payload合計=1610187 bytes。NPZ内の各NPY entryのcompressed/uncompressed bytesとcompression methodもJSONへ保存（ZIP containerが常に圧縮されているとは限らない）。task間で共有するschemaを重複除外すると13群。

| Task | Episodes | Timesteps | State shape | State dtype | Action shape | Action dtype | Action type | Reward shape | Optional fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiniGrid-DoorKey-5x5-v0 | 24 | 226 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| MiniGrid-Empty-Random-5x5-v0 | 11 | 30 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| MiniGrid-Fetch-5x5-N2-v0 | 17 | 210 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| MiniGrid-GoToDoor-5x5-v0 | 5 | 307 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| MiniGrid-KeyCorridorS3R1-v0 | 11 | 162 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| MiniGrid-Unlock-v0 | 10 | 130 | [T, 7, 7, 3] | uint8 | [T, 1] | int64 | discrete | [T, 1] | agent_dirs, dir_vectors |
| HalfCheetah-v4 | 3 | 3000 | [T, 1, 17] | float64 | [T, 1, 6] | float32 | continuous | [T, 1] | なし |
| Pusher-v4 | 6 | 600 | [T, 1, 23] | float64 | [T, 1, 7] | float32 | continuous | [T, 1] | なし |
| Reacher-v4 | 5 | 250 | [T, 1, 11] | float64 | [T, 1, 2] | float32 | continuous | [T, 1] | なし |
| Acrobot-v1 | 5 | 399 | [T, 1, 6] | float32 | [T, 1] | int64 | discrete | [T, 1] | なし |
| BipedalWalker-v3 | 2 | 144 | [T, 1, 24] | float32 | [T, 1, 4] | float32 | continuous | [T, 1] | なし |
| CartPole-v1 | 5 | 2500 | [T, 1, 4] | float32 | [T, 1] | int64 | discrete | [T, 1] | なし |
| InvertedDoublePendulum-v4 | 10 | 538 | [T, 1, 11] | float64 | [T, 1, 1] | float32 | continuous | [T, 1] | なし |
| InvertedPendulum-v4 | 5 | 1419 | [T, 1, 4] | float64 | [T, 1, 1] | float32 | continuous | [T, 1] | なし |
| LunarLander-v2 | 3 | 744 | [T, 1, 8] | float32 | [T, 1] | int64 | discrete | [T, 1] | なし |
| MountainCar-v0 | 10 | 1041 | [T, 1, 2] | float32 | [T, 1] | int64 | discrete | [T, 1] | なし |
| Pendulum-v1 | 10 | 2000 | [T, 1, 3] | float32 | [T, 1, 1] | float32 | continuous | [T, 1] | なし |
| FetchPickAndPlace-v2 | 10 | 500 | [T, 1, 26] | float64 | [T, 1, 4] | float32 | continuous | [T, 1] | achieved_goals, desired_goals |
| FetchPush-v2 | 10 | 500 | [T, 1, 26] | float64 | [T, 1, 4] | float32 | continuous | [T, 1] | achieved_goals, desired_goals |
| FetchSlide-v2 | 10 | 500 | [T, 1, 26] | float64 | [T, 1, 4] | float32 | continuous | [T, 1] | achieved_goals, desired_goals |

合計 **20 tasks / 172 episodes / 15,200 timesteps**。各task内はTを可変とするschemaで統一されており、dtype・key・非T軸が異なるepisodeはなし。exact shapeはTが違えば異なる。全filename、T、bytes、dtype、shape、ndim、要素数、8 keyの存在判定を `outputs/dataset_inventory/llmx_schema.json` と `llmx_files.csv` に保存。task集計は `llmx_tasks.csv`。旧 `outputs/dataset_inventory.{json,csv}` も互換出力する。

`T=1` の3件はinventoryには含めるが、上流 `Episode.load` の最低2step条件を満たさない:

- `MiniGrid-Empty-Random-5x5-v0/episode_1_seed42.npz`
- `MiniGrid-Empty-Random-5x5-v0/episode_2_seed213.npz`
- `MiniGrid-Fetch-5x5-N2-v0/episode_1_seed100.npz`

## 各taskの1 timestepとdimension

以下の `s` は説明用に `states[t]` をflattenしたindex。表示例自体はrawの階層を保持し、先頭episodeの実測t=0を省略せず示す。`episodic_return` はepisode単位であり、stepごとのrewardと区別する。`(state_t, action_t, reward_t)` の列がT個並ぶ。NPZに `next_states`、`terminated`、`truncated`、timestampは存在しない。次状態は同一episode内の隣接stateで扱えるが、最終行の次状態と収集時のreward時刻規約はデータ単体では確認できない。

### MiniGrid-DoorKey-5x5-v0

24 episodes、226 steps。Tのmin/mean/max=7/9.41667/13。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-DoorKey-5x5-v0/episodes/episode_0_seed0.npz`（3547 bytes、T=11）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [11, 1] | 2 | 11 |
| agent_dirs | numpy.ndarray | int64 | [11, 1] | 2 | 11 |
| dir_vectors | numpy.ndarray | int64 | [11, 2] | 2 | 22 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [11, 1] | 2 | 11 |
| states | numpy.ndarray | uint8 | [11, 7, 7, 3] | 4 | 1617 |

```text
states[0] = [[[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [1, 0, 0],
  [5, 4, 0],
  [1, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [4, 4, 2],
  [2, 5, 0],
  [2, 5, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]]]
agent_dirs[0] = [3]
dir_vectors[0] = [ 0, -1]
actions[0] = [1]
rewards[0] = [0.]
episodic_return = [0.9603999853134155]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MiniGrid-Empty-Random-5x5-v0

11 episodes、30 steps。Tのmin/mean/max=1/2.72727/6。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-Empty-Random-5x5-v0/episodes/episode_0_seed0.npz`（2266 bytes、T=4）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [4, 1] | 2 | 4 |
| agent_dirs | numpy.ndarray | int64 | [4, 1] | 2 | 4 |
| dir_vectors | numpy.ndarray | int64 | [4, 2] | 2 | 8 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [4, 1] | 2 | 4 |
| states | numpy.ndarray | uint8 | [4, 7, 7, 3] | 4 | 588 |

```text
states[0] = [[[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [8, 1, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]]]
agent_dirs[0] = [1]
dir_vectors[0] = [0, 1]
actions[0] = [2]
rewards[0] = [0.]
episodic_return = [0.9639999866485596]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MiniGrid-Fetch-5x5-N2-v0

17 episodes、210 steps。Tのmin/mean/max=1/12.3529/125。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-Fetch-5x5-N2-v0/episodes/episode_0_seed0.npz`（2083 bytes、T=3）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [3, 1] | 2 | 3 |
| agent_dirs | numpy.ndarray | int64 | [3, 1] | 2 | 3 |
| dir_vectors | numpy.ndarray | int64 | [3, 2] | 2 | 6 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [3, 1] | 2 | 3 |
| states | numpy.ndarray | uint8 | [3, 7, 7, 3] | 4 | 441 |

```text
states[0] = [[[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [5, 2, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [6, 3, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]]]
agent_dirs[0] = [2]
dir_vectors[0] = [-1,  0]
actions[0] = [2]
rewards[0] = [0.]
episodic_return = [0.9783999919891357]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MiniGrid-GoToDoor-5x5-v0

5 episodes、307 steps。Tのmin/mean/max=3/61.4/100。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-GoToDoor-5x5-v0/episodes/episode_0_seed0.npz`（2266 bytes、T=4）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [4, 1] | 2 | 4 |
| agent_dirs | numpy.ndarray | int64 | [4, 1] | 2 | 4 |
| dir_vectors | numpy.ndarray | int64 | [4, 2] | 2 | 8 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [4, 1] | 2 | 4 |
| states | numpy.ndarray | uint8 | [4, 7, 7, 3] | 4 | 588 |

```text
states[0] = [[[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [4, 1, 1]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [4, 4, 1],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [4, 2, 1]],

 [[2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]]]
agent_dirs[0] = [3]
dir_vectors[0] = [ 0, -1]
actions[0] = [2]
rewards[0] = [0.]
episodic_return = [0.9639999866485596]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MiniGrid-KeyCorridorS3R1-v0

11 episodes、162 steps。Tのmin/mean/max=14/14.7273/15。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-KeyCorridorS3R1-v0/episodes/episode_0_seed0.npz`（4279 bytes、T=15）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [15, 1] | 2 | 15 |
| agent_dirs | numpy.ndarray | int64 | [15, 1] | 2 | 15 |
| dir_vectors | numpy.ndarray | int64 | [15, 2] | 2 | 30 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [15, 1] | 2 | 15 |
| states | numpy.ndarray | uint8 | [15, 7, 7, 3] | 4 | 2205 |

```text
states[0] = [[[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [4, 2, 1],
  [1, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]]]
agent_dirs[0] = [2]
dir_vectors[0] = [-1,  0]
actions[0] = [1]
rewards[0] = [0.]
episodic_return = [0.949999988079071]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MiniGrid-Unlock-v0

10 episodes、130 steps。Tのmin/mean/max=9/13/22。state/action/reward要素数 per step=147/1/1。

1 stateは7×7の局所観測で各cellが `[OBJECT_TO_IDX, COLOR_TO_IDX, STATE]`。画像のRGBではない。objectはunseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10。colorはred=0, green=1, blue=2, purple=3, yellow=4, grey=5。door STATEはopen=0, closed=1, locked=2。`agent_dirs`=0右/1下/2左/3上、`dir_vectors`=方向の2成分。actionは整数（0左回転,1右回転,2前進,3pickup,4drop,5toggle,6done）。根拠は `MiniGridUnlockTask` と [MiniGrid公式encoding](https://minigrid.farama.org/environments/minigrid/UnlockEnv/)。他のMiniGrid taskも実配列は同じschemaだが、各taskの収集wrapper・mission文字列はこのNPZには存在しない。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=あり、`desired_goals`=なし、`dir_vectors`=あり、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/minigrid_data/raw_transitions/MiniGrid-Unlock-v0/episodes/episode_0_seed0.npz`（4645 bytes、T=17）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [17, 1] | 2 | 17 |
| agent_dirs | numpy.ndarray | int64 | [17, 1] | 2 | 17 |
| dir_vectors | numpy.ndarray | int64 | [17, 2] | 2 | 34 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [17, 1] | 2 | 17 |
| states | numpy.ndarray | uint8 | [17, 7, 7, 3] | 4 | 2499 |

```text
states[0] = [[[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0],
  [2, 5, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [5, 3, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0]],

 [[0, 0, 0],
  [0, 0, 0],
  [2, 5, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0]]]
agent_dirs[0] = [3]
dir_vectors[0] = [ 0, -1]
actions[0] = [1]
rewards[0] = [0.]
episodic_return = [0.9468749761581421]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### HalfCheetah-v4

3 episodes、3000 steps。Tのmin/mean/max=1000/1000/1000。state/action/reward要素数 per step=17/6/1。

`s[0]`=rootz、`s[1]`=rooty角、`s[2:8]`=bthigh,bshin,bfoot,fthigh,fshin,ffootの角度（この順）。`s[8]`=rootx速度、`s[9]`=rootz速度、`s[10]`=rooty角速度、`s[11:17]`=上記6関節角速度。actionも上記6関節順。標準v4の `_get_obs()` と実測17次元は整合するが、収集時wrapperの完全な構成は未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/half_cheetah_v4.py)。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/mujoco_data/raw_transitions/HalfCheetah-v4/episodes/episode_0_seed1.npz`（169024 bytes、T=1000）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [1000, 1, 6] | 3 | 6000 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [1000, 1] | 2 | 1000 |
| states | numpy.ndarray | float64 | [1000, 1, 17] | 3 | 17000 |

```text
states[0] = [[ 8.9720421888801977e-04, -7.0981467051410075e-04,
   8.9361962937328375e-04, -3.7603332431922800e-04,
  -1.5331374665583877e-04,  6.5393692321908314e-04,
  -1.8155363899338464e-04,  9.9172579833596398e-05,
   2.9397597271069506e-04,  2.8419284468101551e-05,
   5.4584334011663002e-04, -7.3439199434071234e-04,
  -1.6287205073097009e-04, -4.8151192891931067e-04,
   5.9771574960089402e-04,  3.9717822190389067e-05,
  -2.9230255388062770e-04]]
actions[0] = [[-0.08077647, -1.2811437 , -0.6492183 , -1.1980596 , -0.67304826,
   0.46901798]]
rewards[0] = [-0.9443352492820616]
episodic_return = [[261.29556]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### Pusher-v4

6 episodes、600 steps。Tのmin/mean/max=100/100/100。state/action/reward要素数 per step=23/7/1。

`s[0:7]`=shoulder pan, shoulder lift, upper arm roll, elbow flex, forearm roll, wrist flex, wrist rollの角度（この順）。`s[7:14]`=同順角速度、`s[14:17]`=fingertip xyz、`s[17:20]`=object xyz、`s[20:23]`=goal xyz。actionは7関節入力。標準v4と実測23次元が整合。収集時wrapperは未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/pusher_v4.py)。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/mujoco_data/raw_transitions/Pusher-v4/episodes/episode_0_seed1.npz`（23024 bytes、T=100）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [100, 1, 7] | 3 | 700 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [100, 1] | 2 | 100 |
| states | numpy.ndarray | float64 | [100, 1, 23] | 3 | 2300 |

```text
states[0] = [[ 0.0000000000000000e+00,  0.0000000000000000e+00,
   0.0000000000000000e+00,  0.0000000000000000e+00,
   0.0000000000000000e+00,  0.0000000000000000e+00,
   0.0000000000000000e+00, -3.5580255285026071e-05,
   4.4860007050835970e-05, -1.8814939904845235e-05,
  -7.6665861524882655e-06,  3.2766806615895860e-05,
  -9.0791746577959087e-06,  4.9588722455250496e-06,
   6.3450371323048212e-03, -5.1445792524711671e-03,
   0.0000000000000000e+00,  5.3311182519198402e-03,
  -1.9275030840456889e-03, -2.6513183260634598e-03,
   4.1033055221244346e-03, -4.9932635882651811e-04,
  -3.0733636961753717e-03]]
actions[0] = [[ 0.5168305 ,  0.57484514, -0.3237068 ,  0.19645268,  0.07236628,
   0.31401503, -0.6544804 ]]
rewards[0] = [-0.6265664751394259]
episodic_return = [[-60.563763]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### Reacher-v4

5 episodes、250 steps。Tのmin/mean/max=50/50/50。state/action/reward要素数 per step=11/2/1。

`s[0:2]`=2関節cos、`s[2:4]`=2関節sin、`s[4:6]`=target xy、`s[6:8]`=2関節角速度、`s[8:11]`=fingertip-target xyz。actionは2関節トルク。標準v4は11次元であり、現行v5の10次元説明で置き換えない。収集時wrapperは未確認。[v4実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/mujoco/reacher_v4.py)。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/mujoco_data/raw_transitions/Reacher-v4/episodes/episode_0_seed1.npz`（6224 bytes、T=50）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [50, 1, 2] | 3 | 100 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [50, 1] | 2 | 50 |
| states | numpy.ndarray | float64 | [50, 1, 11] | 3 | 550 |

```text
states[0] = [[ 7.0707043672529847e-03,  7.0563310978391126e-03,
   2.3640797105259494e-05,  8.9600073476944981e-04,
  -7.5047656042662137e-04, -3.0651944078251097e-04,
   3.2766806615895860e-05, -9.0791746577959087e-06,
   2.7388035284848463e-03,  4.1022978100918074e-04,
   0.0000000000000000e+00]]
actions[0] = [[0.040847793, 0.017244924]]
rewards[0] = [-0.2897081246601068]
episodic_return = [[-14.232287]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### Acrobot-v1

5 episodes、399 steps。Tのmin/mean/max=69/79.8/90。state/action/reward要素数 per step=6/1/1。

`s[0]`=cos(theta1)、`s[1]`=sin(theta1)、`s[2]`=cos(theta2)、`s[3]`=sin(theta2)、`s[4]`=theta1角速度、`s[5]`=theta2角速度。theta2は第1リンク相対。action 0/1/2 はトルク -1/0/+1。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/Acrobot-v1/episodes/episode_0_seed1.npz`（4024 bytes、T=75）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [75, 1] | 2 | 75 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [75, 1] | 2 | 75 |
| states | numpy.ndarray | float32 | [75, 1, 6] | 3 | 450 |

```text
states[0] = [[ 0.9999972   ,  0.0023643228,  0.9959444   ,  0.08997092  ,
  -0.07116808  ,  0.08972989  ]]
actions[0] = [0]
rewards[0] = [-1.]
episodic_return = [[-74.]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### BipedalWalker-v3

2 episodes、144 steps。Tのmin/mean/max=70/72/74。state/action/reward要素数 per step=24/4/1。

`s[0]`=胴体角、`s[1]`=正規化胴体角速度、`s[2:4]`=正規化水平/鉛直速度。`s[4:9]`=第1脚の股角/正規化股角速度/膝角+1/正規化膝角速度/接地、`s[9:14]`=第2脚の同5値、`s[14:24]`=10本のlidar fraction。位置座標は含まない。actionは股・膝・股・膝の4入力。標準コードと実測24次元が整合。収集時wrapperは未確認。[実装](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v0.29.1/gymnasium/envs/box2d/bipedal_walker.py)。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/BipedalWalker-v3/episodes/episode_0_seed1.npz`（9424 bytes、T=70）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [70, 1, 4] | 3 | 280 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [70, 1] | 2 | 70 |
| states | numpy.ndarray | float32 | [70, 1, 24] | 3 | 1680 |

```text
states[0] = [[ 2.7452144e-05,  1.2600077e-07, -1.6429498e-05, -1.5996436e-04,
   9.2152634e-04,  3.8139773e-05,  6.5187830e-03, -1.6300508e-05,
   7.0707141e-03,  3.2832442e-04,  3.8138420e-05,  6.4915535e-03,
  -2.5950530e-05,  7.0707141e-03,  4.0332824e-03,  4.0715290e-03,
   4.1893646e-03,  4.3965341e-03,  4.7107930e-03,  5.1600644e-03,
   5.7842112e-03,  6.6308873e-03,  7.0707141e-03,  7.0707141e-03]]
actions[0] = [[ 0.48204267,  0.5798578 ,  0.4753429 , -1.8412426 ]]
rewards[0] = [-10.]
episodic_return = [[-98.52386]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### CartPole-v1

5 episodes、2500 steps。Tのmin/mean/max=500/500/500。state/action/reward要素数 per step=4/1/1。

`s[0]`=台車位置、`s[1]`=台車速度、`s[2]`=棒の角度(rad)、`s[3]`=棒の角速度。action 0/1 は左/右。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/CartPole-v1/episodes/episode_0.npz`（17024 bytes、T=500）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [500, 1] | 2 | 500 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [500, 1] | 2 | 500 |
| states | numpy.ndarray | float32 | [500, 1, 4] | 3 | 2000 |

```text
states[0] = [[ 0.0011821624,  0.04504637  , -0.03558404  ,  0.044864945 ]]
actions[0] = [1]
rewards[0] = [1.]
episodic_return = [[500.]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### InvertedDoublePendulum-v4

10 episodes、538 steps。Tのmin/mean/max=31/53.8/135。state/action/reward要素数 per step=11/1/1。

`s[0]`=台車位置、`s[1]`=sin(theta1)、`s[2]`=sin(theta2)、`s[3]`=cos(theta1)、`s[4]`=cos(theta2)、`s[5]`=台車速度、`s[6]`=theta1角速度、`s[7]`=theta2角速度、`s[8:11]`=3自由度のconstraint force。theta1は台車と第1棒、theta2は棒間の角度。actionは連続1成分。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/InvertedDoublePendulum-v4/episodes/episode_0_seed1.npz`（5224 bytes、T=42）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [42, 1, 1] | 3 | 42 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [42, 1] | 2 | 42 |
| states | numpy.ndarray | float64 | [42, 1, 11] | 3 | 462 |

```text
states[0] = [[ 2.3640819130641867e-05,  8.9600073476944981e-04,
  -7.0921864711886160e-04,  7.0563310978391126e-03,
   7.0617470140708427e-03, -1.2921039106154862e-03,
   9.0157863092858791e-04,  4.4588603240189334e-04,
   0.0000000000000000e+00,  0.0000000000000000e+00,
   0.0000000000000000e+00]]
actions[0] = [[0.010357507]]
rewards[0] = [9.354445281529763]
episodic_return = [[385.77005]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### InvertedPendulum-v4

5 episodes、1419 steps。Tのmin/mean/max=5/283.8/1000。state/action/reward要素数 per step=4/1/1。

`s[0]`=台車位置、`s[1]`=棒の鉛直からの角度(rad)、`s[2]`=台車速度、`s[3]`=棒の角速度。actionは連続1成分。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/InvertedPendulum-v4/episodes/episode_0_seed100.npz`（45024 bytes、T=1000）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [1000, 1, 1] | 3 | 1000 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [1000, 1] | 2 | 1000 |
| states | numpy.ndarray | float64 | [1000, 1, 4] | 3 | 4000 |

```text
states[0] = [[ 6.6988123736702799e-05,  1.9308838414806137e-05,
  -4.2222752767382232e-05, -9.1396727760748167e-05]]
actions[0] = [[-19.267162]]
rewards[0] = [1.]
episodic_return = [[1000.]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### LunarLander-v2

3 episodes、744 steps。Tのmin/mean/max=246/248/252。state/action/reward要素数 per step=8/1/1。

`s[0]`=横位置、`s[1]`=縦位置、`s[2]`=横速度、`s[3]`=縦速度、`s[4]`=角度、`s[5]`=角速度、`s[6]`=第1脚接地、`s[7]`=第2脚接地。action 0=無操作、1=左姿勢エンジン、2=主エンジン、3=右姿勢エンジン。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/LunarLander-v2/episodes/episode_0.npz`（12832 bytes、T=246）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [246, 1] | 2 | 246 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [246, 1] | 2 | 246 |
| states | numpy.ndarray | float32 | [246, 1, 8] | 3 | 1968 |

```text
states[0] = [[-0.002718258 ,  1.4172997   , -0.27534413  ,  0.28354123  ,
   0.0031565526,  0.062369574 ,  0.          ,  0.          ]]
actions[0] = [3]
rewards[0] = [1.803151203822183]
episodic_return = [[261.0086]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### MountainCar-v0

10 episodes、1041 steps。Tのmin/mean/max=88/104.1/112。state/action/reward要素数 per step=2/1/1。

`s[0]`=x位置、`s[1]`=速度。action 0/1/2 は左加速/無加速/右加速。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/MountainCar-v0/episodes/episode_0.npz`（3544 bytes、T=105）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | int64 | [105, 1] | 2 | 105 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [105, 1] | 2 | 105 |
| states | numpy.ndarray | float32 | [105, 1, 2] | 3 | 210 |

```text
states[0] = [[-0.49763566,  0.        ]]
actions[0] = [2]
rewards[0] = [-1.]
episodic_return = [[-105.]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### Pendulum-v1

10 episodes、2000 steps。Tのmin/mean/max=200/200/200。state/action/reward要素数 per step=3/1/1。

`s[0]`=cos(theta)、`s[1]`=sin(theta)、`s[2]`=角速度。actionは1成分の連続トルク。

keyの有無: `achieved_goals`=なし、`actions`=あり、`agent_dirs`=なし、`desired_goals`=なし、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/physics_data/raw_transitions/Pendulum-v1/episodes/episode_0.npz`（5824 bytes、T=200）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| actions | numpy.ndarray | float32 | [200, 1, 1] | 3 | 200 |
| episodic_return | numpy.ndarray | float32 | [1, 1] | 2 | 1 |
| rewards | numpy.ndarray | float64 | [200, 1] | 2 | 200 |
| states | numpy.ndarray | float32 | [200, 1, 3] | 3 | 600 |

```text
states[0] = [[0.9972427  , 0.074209176, 0.90092736 ]]
actions[0] = [[-2.]]
rewards[0] = [-0.09068415754263699]
episodic_return = [[-1.1953256]]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### FetchPickAndPlace-v2

10 episodes、500 steps。Tのmin/mean/max=50/50/50。state/action/reward要素数 per step=26/4/1。

`s[0:3]`=gripper xyz、`s[3:6]`=object xyz、`s[6:9]`=object-gripper相対xyz、`s[9:11]`=左右finger変位、`s[11:14]`=object XYZ Euler角、`s[14:17]`=objectのgripper相対速度xyz、`s[17:20]`=object角速度xyz、`s[20:23]`=gripper速度xyz、`s[23:25]`=finger速度2値。これは `task.py` の25次元説明との対応。`s[25]`（rawに追加された26番目）は**意味を一意に確認できず**。時刻等と決めつけない。`task.py` は23と24をともにRightと記しており、左右の厳密な割当はその説明だけでは一意に特定できない。promptでは26番目を除外する。`achieved_goals`=現在object xyz、`desired_goals`=目標object xyz。actionは4成分（xyz変位指令とgripper開閉指令）。

keyの有無: `achieved_goals`=あり、`actions`=あり、`agent_dirs`=なし、`desired_goals`=あり、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/robotics_data/raw_transitions/FetchPickAndPlace-v2/episodes/episode_0_seed0.npz`（15346 bytes、T=50）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| achieved_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| actions | numpy.ndarray | float32 | [50, 1, 4] | 3 | 200 |
| desired_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [50, 1] | 2 | 50 |
| states | numpy.ndarray | float64 | [50, 1, 26] | 3 | 1300 |

```text
states[0] = [[ 1.3419548590699211e+00,  7.4910049654430777e-01,
   5.3470720455812448e-01,  1.2042470788792721e+00,
   6.0405878653747447e-01,  4.2470209084782223e-01,
  -1.3770778019064900e-01, -1.4504171000683330e-01,
  -1.1000511371030225e-01,  3.1845006837983406e-06,
  -4.9683582779975164e-08, -8.8244968542634476e-08,
   1.3576149002011685e-07,  3.7875054250030615e-15,
   3.2514113216212053e-06, -1.4363906573848636e-08,
   4.1661973449776800e-05,  5.0390716555683531e-08,
  -7.7524179302630672e-08,  2.4445459323963814e-18,
  -3.2533464143938494e-06,  1.3106096271248493e-08,
   5.1657922752134944e-06,  5.2633918892978691e-07,
   1.7535842477079927e-07,  1.0000000000000000e+00]]
achieved_goals[0] = [[1.204247078879272  , 0.6040587865374745 , 0.42470209084782223]]
desired_goals[0] = [[1.4359360934584955, 0.8729271690622322, 0.424699749459536 ]]
actions[0] = [[-0.96485746, -0.99953914,  0.4901874 , -0.6385241 ]]
rewards[0] = [-1.]
episodic_return = [-11.]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### FetchPush-v2

10 episodes、500 steps。Tのmin/mean/max=50/50/50。state/action/reward要素数 per step=26/4/1。

`s[0:3]`=gripper xyz、`s[3:6]`=object xyz、`s[6:9]`=object-gripper相対xyz、`s[9:11]`=左右finger変位、`s[11:14]`=object XYZ Euler角、`s[14:17]`=objectのgripper相対速度xyz、`s[17:20]`=object角速度xyz、`s[20:23]`=gripper速度xyz、`s[23:25]`=finger速度2値。これは `task.py` の25次元説明との対応。`s[25]`（rawに追加された26番目）は**意味を一意に確認できず**。時刻等と決めつけない。`task.py` は23と24をともにRightと記しており、左右の厳密な割当はその説明だけでは一意に特定できない。promptでは26番目を除外する。`achieved_goals`=現在object xyz、`desired_goals`=目標object xyz。actionは4成分（xyz変位指令とgripper開閉指令）。

keyの有無: `achieved_goals`=あり、`actions`=あり、`agent_dirs`=なし、`desired_goals`=あり、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/robotics_data/raw_transitions/FetchPush-v2/episodes/episode_0_seed0.npz`（15346 bytes、T=50）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| achieved_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| actions | numpy.ndarray | float32 | [50, 1, 4] | 3 | 200 |
| desired_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [50, 1] | 2 | 50 |
| states | numpy.ndarray | float64 | [50, 1, 26] | 3 | 1300 |

```text
states[0] = [[ 1.3481592236505489e+00,  7.4895090214469273e-01,
   4.1361868450079714e-01,  1.2104978245640396e+00,
   6.0390713255892414e-01,  4.2470209084782223e-01,
  -1.3766139908650921e-01, -1.4504376958576859e-01,
   1.1083406347025093e-02, -1.5094483631619841e-06,
   1.2515977221221049e-03, -8.8244968536260473e-08,
   1.3576149001724429e-07,  4.6786360883405283e-15,
   9.3091960051922842e-04, -4.1197993083757154e-05,
   3.8935046106396701e-05,  5.0390716556260982e-08,
  -7.7524179293556950e-08,  2.8743230251252930e-20,
  -9.3092153561200096e-04,  4.1196735273454654e-05,
   7.8927196185929473e-06,  3.7975761762202758e-07,
   4.3931491299071884e-05,  1.0000000000000000e+00]]
achieved_goals[0] = [[1.2104978245640396 , 0.6039071325589241 , 0.42470209084782223]]
desired_goals[0] = [[1.442186839143263 , 0.8727755150836819, 0.424699749459536 ]]
actions[0] = [[-0.99875706, -0.9999027 ,  0.999238  ,  0.8369149 ]]
rewards[0] = [-1.]
episodic_return = [-14.]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

### FetchSlide-v2

10 episodes、500 steps。Tのmin/mean/max=50/50/50。state/action/reward要素数 per step=26/4/1。

`s[0:3]`=gripper xyz、`s[3:6]`=object xyz、`s[6:9]`=object-gripper相対xyz、`s[9:11]`=左右finger変位、`s[11:14]`=object XYZ Euler角、`s[14:17]`=objectのgripper相対速度xyz、`s[17:20]`=object角速度xyz、`s[20:23]`=gripper速度xyz、`s[23:25]`=finger速度2値。これは `task.py` の25次元説明との対応。`s[25]`（rawに追加された26番目）は**意味を一意に確認できず**。時刻等と決めつけない。`task.py` は23と24をともにRightと記しており、左右の厳密な割当はその説明だけでは一意に特定できない。promptでは26番目を除外する。`achieved_goals`=現在object xyz、`desired_goals`=目標object xyz。actionは4成分（xyz変位指令とgripper開閉指令）。

keyの有無: `achieved_goals`=あり、`actions`=あり、`agent_dirs`=なし、`desired_goals`=あり、`dir_vectors`=なし、`episodic_return`=あり、`rewards`=あり、`states`=あり。

実例: `offline_data/robotics_data/raw_transitions/FetchSlide-v2/episodes/episode_0_seed1.npz`（15346 bytes、T=50）。

| key | type | dtype | episode shape | ndim | 要素数 |
| --- | --- | --- | --- | --- | --- |
| achieved_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| actions | numpy.ndarray | float32 | [50, 1, 4] | 3 | 200 |
| desired_goals | numpy.ndarray | float64 | [50, 1, 3] | 3 | 150 |
| episodic_return | numpy.ndarray | float64 | [1] | 1 | 1 |
| rewards | numpy.ndarray | float32 | [50, 1] | 2 | 50 |
| states | numpy.ndarray | float64 | [50, 1, 26] | 3 | 1300 |

```text
states[0] = [[ 9.9605634876311655e-01,  7.4890838715104890e-01,
   4.1266657189005884e-01,  9.2495175162347587e-01,
   8.3863575673771940e-01,  4.1402256159723194e-01,
  -7.1104597139640680e-02,  8.9727369586670491e-02,
   1.3559897071730997e-03, -2.0356242308826478e-06,
   1.4715897669077866e-03, -5.3296426859128143e-03,
   1.2454865440853600e-04, -1.8820223759885339e-02,
   1.2269826076120058e-03, -5.5673693368128687e-05,
   6.4386420383108549e-05,  1.8645099333865673e-02,
  -4.6484533976116861e-04, -4.8303613080154482e-03,
  -1.2696965563917134e-03,  5.0383869585670996e-05,
   7.9751134607851388e-06,  4.6619486195805660e-07,
   5.4924178905125209e-05,  1.0000000000000000e+00]]
achieved_goals[0] = [[0.9249517516234759 , 0.8386357567377194 , 0.41402256159723194]]
desired_goals[0] = [[1.2832187002858404 , 0.7029017366938161 , 0.41401894352053975]]
actions[0] = [[-0.991397  ,  0.99117804,  0.914531  ,  0.08079183]]
rewards[0] = [-1.]
episodic_return = [-34.]
```

Tを除くschema差異filename: なし。全episodeのexact shape/filenameは機械可読inventoryを参照。

## Rawとpromptの違い

`llm_x/data.py` の `_array_text` と `questions.py` の `_text` は `np.array2string(..., precision=4, separator=", ")`。4桁固定小数ではなくNumPyの数値表示設定であり、dtype指定そのものは送られない。ブラケットの階層は維持する。indexed historyは `Step t:` に state/action/reward を続け、step間は空行。unindexedではslice全体を `states: / actions: / rewards:` として表示。linewidth/thresholdは明示されていないのでNumPy既定値の改行や `...` 省略が起こり得る。

Fetchだけraw26次元の最後を除き25次元を文字列化する。optional fieldsと `episodic_return` はhistoryへ追加しない。`next-state` 質問内のrewardは `Episode.reward()` がscalarをPython floatへ変換するので、history中のarray表記とは異なる。質問によって現在state、次state、action等を挿入する。

最終的な送信対象はtask説明から作るsystem文字列と、history＋質問から作るuser文字列でありNPZ自体ではない。詳しい実装経路、history範囲、現在の対応taskは[workspace README](../README.md)を参照。
