# Reward図

既存state/action図と同じNPZデータを使用。既存summary.jsonに記録された全episodeのSHA256と照合済み。

- 時系列：既存state/action時系列と同じ各Taskの先頭episode（episode_0）、0始まりtimestepに対する保存済みreward。累積報酬ではない。
- 分布：既存state分布と同じ各Taskの全10 episodeをpoolしたrewardのヒストグラム。縦軸はtimestep数。値が一定の場合は1本の棒で表示。
- 出力：figures/MountainCar_reward.png、MountainCar_reward_distribution.png、Pendulum_reward.png、Pendulum_reward_distribution.png。
- reward_summary.jsonに件数・範囲・ヒストグラム・出典SHAを保存。

run.pyの通常実行でもreward図を再生成する。今回の追加時は既存図・集計を変更していない。
