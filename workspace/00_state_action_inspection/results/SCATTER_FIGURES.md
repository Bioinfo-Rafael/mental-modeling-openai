# 時系列の散布図

既存の折れ線・階段状の時系列図と同じ各Taskのepisode_0を使用。summary.jsonに保存された出典SHA256と一致することを確認済み。

横軸は0始まりのtimestep、縦軸は保存済みのstate・action・reward。点を線で結ばず、間引き・平滑化・jitterはしない。stateは次元ごとにパネルを分ける。

figures/にMountainCarとPendulumそれぞれのstate_scatter.png、action_scatter.png、reward_scatter.pngを保存。既存図は保持。run.pyの通常実行でも再生成する。
