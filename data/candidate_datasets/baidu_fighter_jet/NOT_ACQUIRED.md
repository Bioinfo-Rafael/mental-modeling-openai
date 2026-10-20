# NOT_ACQUIRED

exact dataset URL/IDを一意に特定できないため未取得を維持。似た名称のdatasetで代替せずschemaも推測していない。

[論文](https://link.springer.com/article/10.1007/s42452-026-08398-3) はBaidu AI Studioの検索URLと “Fighter jet” 検索を案内するが、固定dataset page/IDを提示しない。前回調査では公開データをCC0 1.0のairborne trajectory datasetと記述する一方、独自収集データは著者へのrequestが必要とされていた。実fileのlicense/schemaは未確認。

確認した候補導線は [AI Studioの「飞机轨迹」検索](https://aistudio.baidu.com/global/search?keyword=%E9%A3%9E%E6%9C%BA%E8%BD%A8%E8%BF%B9&tab=DATASET) と論文中 “Fighter jet” 表記。JavaScript検索結果から引用対象と一致する固定IDを確定できなかった。取得可能と確認した別candidateは存在しない。今回は再取得・ログイン・著者への連絡をしていない。

手動確認する場合:

1. 上記URLをbrowserで開き、Datasetタブで “Fighter jet” と “飞机轨迹” を確認する。
2. 論文と一致する作者・説明・公開日・licenseを確認し、固定dataset page URLとIDを記録する。
3. 複数候補が残れば著者確認等で解決し、推測でdownloadしない。
4. URL/IDと利用条件の確定後、ユーザーが取得を別途指示する。

今回の結論: format、size、columns、trajectory数、state/action/reward、互換性はいずれも未確認。A/B/Cは判定保留。
