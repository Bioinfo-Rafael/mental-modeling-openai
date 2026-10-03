# Joint解析結果（seed=42、bootstrap=1,000）

対象は取得済み960回答、32条件×30件。新規API送信なし。
解析定義と全図・CSV一覧は [README.md](README.md)。以下は今回実行した結果のスナップショット。

## Parse結果

各行の母数はH=5/10/20/30を合計した120回答。`—`は対象外。

| Task | Metric | action / torque | bin | direction | absolute state | state delta | 全required成功 |
|---|---|---:|---:|---:|---:|---:|---:|
| MountainCar | Next Action | 120 | — | — | — | — | 120 |
| MountainCar | Last Action | 118 | — | — | — | — | 118 |
| MountainCar | Next State | — | — | 119 | 106 | 107 | 103 |
| MountainCar | Last State | — | — | 119 | 112 | 54 | 54 |
| Pendulum | Next Action | 114 | 119 | — | — | — | 114 |
| Pendulum | Last Action | 116 | 118 | — | — | — | 116 |
| Pendulum | Next State | — | — | 113 | 104 | 97 | 92 |
| Pendulum | Last State | — | — | 113 | 111 | 102 | 98 |

全component成功は815/960件（84.90%）。少なくとも1componentが失敗した回答は145件。
component単位の失敗は198件で、1回答内の複数失敗を含むため145件とは異なる。
詳細理由は [parse_summary.csv](tables/parse_summary.csv)、回答ごとの成否は [parsed_records.csv](tables/parsed_records.csv)。

MountainCar Last Stateのdelta失敗66件のうち58件は非literal/placeholder、8件はmarkerなし。
H別のdelta成功件数は23/30、16/30、9/30、6/30。小さいsubsetでは有効1件の条件もあり、bootstrap std=0を「安定」と読んではいけない。
他のcomponentが使える回答はその指標に残し、全component成功815件だけへ一括で絞ってはいない。

## N=30のAccuracy（%、有効回答を分母）

| Task | Metric | H=5 | H=10 | H=20 | H=30 |
|---|---|---:|---:|---:|---:|
| MountainCar | Next Action | 73.33 | 66.67 | 86.67 | 83.33 |
| MountainCar | Last Action | 60.00 | 51.72 | 53.33 | 58.62 |
| MountainCar | Next State | 77.59 | 81.67 | 71.67 | 68.33 |
| MountainCar | Last State | 62.07 | 38.33 | 48.33 | 51.67 |
| Pendulum | Next Action（bin） | 16.67 | 17.24 | 10.00 | 23.33 |
| Pendulum | Last Action（bin） | 6.67 | 17.24 | 27.59 | 16.67 |
| Pendulum | Next State（direction） | 33.33 | 40.23 | 27.38 | 31.11 |
| Pendulum | Last State（direction） | 43.59 | 37.93 | 33.33 | 30.00 |

これは旧scorerのpredictionではなく、final markerから再構築した採点。
N=30は選択30件という意味で、parse失敗があればAccuracyの実際の分母は30未満になる。
Stateは有効query×全次元をpoolする。正解数・分母・失敗を不正解と数える別Accuracyは [accuracy_metrics.csv](tables/accuracy_metrics.csv)。

## 注意すべき点

- Hが長いほど一貫して改善する結果ではない。各Hのquery集合も完全同一ではなく、この値だけでHの因果効果は判断できない。
- Pendulum actionのbin正解率は6.67〜27.59%。torqueの小数値をbinと誤認した旧scorerの結果とは分けて扱う。
- Pendulumのabsolute state・ΔthetaのNRMSEは小さく見えるが、特にNext StateのΔtheta Pearsonは低い。理論幅で正規化した誤差の小ささと、変化の追従性は別である。
- componentやNによって有効集合が異なる。連続指標は必ず`valid_n`と`parse_success_rate`を併読する。
- 全long tableで定義不能な値は6行（Pearson: 有効1件が4行・定数列が1行、Cosine: zero normが1行）。0補完せず理由付き欠損で保存した。
- bootstrapはqueryの復元抽出であり、重なるtrajectory windowの依存性やparse失敗の不確実性を補正していない。

N=10⊂20⊂30を32条件すべてで確認済み。18テスト通過、実データassert・出力読戻し・保護対象hash一致を確認。
66図×PNG/SVG、12CSVを生成した。API call・commit・pushはいずれも行っていない。
