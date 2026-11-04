# tactical_short_v2：目的を中心に行動を説明する短縮条件

## 今回使う本文

- 編集用原本：`commentary_experiment/tactical_short_v2_system.txt` と `tactical_short_v2_user.txt`
- CLI：`--prompt-variant tactical_short_v2`
- 結果保存先：`results/tactical_short_batch_v2/`
- 計画生成時点の本文：同runの `prompts/templates/system.txt` と `user.txt`

`prompt_changes/` は比較資料の原本、`results/.../prompts/` はrunごとの保存コピー。比較資料・README・参照資料はモデルに渡さない。長いファイル名の `.txt` は履歴を挿入した5窓分の入力全文。

## 出力の方針

「目的／目指す近い将来の状況を実現するために、その行動を取っていると考えられる」という論理を一文内に置く。行動を述べた後に理由を別の文で付ける構成を要求しない。

- 原則1文。重要な目的の変化や解釈を左右する留保を残すために必要な場合のみ2文まで。
- 目安35〜50英単語、1機あたり合計55語以内。短くて足りれば下限なし。全期間非活動の機体は短い1文。
- 目的・主な行動・目的を支える主要な状況は残す。きっかけは、根拠があればその中に含める。
- 「生存するため」のような抽象的な目的だけで済ませず、間隔を作る・発射可能な位置関係を得るなど、目指す状況を示す。
- 重要な行動・目的の変化は短く残す。全機の変化や中心時点での変化を仮定しない。
- 役割分担・標的・指示・重視点は、根拠と必要性がある場合に目的の説明へ含める。推測を事実に変えず、不明な目的を文型のために捏造しない。
- 細かな操作履歴、数値、二次的な目的の列挙を省く。結果や別解は解釈を左右する場合だけ。

systemの分析部分と例文も目的を中心とする形に合わせた。冒頭のタスク定義と中心stepの扱いは元のまま。userはshort v1から変更しない。出力要件はsystemに集約する。4機のJSON文字列フィールド、入力61時点、5窓、3モデル、reasoning_effortは維持し、Structured Outputsは追加しない。

## 2種類のdiff

| 比較 | 保存場所 |
|---|---|
| tactical_short_batch_v1 → tactical_short_batch_v2 | `from_tactical_short_batch_v1/system.diff`、`user.diff` |
| tactical_batch_v1 → tactical_short_batch_v2 | `from_tactical_batch_v1/system.diff`、`user.diff` |

diffの `---` が比較元、`+++` がv2。各フォルダの `baseline_system.txt` / `baseline_user.txt` はその比較元であり、v2として送る本文ではない。short v1からのuser.diffは変更がないため0バイト。tactical v1からのuser.diffには、short v1で行った重複指示の整理が含まれる。

各比較元は保存済みmanifestの全5窓と照合し、本文・manifestのSHA-256と出典を `provenance.json` に記録する。作業中の現行ファイルとの差分ではない。旧runと旧プロンプトは上書きしない。

## 資料との関係と限界

資料を読んで決めた出力方針であり、短縮後の生成品質・忠実性・有用性が確認されたという意味ではない。以下のページ番号は投影版。

1. 20261001 キックオフ資料 p19：人が行動を決められる形で意図を伝えることが目的。事後説明と実際の判断根拠のずれを課題としている。目的を中心にすることは方針に合うが、目的の真偽を確定するものではない。
2. 20261022 定例資料 p17〜22：人の状況認識・判断への有用性、戦術単位、要素・正しさ・タイミング・判断への寄与を評価。p19〜20には目的から行動につなぐ説明例がある。p29の短い版は1機1文、p30は要素の相談案。
3. 20261022 議事録「意図翻訳：初期検証と出力の中身」：理由と戦術的な次の動きを重視し、1機1文程度を目安としている。
4. 20261023 先方メール【意図翻訳】P30/P31（投影版p29/p30）：短い版を希望。「今何をしているか」「なぜそうするか」は必須。理由に近い将来の理想的状況が含まれれば細かな次の動きは必須でなく、確からしさの優先度は下げてよいとしている。これを今回の内容要件として優先する。
5. 背景：20261001 キックオフ議事録のテーマ①・評価・デモ像。短くしても行動と説明の整合、理解への寄与は別途評価が必要。

メールには「時刻t以前の情報のみ」「2〜3秒程度の待ち時間」という運用条件もある。今回は比較のため `[t-30,t+30]` の前後6秒を維持するので、満たすことを狙うのは説明の内容・分量であり、オンライン運用条件全体ではない。生成もまだ実行しておらず、短くなったか・重要内容を失っていないかは未検証。

アノテーションの存在や変化点ラベルは入力に含めない。10月22日の文字起こし原文は今回も参照していない。補助手順ファイルを内容の根拠にはしていない。

## 比較実験で確認すること

- 語数と文数、特に「原則1文」が守られたか。
- 目的・目指す状況・行動の手段関係と、主要な根拠が残ったか。
- 重要な目的の変化を落としていないか。
- 短縮によって標的・命中・認知・連携を断定していないか。
- 同じ5窓でshort v1と比較する。語数制限と説明構成を同時に変えるため、差を片方だけの因果効果とは扱わない。人の理解・判断の改善は別途検証する。

## 実行

リポジトリルートでdry run：

```sh
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical_short_v2 --dry-run
```

API実行する場合：

```sh
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical_short_v2 --execute --wait
```

dry runは15件の入力を準備するだけでAPIへ送信しない。文数・語数はプロンプトによる指示であり、API側の強制ではない。既存評価では語数と55語超過を集計し、回答を切り詰めない。文数・目的と行動の関係は回答後に内容を確認する。

## 参照した原本

- [20261001_三菱重工業様_キックオフ_v1.0.pptx](</Users/cls-lab/Library/CloudStorage/GoogleDrive-mugi.suzuki@matsuo-institute.com/共有ドライブ/MTD-RL/03_定例資料/20261001_三菱重工業様_キックオフ_v1.0.pptx>)
- [20261022_第3回週次定例_v1.0.pdf](</Users/cls-lab/Library/CloudStorage/GoogleDrive-mugi.suzuki@matsuo-institute.com/共有ドライブ/MTD-RL/03_定例資料/20261022_第3回週次定例_v1.0.pdf>)
- [20261022_三菱重工業第3回週次定例.md](</Users/cls-lab/Library/CloudStorage/GoogleDrive-mugi.suzuki@matsuo-institute.com/共有ドライブ/MTD-RL/01_Vault/minutes/20261022_三菱重工業第3回週次定例.md>)
- [content.md](</Users/cls-lab/Library/CloudStorage/GoogleDrive-mugi.suzuki@matsuo-institute.com/共有ドライブ/MTD-RL/01_Vault/source/20261023_三菱重工業_第3回週次定例へのご意見/content.md>)
- [20261001_三菱重工業キックオフ.md](</Users/cls-lab/Library/CloudStorage/GoogleDrive-mugi.suzuki@matsuo-institute.com/共有ドライブ/MTD-RL/01_Vault/minutes/20261001_三菱重工業キックオフ.md>)
