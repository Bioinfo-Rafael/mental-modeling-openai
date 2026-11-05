# PROGRESS

> Auto-maintained by /checkpointing. Shows the most recent 5 checkpoints (newest first).
> Full checkpoints live in `.claude/checkpoints/` (git-ignored).

## [2026-11-04-105709](.claude/checkpoints/2026-11-04-105709.md)

## 何をしたのか

- LAG 向けに、2 層の LLM Translator と LLM-as-a-judge の評価基盤を設計・実装し、3 本の PR として main にマージした。
  - PR #3：Translator と judge、ローカル vLLM 部品、動画フレーム、3 時点 × 4 観点の評価機能。
  - PR #5：1〜5 の採点基準（アンカー）。
- 2 層の Translator の中身。
  - 検出層は clef-flash（System One、ローカル vLLM・UDS）。
  - 翻訳層は gpt-6.1-terra で、出力は日本語。
  - 契約は v2 で、検出器が開始点・変化点を 1 つの choice 質問で判定する。
  - 指示・警報・質問文は t 以前のものだけを入力に入れる。
- LLM 呼び出し部品（`src/mtd_rl_xavier/llm/`）。
  - OpenAI Responses。
  - System One。
  - vllm_chat と vllm_server（起動部品）、`scripts/translator_vllm.sbatch`。
- 要点採点 judge（gpt-6.1-sol、D29）。
  - YAML で 6 入力を選ぶ。
  - 一部だけの採点を許す。
  - 動画フレームを入力にできる。
- 3 時点 × 4 観点の評価（`evaluation/assess_*.py`）。
  - 未来情報の混入を呼び出しごとに検査する。
  - 機械判定、Judge パケット（FUTURE ブロックつき）、人の評価の取り込み、レポートを持つ。
  - 機械・人・Judge の結果は別ファイルに追記のみで保存する。
  - 未決の値は null にし、その場合は not_evaluated として理由を残す。
- 24 の評価基準すべてに、観察できる条件で 1〜5 の基準文を付けた（rubric pilot-2）。
  - 採点は「当てはまる最も低い段階」とし、「なぜ n+1 でないか」を根拠に書かせる。
  - 要点採点 judge の総合 1〜5 も、要点スコアにひもづけた。
- 評価基盤のシステム図を Artifact として公開した（https://claude.ai/artifact/5cFqxc56Xw86JF4q1S194Z）。
- 作業用の worktree 2 つと feature ブランチ 3 本は、マージ後に削除した。

## どういうやり取りをユーザーと行ったのか

- 最初の依頼は judge のみだった。そこへ「そもそも Translator がない」と分かり、Translator を先に定義する案 A をユーザーが選んだ。
- 7 回以上の聞き取りで、D1〜D68 を決めた（`.claude/docs/research/translator-requirements.md`）。
- 主な決定。
  - API 呼び出しの部品とローカル vLLM の部品を分ける。
  - 検出器は OpenJev から clef-flash に変更した。
  - 出力は日本語にする。
  - 事後モードはアブレーションとして残す。
  - 「最大限シンプル」に作る。
  - Translator のきっかけを新しい定義に合わせる（D50）。
  - 尺度は試行版の 1〜5 と判定不能にし、総合点は作らない。
  - 応答期限 3 秒、質問間隔 10 秒。
  - 簡素化は 2928 行で受け入れる。
- レビュー指摘の修正範囲、UDS を既定にすること、OPENAI_LOG を拒否すること、コミットと PR の方法は、ユーザーがそれぞれ承認した。
- 1〜5 の基準について、ユーザーは「sol（Codex）にも聞いて」「fable にも聞いて（叩き台で良い）」「web 検索してよい」と指示した。

## どうやったのか

- /feature（MODE=existing）で進めた。
  - Codex（sol）に、規模判定・設計・計画・検証の 4 種を毎回相談した。
  - 途中で NEEDS_REVISION が出たら、直して再検証し、PASS してから承認ゲートに出した。
- 実装は /team-execute で、担当範囲を事前にチェックしてから並列で進めた（P0〜P4、L1・L3、T1・E1・E2）。
- 担当の報告は信用せず、こちらで検証を再実行した。
  - テスト、lint、format。
  - verify_delegation、所有範囲の照合。
  - assert 数の比較、作りかけ箇所の検査。
- レビューはセキュリティ・品質・テストの 3 観点で並列に行い、指摘を修正した。
  - 送信先の固定。
  - プロンプト注入の囲い込み。
  - JSONL の分割。
  - 中断時に uncertain を保持すること。
- コミットはユーザーのブランチを切り替えずに作った。
  - 最初は plumbing で作った。
  - 以後は `../mtd-rl-xavier-assess` と `../mtd-rl-xavier-rubric` の worktree で作った。
- 1〜5 の基準は、fable の叩き台、web 調査、sol の設計・レビュー・差分確認を突き合わせて決めた。

## 途中でどういう課題が起こったのか

- OpenJev-4B は vLLM で動かないため、clef-flash（9B）に切り替えた。
- 全体ゲートの verify.sh は、以前からの理由で fail のまま。
  - 既存の ruff 違反 99 件、ty が入っていない、外部データ欠落による失敗 69 件。
  - そのため、新しいパスに絞ったチェックで代用した。
- ユーザーが weightreading を並行して進めていたので、ブランチ切り替えとコミットの混入を避ける必要があった。
- L1 の担当が、担当外の既存ファイル 39 件を誤ってフォーマットした。HEAD から復元済み。
- 評価のコードが見積もりの 2.5 倍（3.2k 行）になった。簡素化で 2928 行まで減らし、ユーザーが受け入れた。
- 別名チェックが部分文字列で照合していたため、window や close まで弾いていた。単語単位に修正した。
- 一部の Codex 呼び出しが 600 秒でタイムアウトした。システムの python3 が 3.10 のため、ラッパーは `.venv/bin/python` で実行した。

## 将来のアクション

- 基準文（`.claude/docs/research/judge-score-anchors.md`）の文言をユーザーが確認する。特に「一つに特定できる」と、判断への寄与の「必要な三点／四点」。
- 実行時の承認（D49）を得てから、次を行う。
  - vLLM 環境の作成。
  - clef-flash のダウンロード（約 19 GB）。
  - GPU での確認ジョブ。
  - 検出閾値 0.65 の較正。
- 本物の注釈と人の評価データを用意し、試行を行う。
  - 観点ごとに約 50 件。
  - 重み付き κ を見て、「なぜ n+1 でないか」の扱い、呼び出しの単位、尺度の細かさを見直す。
- 未決の項目を決める。
  - 戦術の語彙、注釈する単位、評価者の人数。
  - 総合点の作り方、Judge を人の代わりに使ってよい一致度の基準。
- 残っている課題。
  - 事後モード（L2）の実装。
  - ジョブごとのソケットパス。
  - openai_responses の例外連鎖。
  - trajectory.text の JSONL 厳密検証。
- メインのチェックアウトに、古い未追跡のコピーが残っている。対象は `src/mtd_rl_xavier/{llm,lag,translator}`、`evaluation/`、`tests/{llm,lag,translator,evaluation}` など。main を取り込む前に削除が必要。

## [2026-11-04-090732](.claude/checkpoints/2026-11-04-090732.md)

## 何をしたのか

- 意図翻訳の比較（A：トラジェクトリ版 Translator、B：重み＋中間状態＋観測）を日本語化しました。出力文もビューアの UI も日本語です（run `seed100601-ja`）。
- 回答の長さに上限を設け、A と B で条件をそろえました（run `seed100601-ja-short`）。上限は意図 200 字以内、根拠 4 件以内、根拠 1 件 80 字以内です。結果は両手法とも 8/8 件、スキップ 0 件でした。意図の長さは A が 47〜90 字、B が 79〜122 字です。費用は A が約 $0.76、B が約 $0.60 でした。
- Translator が PR #3 で main に入ったので、保留していた method A の実行部分も含めました。ブランチ `feature/intent-compare-ja-method-a` に 2 コミット（8acdf0d、bfb5134）を push し、PR #4 を作成しました。

## どういうやり取りをユーザーと行ったのか

- 「ビューワー見たけど，英語になってない？」→ 日本語で再実行しました。「いや，全部日本語よ．」の指示で、UI も日本語にしました。
- 「翻訳はもう少し短くてもいいかも」→ 上限は「意図 200 字、根拠 4 件」を選んでもらいました。「条件は揃えたい」の指示で、A と B の上限の文言と違反時の扱いを同一にしました。
- 有料ジョブ（A 5001646/5001941、B 5001647/5001942）はユーザーが自分のシェルから投入しました。
- 「ここまでの作業を push までして．PR もして」→ 新ブランチで PR #4 を作成しました。

## どうやったのか

- 比較設定に任意キー `output_language`、`max_intent_chars`、`max_evidence_items`、`max_evidence_chars` を追加しました。既定値では従来の動作が変わりません。
- 上限を伝える文と違反の判定は `intent_protocol`（`limits_sentence`、`payload_problem`）にまとめ、A もこれを再利用しています。
- ビューアの文言は `scripts/intent_compare_labels.py` に移しました。
- 他セッションの作業ツリーを汚さないよう、origin/main から scratchpad 上に git worktree を切りました。自分のファイルだけをそこへコピーし、テストとコミットを行っています。

## 途中でどういう課題が起こったのか

- A が上限超過の回答で異常終了していました。B と同じく、スキップして続行するように変えました。
- main の Translator は prompt v2 で、`TranslationModelRequest` に `alarms` と `question` が追加されていました。A の `build_request` にはこの 2 つの指定を足しました。また main の `run.yaml` は `max_intent_chars` を 600 に変えていたので、上限が translator 設定と一致することを前提にした古いテストを直しました。
- 全テストで 29 件の失敗と 42 件のエラーが出ました。変更を外した origin/main でも同じ数が出ており、worktree にローカルデータがないためです。
- B では HTTP 500 が毎回 3 回ほど出ましたが、自動再試行ですべて回復しました。

## 将来のアクション

- PR #4 のレビューとマージ。
- main の Translator（prompt v2）で A を再実行して確認する。
- B のミサイル脅威の読み（obs[27:33]）を ACMI のミサイル航跡と突き合わせる。
- `human_verdict` を記入する。
- ルートの `.gitignore` に `outputs/intent_compare/` を追加する。
- seed やシナリオを増やす。
- 元の作業ツリー（ブランチ feature/weightreading）にはコミット前の同じファイルがまだ残っています。PR #4 のマージ後に整理が必要です。

## [2026-11-04-045406](.claude/checkpoints/2026-11-04-045406.md)

### 何をしたのか

- RL ポリシーの重みを LLM に直接読ませて解釈させる weightreading を `src/weightreading/` に新しく作った（参考：Sparks of Weight-Space Programming の Exp 3）。
  - LLM は重み `W` に numpy を当てて調べられるが、自分で入力を作って forward pass を回すことは禁止。
  - 最終回答には、機械可読な claims（入力 → ヘッド、符号、確信度）を含める。
- vendored の `mentalmodeling/` を `src/mentalmodeling/` に移した。既存の結果と比較できる状態は保っている。
- MHI の 2v2 MAPPO actor（v1.3、v1.3-1）に対応した。
  - claims は、numpy float64 の参照 forward pass で実測と照合する（`evaluate`）。torch との突き合わせは全 4 ネットワークで通過した。
  - 正答率は baseline 79%、v1.3 80%、v1.3-1 83%。符号を逆に読んだ claims は 0 件。
- mtd-rl-lag で、MHI v1.3 の 2v2 ロールアウトを計算ノードで記録した（シード 100600〜100603）。numpy で再生すると、行動は 100% 一致した。
- 意図翻訳を 2 つの手法で比べた。対象はシード 100601 のステップ 45/145/245/315、A0100 と A0200 の各機、モデルは gpt-6.1-sol。
  - A：トラジェクトリから翻訳する既存の Translator
  - B：重み、中間状態、観測から翻訳する新しい weightreading の intent モード
- 翻訳時刻ごとに前後 30 秒の動画を作り、A と B を並べた HTML ビューアを作った。
- OpenAI の 5xx に対して、記録を残しながら自動で再試行する仕組みを入れた。
- PR #2（base main、`feature/weightreading`、コミット 10 件）を作り、本文を更新した。

### どういうやり取りをユーザーと行ったのか

- **weightreading の方針**：汎用に作り、まず低レベル制御器で検証する。ブログに準拠して LLM に重みを直接読ませる。置き場所は src。backend は OpenAI Responses API。課題説明は既定で見せ、設定で伏せられる。
- **作業環境**：ログインノードには uv だけを入れ、重い処理は計算ノード（Slurm `151-partition`、`-n 1`）で動かす。
- **有料 API**：ユーザーが承認した。ジョブは、ユーザーが自分のシェルで `.env` を読み込んでから投入する。Claude は `.env` に触れない。
- **MHI のモデル**：`mtd-rl-lag/models` に置いてもらい、mtd-rl-lag 側では ignore した。
- **重みだけでなく中間状態も読むべきでは、という問い**：ユーザーは、重みと中間状態の両方を渡す方式を選んだ。そのうえで、トラジェクトリによる意図翻訳と比べ、動画付きで見たいという要望が出た。
- **比較の条件**：人手の注釈と同じ時刻にする。B には重み、中間状態、観測を渡す。翻訳は機体ごと。モデルは「全部 gpt-6.1-sol」。
- **B のサーバーエラー**：B が HTTP 500 で 2 回止まった。ユーザーは自動再試行を入れることを選んだ。
- **コミット**：「適切に分割したい」という指示で 4 コミットに分けた。A の部分は、別セッションの translator に依存するため保留にした。最後にチェックポイントを依頼された。

### どうやったのか

- **作業の分担**：調査、設計、実装は、general-purpose-opus と general-purpose-sonnet のサブエージェントに任せた。設計と再試行の処理は、Codex に read-only でレビューさせた。メインのエージェントは検証とコミットを担当した。
- **安全性**：
  - 重みの読み込みは stdlib だけの allow-list unpickler で行い、torch に依存しない。
  - サンドボックスは `unshare` で作り、fail-closed の preflight を通す。
  - 有料呼び出しは `--execute --confirm-paid-api` を付けたときだけで、sbatch では `INTENT_COMPARE_CONFIRM_PAID=yes` も必要。
- **記録の取り方**：torch に読み取り専用の hook を付けた。記録しない実行とビット単位で一致することを確かめ、記録がロールアウトに影響しないことを確認した。
- **動画**：ACMI 付きで再実行し、mtd-rl-lag の `acmi_to_mp4.py` で描いた。ffmpeg は imageio-ffmpeg に同梱のものを使い、クリップは ffmpeg で切り出した。
- **B の中間状態**：h=0 からの numpy 再生で作り、記録との一致（差が 1e-4 以下、行動が全一致）を満たさなければ実行を拒否する。
- **コミットの検証**：コミットごとに、stage した内容だけを書き出して pytest と ruff を回した。部分的な hunk は `git update-index --cacheinfo` で stage した。

### 途中でどういう課題が起こったのか

- `codex_consult.py` が Python 3.10 で ImportError になった（`datetime.UTC`）。uv の Python 3.12 で回避した。
- 最初の有料実行では、LLM がコードを出さずに返したため 0 runs で終わった。プロトコルを直し、整った最終回答のときだけ終了するようにした。
- 同じ spec で checkpoint が違うと、出力ディレクトリが衝突した。ディレクトリ名に sha12 を付けて解決した。
- B の有料実行が、OpenAI の HTTP 500 で 2 回止まった（ジョブ 5000680、5000719）。止まったリクエストの消費トークンは 0 だった。自動再試行を入れて `RESUME=yes` で再開し、ジョブ 5000952 では 500 が 4 回出たが回復した。500 になったリクエストは約 3 分待たされていて、これが課金されたかどうかは確認していない。
- サブエージェントが一度、実行中のプレビュー用ジョブ（5000428）の出力ディレクトリを消してしまった。作り直したジョブ 5000431 は通った。
- A の実行部分は、別セッションの未コミットの translator、lag、llm を import している。そのため A の部分は保留にし、コミットに含めなかった。
- ルートの `.gitignore` は、`outputs/intent_compare/` の mp4、html、npz を除外していない。ネストした `.gitignore` で一時的に除外している。
- 作業ツリーに、別セッションの未コミットの変更が多くある（pyproject.toml、uv.lock、.gitignore、DESIGN.md、STATE.md、translator 一式）。どれにも触れていない。

### 将来のアクション

- 保留にした A の部分をコミットする。対象は `scripts/intent_compare_{convert,a_requests,run_a}.py`、`run_a.sbatch`、`tests/intent_compare/test_{convert,a_requests}.py`、video.sbatch の変換手順で、別セッションの translator がコミットされてからにする。
- B が読んだ「ミサイルの脅威」（obs[27:33]）を、記録したミサイルの軌跡と突き合わせる。
- B の 8 セッションと、weightreading の 3 回の実行について、`audit.json` の `human_verdict` を人が確認して埋める。
- ルートの `.gitignore` に `outputs/intent_compare/` のルールを足す。別セッションの変更と調整してから行う。
- 記録を増やす（シード 20 本、BeamBreak シナリオ、v1.3-1）。blind モードも試す。
- PR #2 のレビューに対応する。`poe test` と `poe test-workspace` は、コミット 4〜7 のあとまだ回していない。
