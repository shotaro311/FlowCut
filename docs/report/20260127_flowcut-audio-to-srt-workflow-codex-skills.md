# リサーチレポート：FlowCut（音声/動画→テロップ/SRT）現状把握と Codex Skills への落とし込み

## 0. まず結論（要約）
- 結論（わかったこと）: FlowCut は **Python + Tkinter のデスクトップアプリ**で、音声/動画（動画は音声抽出）→ローカル Whisper で word-level タイムスタンプ付き文字起こし →（任意）LLM で字幕向けに整形（Pass1〜4 + 任意Pass5）→ `.srt` 出力、というパイプラインです。出典: `README.md`, `docs/README.md`, `src/gui/controller.py`, `src/pipeline/poc.py`, `src/llm/two_pass.py`
- 解決策（ある/ない/部分的）: 「このアプリを再現する」ことは可能です。ただし **Codex Skills は“アプリ機能そのもの”ではなく、開発手順/資材を再利用する仕組み**なので、再現は「機能単位に分けてコード生成・検証を繰り返す」形になります。出典: OpenAI公式「Skills」ドキュメント（末尾参照）
- 次にやること（最短）: 依頼どおり **1つの Skill（script-backed）で完結**させ、音声/動画パスを受け取ったら `workflow2` 相当の処理を走らせて `output/` に書き出す形に寄せるのが最短です。出典: 本レポート 4.3 / OpenAI公式「Create custom skills」

## 1. 調査対象（テーマ）
- FlowCut（このリポジトリ）の「音声からテロップ（字幕 .srt）を作成する」機能の、現状の全体像（UI/内部処理/入出力/主要ファイル）を把握する。
- その上で、現状ワークフローを細かく分割し、Codex Skills（公式仕様）にどう組み込んで再現プロジェクトへ落とすか、設計案をまとめる。

## 2. 前提 / 解釈（曖昧さがある場合）
- 仮定: 「テロップ」は `.srt` 字幕ファイルの生成を指す（README/ガイドが `.srt` 前提）。出典: `README.md`, `docs/README.md`
- 仮定: “再現”は、機能を同等に持つ別プロジェクト（または同一プロジェクト内の再実装）を、Codex Skills を用いて進めることを指す（Skillの用途は開発手順の定型化）。出典: OpenAI公式「Skills overview」
- 未確定: 再現対象のスコープ（GUI必須か、Windows配布（PyInstaller）まで含むか、LLMなし運用も必須か）。

## 3. 調査方法（どう調べたか）
- 仕様/運用ドキュメント確認: `docs/requirements.md`, `docs/README.md`, `docs/runbook.md`, `docs/plan/20251203_PLAN1.md`
- 実コード確認（処理の“現状の真実”）:
  - GUI→パイプライン呼び出し: `src/gui/controller.py`, `src/gui/workflow_panel.py`, `src/gui/config.py`, `flowcut_gui_launcher.py`
  - CLI→パイプライン呼び出し: `src/cli/main.py`
  - パイプライン本体: `src/pipeline/poc.py`
  - 文字起こし: `src/transcribe/*_runner.py`, `src/transcribe/base.py`
  - LLM整形: `src/llm/two_pass.py`, `src/llm/workflows/workflow2.py`, `src/llm/providers/*.py`
  - SRT生成: `src/alignment/srt.py`
- Codex Skills 公式仕様確認（OpenAI公式）:
  - Skills overview / Create custom skills / AGENTS.md（リンクは末尾）

## 4. 調査結果（詳細）
### 4.1 重要ポイント（現状のアプリ像）
- **アプリ形態**: Python 製デスクトップ（GUI: Tkinter / CLI: Typer）。出典: `README.md`, `src/cli/main.py`, `src/gui/app.py`
- **入力**: 音声/動画ファイル（動画は内部で音声抽出）。出典: `README.md`, `src/pipeline/poc.py`, `src/utils/audio_extractor.py`
- **文字起こし**: ローカル Whisper ランナー（GUIのデフォルトは `openai`）。複数ランナーに対応。出典: `src/gui/config.py`, `src/pipeline/poc.py`, `src/transcribe/__init__.py`
- **整形**: LLM（Google/OpenAI/Anthropic）を Pass1〜4（+任意 Pass5）で実行し、字幕向けの改行/校正/違反修正を行う。長尺は約5分で分割し最大10並列で処理して結合。出典: `src/llm/two_pass.py`, `src/llm/workflows/workflow2.py`, `src/pipeline/poc.py`
- **出力**: `.srt` と、任意で logs/（poc_samples/progress/metrics/llm_raw）をまとめて保存。出典: `docs/README.md`, `src/pipeline/poc.py`

### 4.2 根拠つき解説（段落ごとに引用）
#### 4.2.1 ユーザー操作フロー（GUI）
FlowCut の利用者は、(1) APIキーを「API設定」で保存 → (2) スロット1/2でファイルを選択 → (3) 保存先/詳細設定（モデル、開始遅延、ログ保存など）を調整 → (4) 実行 → (5) 保存先に `.srt` 生成、という流れで使います。  
出典: `docs/README.md`, `src/gui/workflow_panel.py`, `src/gui/config.py`

GUI は 2 スロットを持ちますが、内部で同時実行を避けるロックを取るため、押し方によっては待機が発生します（待機中メッセージを表示）。  
出典: `src/gui/controller.py`

#### 4.2.2 内部処理フロー（共通：GUI/CLI）
内部の主処理は `src/pipeline/poc.execute_poc_run` に集約されており、GUIはそれを薄く呼び出す設計です（GUI側で処理ロジックを再実装しない）。  
出典: `src/gui/README.md`, `src/gui/controller.py`, `src/pipeline/poc.py`

処理は大きく、(A) 入力が動画なら ffmpeg 等で音声抽出 → (B) Whisper で word-level タイムスタンプ付き文字起こし → (C) 文字起こし JSON 保存 → (D) LLM整形（任意）→ (E) `.srt` 保存 → (F) 進捗/メトリクス保存、で構成されています。  
出典: `src/pipeline/poc.py`, `src/utils/audio_extractor.py`, `src/alignment/srt.py`

#### 4.2.3 LLM整形（workflow2 / Pass1〜4 + 任意Pass5）
LLM整形の中心は `TwoPassFormatter.run()` で、Pass1（置換/削除）→ Pass2（行分割）→ Pass3（問題検出と最小修正）→ Pass4（長さ/時間幅違反の再分割）を行い、最後に単語の欠落があればフォールバックで末尾カバレッジを保証します。  
出典: `src/llm/two_pass.py`

workflow2 のプロンプトは `src/llm/workflows/workflow2.py` に定義され、自然な改行の優先ルール、最小行長（5文字以上）、引用表現の保持、行末句読点削除などが明記されています。  
出典: `src/llm/workflows/workflow2.py`

長尺は `src/pipeline/poc._run_workflow2_chunked_two_pass` が 5分目安で分割し、最大10並列で処理して結合し、ギャップ埋めと開始遅延をまとめて適用します。  
出典: `src/pipeline/poc.py`

#### 4.2.4 設定（APIキー/Glossary/モデル）
APIキーや Glossary は `~/.flowcut/config.json` に保存され、GUI 起動後の「API設定」「辞書」から編集できます。  
出典: `docs/README.md`, `src/gui/config.py`

LLM のデフォルトプロバイダーやモデル名、タイムアウトなどは環境変数（`.env`）から読み込みます（例: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `LLM_PASS1_MODEL` など）。  
出典: `src/config/settings.py`, `docs/requirements.md`

#### 4.2.5 重要な“現状差分”メモ（ドキュメントと実装のズレ）
`docs/requirements.md` では “デフォルトは mlx-whisper” と読めますが、実装上は GUI/パイプライン共にデフォルトランナーが `openai` です。一方、このSkill版（`scripts/make_srt.py`）は `WHISPER_BACKEND=auto` で mac=mlx / Windows=faster / その他=openai に自動選択するようにしています。  
出典: `docs/requirements.md`, `src/gui/config.py`, `src/pipeline/poc.py`

### 4.3 単一Skill（script-backed）で完結させる設計
「FlowCut本体を import して使う」のではなく、FlowCutの挙動（`workflow2`）に必要な最小ロジックを **Skill 内 `scripts/` に同梱**して完結させます。  
出典: OpenAI公式「Skills overview」「Create custom skills」

#### 4.3.1 実装配置（このリポジトリ内に実装済み）
- Skill: `.codex/skills/flowcut-srt/`
  - `SKILL.md`: 使い方/前提/出力を定義（自然言語の依頼 → スクリプト実行へ誘導）
  - `scripts/make_srt.py`: 実行本体（FlowCutを import せずに完結）
  - `references/`: 補足ドキュメント

#### 4.3.2 実行フロー（ユーザーのイメージに合わせた“1発実行”）
1) 入力: 音声/動画ファイルパスを受け取る  
2) 動画なら `ffmpeg` で WAV 抽出（mono 16kHz）  
3) Whisper（デフォルト: mac=mlx-whisper / Windows=faster-whisper / その他=openai-whisper）で `word_timestamps=True` の文字起こし  
4) Gemini CLI（`gemini`）を Pass1〜4 で呼び出し（FlowCutのプロンプト/ルールを同梱）  
5) `output/<stem>_<timestamp>/` に `.srt` と `logs/` を保存（固定）  

#### 4.3.3 重要な事前確認（Gemini CLIの出力揺れ対策）
Gemini CLI は、モデル応答の前に補助ログ（例: `Loaded cached credentials.`）が混ざる場合があります。  
そのため `scripts/make_srt.py` は「最初の `{` / `[` から JSON を抽出してパース」する方式にしています（FlowCut本体の JSON 抽出と同系統）。  

#### 4.3.4 Skill の置き場所（共有/個人）
- チームで共有: `<repo>/.codex/skills/`  
- 個人で使い回し: `~/.codex/skills/`  
出典: OpenAI公式「Skills overview」

## 5. 高リスク領域の注意（該当する場合のみ）
- 機密情報: APIキーは `.env` / `~/.flowcut/config.json` に入りやすいので、再現プロジェクトでも **出力/ログ/レポートに値を出さない**運用が必須です。出典: `docs/README.md`, `src/gui/config.py`
- 外部API: LLM整形はテキストを外部に送信するため、取り扱う音声内容（個人情報/機密）に注意が必要です。出典: `docs/README.md`

## 6. 限界 / 未検証 / TODO
- 未検証: このレポートはコード/ドキュメントからの把握で、実際にGUIを起動しての画面確認までは行っていません（仕様はコード優先で記載）。  
- TODO（再現に向けた確認）:
  - （運用）`gemini` の認証方式をチームで揃える（gcloud/鍵/権限）
  - （運用）長尺音声でのコスト/レート制限（並列数10が辛い場合の方針）
- Run 2の方向性案（必要なら）:
  - 出力品質が不足した場合、Passプロンプトとバリデーション（Pass3/Pass4条件）を FlowCut と突き合わせて差分調整する。

## 7. 参考文献（リンク）
- OpenAI Developer Platform: Skills overview: https://developers.openai.com/codex/skills
- OpenAI Developer Platform: Create custom skills: https://developers.openai.com/codex/skills/create-skill
- OpenAI Developer Platform: AGENTS.md: https://developers.openai.com/codex/agents-md
