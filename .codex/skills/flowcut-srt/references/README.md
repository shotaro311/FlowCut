# flowcut-srt references

このSkillは、FlowCutの処理フロー（Whisper → LLM Pass → SRT）を「Skill内スクリプトだけ」で再現します。

- 実装方針: `scripts/make_srt.py` に集約（FlowCut本体には import しない）
- LLM: `gemini` CLI を subprocess で呼び出し（JSON抽出は生ログ混入を許容）
- 出力: `output/<stem>_<timestamp>/` に `.srt` と `logs/` を固定で保存

## 設定（環境変数 / 任意）
- Whisper
  - `WHISPER_BACKEND`（`auto` / `mlx` / `faster` / `openai`）
  - `WHISPER_MODEL`（例: `large-v3` / `small`）
    - `mlx` の場合、`large-v3` → `mlx-community/whisper-large-v3-mlx` のように自動でHF repo名へ解決します
  - `FASTER_WHISPER_DEVICE`（`faster` のみ、例: `cpu` / `cuda`。デフォルト `cpu`）
  - `FASTER_WHISPER_COMPUTE_TYPE`（`faster` のみ、例: `int8` / `float16`。デフォルト `int8`）
- LLM（Gemini CLI に渡す model 名）
  - `LLM_PASS1_MODEL`, `LLM_PASS2_MODEL`, `LLM_PASS3_MODEL`, `LLM_PASS4_MODEL`
  - `LLM_REQUEST_TIMEOUT`（秒）
- 出力品質
  - `FLOWCUT_START_DELAY`（秒、デフォルト `0.2`）
  - `FLOWCUT_LINE_MAX_CHARS`（デフォルト `17`）
