# 進捗ファイル & 再開仕様ドラフト（v0.1）

## 目的
- 長尺音声処理の途中で失敗した場合でも同じブロックから再開できるようにする。
- PoC (`scripts/poc_transcribe.py`) と将来のCLI (`src/cli/main.py`) で共通のJSONフォーマットを扱う。

## ファイル配置
- デフォルト: `temp/progress/{run_id}.json`
- `run_id = {audio_stem}_{runner_slug}_{YYYYMMDDThhmmss}`（例: `sample_audio_kotoba_20251120T173552`）

## JSONスキーマ
| フィールド | 型 | 説明 |
|-----------|----|------|
| `run_id` | string | 実行ごとのユニークID。CLI引数 `--resume` で指定予定。|
| `audio_file` | string | 入力音声のパス（絶対/相対）。|
| `model` | string | Whisperランナーのslug。|
| `llm_provider` | string/null | LLMフォーマッターのプロバイダーslug（PoCでは未使用）。|
| `total_blocks` | int | 全ブロック数。|
| `status` | enum(`pending`,`running`,`completed`,`failed`) | 実行全体の状態。|
| `created_at` / `updated_at` | ISO8601 string | 進捗レコードの生成/更新時刻。|
| `completed_blocks` | int | 完了したブロック数。|
| `blocks` | BlockProgress[] | ブロックごとの状態配列。|
| `metadata` | object | 実行環境やCLI引数など任意情報。|

### BlockProgress
| フィールド | 型 | 説明 |
|-----------|----|------|
| `index` | int | 1始まりのブロック番号。|
| `status` | enum(`pending`,`in_progress`,`completed`) | ブロック処理状態。|
| `started_at` / `completed_at` | ISO8601 string/null | ブロック着手・完了時刻。|
| `llm_payload` | object? | 追加入力予定。リトライ情報を記録。 *(v0.1では未使用)* |

## 状態遷移（ランタイム）
```
pending --(start run)--> running --> completed
                               \-> failed (例外発生)
```
各ブロック: `pending -> in_progress -> completed`（失敗時は `pending` に戻す／再開でやり直す）。

## CLI `--resume` フロー案
1. `--resume temp/progress/foo.json` が指定されたら `load_progress()` でレコードを復元。
2. `status` が `completed` の場合は即終了。`failed` or `running` の場合は再開可能と判断。
3. `blocks` を先頭から走査し、`status != completed` の最初のブロック番号を特定。
4. `--resume` 実行時は `--audio` / `--models` を再指定せずとも、ファイル内の `audio_file` `model` `llm_provider` をデフォルトに用いる。
5. 再処理ブロックは `temp/poc_samples/*.json` の既存結果を参照し、`--resume` オプションで「未完了ブロックのみ」APIへ送る。成功時は `mark_block_completed` → `save_progress()` → SRTへ追記。
6. すべてのブロックが `completed` になったら `status` を `completed` に更新し、最終SRTとCSVを出力。

## エラー時の扱い
- LLM/APIが失敗したブロックは `status=pending` に戻し、`metadata.retry_count` をインクリメント。
- CLI終了時は `status=failed` と `last_error` を `metadata` に格納（例外メッセージ/スタックトレース）。

## 今後のTODO
- `llm_payload`／`last_error` の正式スキーマ定義。
- `temp/poc_samples/*.json` との紐付け（ブロックID→ファイル名）を `metadata.blocks` に追加。
- `progress` ファイルを読み込んでSRTを部分的に作り直すツールの検討。
