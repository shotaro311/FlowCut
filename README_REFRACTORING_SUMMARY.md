# リファクタリングまとめ（2025-11-23 18:50 JST）

- LLM整形フローを two-pass 固定に統一（旧1パス + [WORD:]アンカー + RapidFuzzアラインを撤去）。
- CLIオプションを簡素化：`--llm-two-pass` / `--align-*` を廃止、two-passのみ利用。
- LLMリクエストタイムアウトをデフォルト 500 秒に延長（`LLM_REQUEST_TIMEOUT` / `--llm-timeout` で上書き可）。
- ドキュメント更新：`docs/requirement.md`・`docs/specs/llm_two_pass_workflow.md`・`docs/runbook.md` を two-pass 前提に同期。
- 依存整理：rapidfuzz を削除。アライン関連テストを整理し two-pass 用テストを通過。

今後の優先リファクタリング候補（最新進捗考慮）
1. README / リリースノートに two-pass固定と旧オプション廃止を明記（利用者周知）。
2. 実サンプルでのワンショットE2E確認（必要なら）と SRT 実出力の最終確認。
3. kotoba/mlx ネイティブの安定化と CLI ヘルプの簡素化（残件があれば）。
