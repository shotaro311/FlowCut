# タスク完了時のチェック

- `python -m pytest -q` が通ること
- 影響範囲の動作確認（少なくとも GUI/CLI の該当フロー）
- 仕様変更があれば `docs/requirement.md` を同期
- 進捗/作業ログが増えたら `docs/plan/20251203_PLAN1.md` を更新
- 配布物が必要なら PyInstaller で再生成（`FlowCut.spec` / `FlowCut_win.spec`）
