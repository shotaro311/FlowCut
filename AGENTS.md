# Repository Guidelines

## プロジェクト構成とモジュール整理
- 主要ドキュメントは `docs/` にあり、要件定義と日次計画 (`docs/plan/`) をここで管理します。
- `.env.example` を基にローカル設定を `.env.local` へ複製し、機密値はコミットしないでください。
- 今後の実装コードは `src/`、共有ライブラリは `packages/`、アセットは `assets/` に配置する想定です。新規ディレクトリを作る際は README を添えて目的を明記します。

## ビルド・テスト・開発コマンド
- 依存導入: `npm install`（pnpm / yarn を使う場合は `pnpm install --frozen-lockfile` 等に読み替え）。
- 静的解析: `npm run lint` は ESLint を、`npm run type-check` は TypeScript の厳格モードを実行します。
- ビルド: `npm run build` で本番用成果物を `dist/` に生成し、PR前に必ず成功を確認してください。

## コーディングスタイルと命名
- TypeScript/JavaScript は2スペースインデント、シングルクオート、セミコロン必須で統一します。
- ファイル名・ブランチ名はケバブケース（例: `feat/refactor-auth`）。Reactコンポーネントはパスカルケース、ユーティリティはキャメルケース。
- Prettier 設定はリポジトリ直下の `.prettierrc`（追加予定）に従い、自動整形後にコミットします。

## テスト指針
- ユニットテストは Jest、E2E は Playwright の導入を前提に `tests/` 配下へ配置します。
- テストファイル名は `<対象>.spec.ts`。新規ロジックは最低1つの失敗パスと成功パスをカバーし、100%に満たない場合は理由をPR本文へ記載します。
- `npm run test -- --watch=false` をCIと同等設定で実行し、スナップショット更新時は差分を確認してください。

## コミットとプルリクエスト
- `chore: bootstrap refactor auth` に見られるように Conventional Commits（type: summary）を踏襲し、Issue番号は `feat: add auth flow (#12)` のように末尾へ添えます。
- PRは draft で作成し、テンプレートの Summary / Checklist / How to test / Notes をすべて埋めます。必要に応じて `gh pr view --json statusCheckRollup` でCI結果を添付。
- スクリーンショットやプレビューURL（Vercel など）がある場合は Notes にまとめ、ロールバック手順も同セクションで更新します。

## セキュリティと設定
- `.env.local` と同等の秘密情報は必ず `git update-index --skip-worktree` で除外し、共有は 1Password などの管理ツール経由で行います。
- 依存アップデートは `npm outdated` で確認し、重大CVSSが出た場合はHotfixブランチ（例: `fix/dependency-cve-2025`）で即時対応します。
- Pull Request 作成後は `gh pr status` でCIが完了するまでモニタリングし、失敗時はログを貼って原因と次アクションをコメントしてください。
