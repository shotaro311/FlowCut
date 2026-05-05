# FlowCut Webアプリ化・月額課金化プロジェクト設計

## 1. 結論

FlowCut の webアプリ化は、Tkinter GUI をそのまま置き換えるのではなく、既存の字幕生成パイプラインを「非同期ジョブAPI」として切り出し、その上に web UI・認証・課金を載せる形で進めるのが最も安全です。

最初のMVPでは、次の4点に絞ります。

1. ユーザー登録とログイン
2. 音声/動画アップロード
3. SRT生成ジョブの進捗確認とダウンロード
4. 月額課金の開始・解約・請求確認

理由は、現状コードの価値は UI ではなく `src/pipeline/poc.py` を中心とした音声→字幕変換ロジックにあるためです。
特に `execute_poc_run()`、`src/transcribe/**`、`src/llm/workflows/**`、`src/llm/usage_metrics.py` は web版でも再利用しやすい資産です。

## 2. 現状から見た前提

### 再利用しやすい部分

- `src/pipeline/poc.py`
  - 文字起こし、LLM整形、SRT保存、進捗保存、メトリクス出力まで一連の処理が集約されている
- `src/transcribe/base.py`
  - ランナー抽象化があり、web版でも実行エンジン差し替えがしやすい
- `src/llm/workflows/registry.py`
  - workflow 単位で字幕整形の振る舞いを切り替えられる
- `src/llm/usage_metrics.py`
  - 将来の課金原価管理に使えるメトリクス出力がすでにある

### web化の前に直すべき部分

- `src/gui/controller.py`
  - GUI前提のスレッド制御なので、そのまま web のジョブ管理には使えない
- `src/pipeline/poc.py`
  - ローカルファイル保存前提のため、ジョブID単位のワークスペース管理に置き換える必要がある
- `src/config/settings.py`
  - `.env` 依存が強いため、ユーザー単位設定とサーバー共通設定を分離したい
- `docs/requirements.md`
  - web版はまだ TODO メモ段階なので、着手前に正式な対象範囲を別セクションで固める必要がある

### いちばん大きい技術判断

サーバー版の標準文字起こしランナーは、Apple Silicon 前提の `mlx` ではなく Linux で運用しやすい構成に寄せるべきです。
現状でも `resolve_models()` はデフォルトを `openai` ランナーにしており、サーバー標準化の方向性とは整合しています。

## 3. 推奨アーキテクチャ

### 構成

- フロントエンド: Next.js
- API/BFF: Next.js のサーバー側 + 認証連携
- 音声処理API: FastAPI
- 非同期ジョブ実行: Python worker + Redis系キュー
- DB: Postgres
- ファイル保管: S3互換オブジェクトストレージ
- 課金: Stripe Billing

### 役割分担

- Next.js
  - LP、ログイン後画面、ジョブ一覧、課金画面、アップロード画面を担当
- FastAPI
  - 「ジョブ作成」「進捗取得」「成果物生成」など、既存 Python パイプラインのラッパーを担当
- Worker
  - `execute_poc_run()` をジョブ単位で実行
- Postgres
  - ユーザー、プラン、サブスクリプション、ジョブ、使用量を管理
- Object Storage
  - 元音声、抽出音声、中間JSON、SRT、ログを保存

### なぜこの構成か

- 既存のコア実装が Python なので、処理系は Python のまま残す方が品質を落としにくい
- web UI と重い音声処理を分離すると、障害切り分けとスケールがしやすい
- 課金や認証は web 側で閉じ、字幕生成はジョブとして独立させた方が責務が明確

## 4. MVPで作る画面

### 必須

1. LP
2. サインアップ / ログイン
3. ダッシュボード
4. 新規変換画面
5. ジョブ詳細画面
6. 請求・プラン管理画面

### 最初は後回しでよいもの

- ブラウザ内の動画編集UI
- チーム共有
- 複数人コラボ
- 高度な辞書マーケット
- 完全リアルタイム処理

## 5. 課金設計の推奨方針

### 最初の売り方

最初は「月額で一定分数まで使える」方式を推奨します。
理由は、初期から細かい従量課金を入れるより、料金の分かりやすさと開発の簡単さを優先した方が検証しやすいからです。

### 推奨プラン構成

- Free trial
  - 登録直後に少量の無料クレジット
- Standard
  - 月額固定 + 月間分数上限
- Pro
  - 月額固定 + より大きい分数上限 + 優先処理

### Phase 2 以降で追加

- 追加分数の買い切り
- 月間上限超過時の従量課金
- 法人向けの席数課金

### 料金判定の内部実装

課金判定は「アップロード時間」ではなく、処理対象の音声秒数を正として保存するのが安全です。
`src/llm/usage_metrics.py` の実行メトリクスと、ジョブごとの音声秒数を `usage_ledger` に記録し、請求判定はそこを唯一の正とします。

## 6. 先に決めるべき仕様

### プロダクト仕様

- 誰向けに売るか
  - 個人動画編集者
  - YouTube運用者
  - 企業の広報/採用動画担当
- 何に一番お金を払うのか
  - 文字起こし精度
  - 日本語の自然な改行
  - 作業時間短縮

### 業務仕様

- ファイル保存期間
- 再ダウンロード期限
- 失敗ジョブの再実行条件
- 1プランあたりの同時実行数
- 1ファイルの上限サイズ / 上限時間

### 法務・運用仕様

- 利用規約
- プライバシーポリシー
- 著作権とアップロード責任
- 退会時のデータ削除ルール

## 7. 段階的な進め方

### Phase 0: 事業前提の固定

目的は「何を売るか」を先に固定することです。

- ターゲット顧客を1種類に絞る
- 課金の単位を決める
- 成果指標を決める
  - 例: 初回アップロード完了率、無料登録→課金化率、1本あたり原価

### Phase 1: コア処理のサーバー分離

目的は、既存処理を GUI から外して API/worker で動かせるようにすることです。

- `execute_poc_run()` をジョブワークスペース単位で動くように整理
- ローカルパス直書きを storage abstraction に寄せる
- GUI向けコールバックをジョブイベントへ置き換える
- 1ジョブ = 1入力 = 1出力 の基本契約を固定

### Phase 2: Backend MVP

- 認証後にジョブを作れる API
- 署名付きURLでアップロード
- キュー投入
- ジョブ状態管理
  - `queued`
  - `processing`
  - `succeeded`
  - `failed`
- SRT ダウンロード
- 使用量記録
- Stripe webhook 連携

### Phase 3: Frontend MVP

- LP
- ログイン
- アップロード
- 進捗表示
- 完了後ダウンロード
- 請求画面
- 利用上限到達時の制御

### Phase 4: 運用強化

- 再試行制御
- 同時実行制限
- 管理者向けジョブ監視
- 原価可視化
- 障害通知
- 保存期限による自動削除

### Phase 5: 商品強化

- 用語辞書のユーザー保存
- テンプレート化
- チームプラン
- API提供

## 8. 最初の8週間でやること

### Week 1-2

- web版の正式 requirements を作る
- DBスキーマを決める
- ジョブ状態の定義を決める
- サーバー標準ランナーを決める

### Week 3-4

- FastAPI でジョブAPIの骨組みを作る
- object storage 連携
- queue/worker 連携
- 単体ジョブで SRT 生成完了まで通す

### Week 5-6

- Next.js ダッシュボード実装
- ログイン実装
- アップロードから完了確認まで接続

### Week 7

- Stripe Checkout / Billing / webhook 連携
- プラン制御
- 無料枠制御

### Week 8

- E2E検証
- 原価試算
- LP公開
- 最小課金テスト開始

## 9. データモデルの最小案

- `users`
- `subscriptions`
- `plans`
- `jobs`
- `job_artifacts`
- `usage_ledger`
- `glossaries`

### `jobs` に最低限必要な項目

- `id`
- `user_id`
- `status`
- `source_file_path`
- `result_srt_path`
- `audio_duration_sec`
- `transcribe_runner`
- `llm_provider`
- `workflow`
- `error_code`
- `error_message`
- `created_at`
- `started_at`
- `finished_at`

## 10. 技術リスク

### 1. 原価が読みにくい

音声長、Whisper実行時間、LLMトークンで原価がぶれます。
そのため、MVPの時点で「ジョブ原価の見える化」を必須にします。

### 2. 大きい動画アップロードが不安定

大容量ファイルはアプリサーバー経由ではなく、直接 object storage へ送る構成にします。

### 3. サーバーでの音声処理負荷が重い

同期HTTPで処理せず、必ず非同期ジョブにします。
同時実行数もプランや内部上限で制御します。

### 4. 個人情報・機密音声の取り扱い

保存期間、削除ルール、ログのマスキング方針を先に決めないと、法人向け販売に進みにくいです。

## 11. 今回の推奨意思決定

現時点では、次の判断で進めるのを推奨します。

1. web版MVPは「字幕生成SaaS」に絞り、動画編集機能は入れない
2. 既存 Python パイプラインを再利用し、web UI は別実装にする
3. 文字起こしはサーバー向けランナーへ標準化する
4. 課金はまず月額固定 + 月間上限で始める
5. 認証・課金・保存・ジョブ管理を先に固め、その後に機能拡張する

## 12. 参考にした一次情報

### リポジトリ内

- `docs/requirements.md`
- `src/pipeline/poc.py`
- `src/gui/controller.py`
- `src/transcribe/base.py`
- `src/config/settings.py`
- `src/llm/workflows/registry.py`
- `src/llm/usage_metrics.py`

### 外部ドキュメント

- Stripe Billing overview: https://docs.stripe.com/billing/subscriptions/tiers
- Stripe Customer Portal: https://docs.stripe.com/no-code/customer-portal
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
- Next.js routing docs: https://nextjs.org/docs/14/app/building-your-application/routing/defining-routes
