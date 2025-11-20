# LLM モジュール

LLM整形やプロンプト管理に関するソースを配置します。

- `formatter.py`: LLMプロバイダー共通インターフェースと `[WORD: ]` 付き字幕行の構造化を担当。
- `prompts.py`: SRT向けの指示テンプレートとメッセージ化ユーティリティ。

Phase 2 以降は各クラウドAPI（OpenAI / Google / Anthropic）のクライアントを `formatter.register_provider()` で登録し、`LLMFormatter` 経由で再利用します。
