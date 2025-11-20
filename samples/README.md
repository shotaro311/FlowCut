# samples データポリシー

- 音声ファイルそのものは著作権の都合でGitに含めません。必要な場合は共有ストレージのURLと `sha256` をここに追記してください。
- 命名規則: `YYYYMMDD_<topic>_<length>.wav`。例: `20251120_news_anchor_5m.wav`。
- 付随情報（書き起こし・ノート）は `samples/meta/` などのサブフォルダにMarkdownで保存します。
- `temp/poc_samples/` に生成されるモデルごとのJSONはキャッシュ扱いのため、必要に応じて `.gitignore` に追加してください。

## 共有手順（Google Drive想定）

1. 音声ファイルをGoogle Driveにアップロードし、「リンクを知っている全員に閲覧権限」を付与してください。
2. ローカルで `shasum -a 256 <file.wav>` を実行し、結果をメモします。
3. 下記テンプレートを `samples/README.md` 内の「共有レコード」節に追記し、Driveリンクと `sha256` を貼り付けます。
4. 音源の利用範囲や注意点がある場合は「備考」に記入してください。
5. 共有後はSlack/Notionにリンクを貼り、再生許可の確認メッセージ（例: 「FlowCut検証用、社内利用のみ」）を添えてください。

## 共有レコード テンプレート

```
### 20251120_news_anchor_5m.wav
- 長さ: 5分12秒
- 用途: ベースライン / ノイズ少
- Google Drive: https://drive.google.com/...
- sha256: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
- 備考: 〇〇社提供素材、社外共有禁止
```

テンプレートをコピーし、ファイルごとに節を増やしてください。Driveリンクは公開範囲を「リンクを知っている全員（閲覧のみ）」に設定し、社外秘の場合は必ず備考に明記します。
