# リサーチレポート

## 0. まず結論（要約）
- 結論（わかったこと）: 「AGENTS.md などの指示文を“常に有効化する仕組み”」は **Skill だけで完全再現は難しく**、基本は AGENTS.md（グローバル/プロジェクト）で運用するのが公式の設計に沿います。一方で「指示文（AGENTS.md 群）や Skills 一覧を集めて“1つの出力（Markdown）にまとめるツール”」としては、Skill（＋スクリプト）で **再現可能**です。
- 解決策（ある/ない/部分的）: 部分的（用途を分けると実現可能）。(A) 常時適用＝AGENTS.md、(B) まとめ出力＝Skill で生成。
- 次にやること（最短）: 「再現したい“このツール”の対象」が (A) 常時適用なのか (B) まとめ出力なのかを確定し、入出力（何を集めて、どこに出すか）を 1枚で固定する。

## 1. 調査対象（テーマ）
- （仮）ユーザー提示の「AGENTS.md instructions…」のように、**共通ルール＋プロジェクト指示（AGENTS.md）＋Skills一覧**を合成して提示/出力する仕組み（＝“このツール”）を、Codex の Skill として再現できるかを整理する。
- 公式一次情報として、Codex の AGENTS.md 仕様と Skills 仕様（作り方/配置場所/読み込み挙動）を確認する。

## 2. 前提 / 解釈（曖昧さがある場合）
- 仮定: ここで言う「このツール」は、あなたが貼ってくれた長文（共通ルール + `--- project-doc ---` + Skills リスト）を **生成/提示する仕組み**を指す。
- 仮定: 目的は「(1) 指示の常時適用」ではなく「(2) 指示やスキル情報を “まとめて出力” できる」こと、または両方である。
- 未確定: 実際のツール名/コマンド/保存先/誰が使うか（個人orチーム）/出力フォーマット要件。

## 3. 調査方法（どう調べたか）
- OpenAI 公式ドキュメントで、Codex が読み込む指示ファイル（`AGENTS.md` / `AGENTS.override.md` / `~/.codex/AGENTS.md` など）の探索・優先順位・サイズ制限を確認した。
- OpenAI 公式ドキュメントで、Skills の構造（必須 `SKILL.md`、読み込みタイミング、配置場所、命名/記述ルール、オープン標準）を確認した。
- 参考として、Agent Skills のオープン標準（.skill 形式）と、OpenAI 公開の skills リポジトリ（例/導入）も確認した。

## 4. 調査結果（詳細）
### 4.1 重要ポイント
- **AGENTS.md と Skills は役割が違う**:
  - AGENTS.md は「このディレクトリ以下で常に効く指示」を Codex が自動で集めます（グローバル/プロジェクト/サブディレクトリの階層）。一方 Skill は「名前+説明でトリガーされる“任意の追加コンテキスト/手順”」で、本文は必要時だけロードされます。  
  - したがって “常時適用したいルール” を Skill だけで担保するのは設計的に不安定で、基本は AGENTS.md 側に置くのが筋です。
- **Skill は静的資材＋（任意）スクリプトで強い**:
  - Skill は `SKILL.md`（YAML frontmatter: `name`, `description`）が必須で、必要なら `scripts/`, `references/`, `assets/` を同梱できます。
  - “毎回同じ集計/生成をする”タイプ（今回なら「AGENTS.md 群と Skills 一覧を収集して Markdown を生成」）は、スクリプト同梱が向きます。
- **配置場所（共有範囲）を先に決める必要**:
  - Skills はプロジェクト共有なら `<repo>/.codex/skills/`、個人用なら `~/.codex/skills/` に置けます（複数階層も可）。  
  - どちらに置くかで、チームへの配布方法と “誰が同じ挙動になるか” が変わります。
- **“Skills 一覧”の扱いは注意**:
  - Codex は各 Skill を自動で読み込むのではなく、通常は “メタ情報（名前・説明・パス）” だけが常時見える設計です。  
  - 「インストールされている Skill を列挙して、説明/パス付きで出力する」こと自体は、ファイルシステムをスキャンするスクリプトで実装可能ですが、プライベートパス/機密ファイル名の露出や、出力量の肥大化に注意が必要です。

### 4.2 根拠つき解説（段落ごとに引用）
#### 4.2.1 AGENTS.md（常時適用の指示）について
Codex は `AGENTS.md`（および `AGENTS.override.md`）を「ディレクトリ配下で効く指示」として扱い、カレントディレクトリから親へ辿って該当ファイルを収集し、最も近い指示を優先して適用します。さらに、ユーザーのホーム配下 `~/.codex/AGENTS.md` を“グローバル指示”として使える設計です。  
出典: OpenAI Developer Platform「AGENTS.md」ページ（探索・優先順位・ホームの扱い）  

また、プロジェクト側の指示（プロジェクト説明）はサイズ上限があり、超過すると Codex が無視する可能性があります（`project_doc_max_bytes` のデフォルト上限が示されています）。よって「全部まとめて出力する」場合も、生成物サイズのガードやトリミング方針が必要です。  
出典: OpenAI Developer Platform「AGENTS.md」ページ（サイズ制限）  

#### 4.2.2 Skills（任意追加の手順/資材）について
Skills は「モジュール化されたプロンプトパッケージ」で、`SKILL.md`（YAML frontmatter: `name`, `description`）が必須、必要に応じて scripts/references/assets を同梱します。重要なのは、通常 “Skill 本文は常時ロードされず”、メタ情報だけが常時見えて、**起動（トリガー）時に本文がロードされる**という点です。  
出典: OpenAI Developer Platform「Skills overview」「Create custom skills」  

Skills の保存場所は、プロジェクト単位なら `<repo>/.codex/skills/`、ユーザー単位なら `~/.codex/skills/` が公式に案内されています。つまり「チーム全員に同じ Skill を配る」なら repo 配下に置くのが自然です。  
出典: OpenAI Developer Platform「Skills overview」「Create custom skills」  

また、Skills は “Agent Skills” のオープン標準（.skill）に基づく位置づけで、公開リポジトリとして `openai/skills` があり、配布/インストールの例が示されています。  
出典: OpenAI「Skills overview」、GitHub `openai/skills` README（導入・配布の例）  

#### 4.2.3 以上を踏まえた「このツール」を Skill で再現できる範囲
「共通ルールを常時適用する」のは AGENTS.md に寄せるのが公式の設計で、Skill は “必要に応じて呼び出す” 仕組みなので、Skill だけで常時適用を担保するのは難しいです。  
ただし、「AGENTS.md 群と Skills 一覧を収集して Markdown にまとめ、指定パスに保存する」タイプの“生成ツール”は、Skill（＋scripts）として十分再現できます（入出力が固定でき、繰り返し同じ処理をするため）。  
出典: OpenAI Developer Platform「AGENTS.md」「Create custom skills」（役割と読み込みモデル）  

### 4.3 Skill 実装に落とすためのワークフロー分割（案）
以下は「“このツール”＝指示/スキル情報の合成出力」と仮定した場合の分割です。

1) 対象決め（入力の確定）
   - どのルートを対象にするか（`cwd` なのか git ルートなのか）
   - 集める指示ファイルの範囲（`~/.codex/AGENTS.md` を含めるか、`AGENTS.override.md` も含めるか）
   - Skills の列挙範囲（`<repo>/.codex/skills` だけか、`~/.codex/skills` も含めるか）

2) 収集（ファイル読み取り）
   - 指示ファイルを探索して順序付きで読み込む（近い順/優先順を明記）
   - **機密ガード**: `.env` 等を誤って読まない/出さない、最大バイトで打ち切る

3) 整形（レンダリング）
   - セクション見出しを固定して Markdown に整形（あなたの例のように `--- project-doc ---` で区切る、など）
   - 長文になる場合の省略ルール（例: “最初の N 行だけ + 省略”）

4) 出力（保存）
   - 既定の保存先（例: `docs/report/YYYYMMDD_*.md`）に保存
   - 最後に `done: <path>` を出す（自動処理/ログのため）

5) 検証
   - “期待した指示ファイルが全部入っているか” をチェック（特に override/階層）
   - サイズ上限・省略が期待通りかチェック

### 4.4 Skill の具体像（構成案）
#### 4.4.1 Skill 名（例）
- `agents-instructions-export`（例）: AGENTS 指示と skills 情報を集計して Markdown に出力する

#### 4.4.2 ディレクトリ構造（例）
```
agents-instructions-export/
  SKILL.md
  scripts/
    export_instructions.py
  assets/
    output_template.md
  references/
    codex_agents_and_skills_links.md
```

#### 4.4.3 SKILL.md に書くべき要点（例）
- 何をする Skill か（AGENTS/skills を集計して 1ファイルにする）
- いつ使うか（「このリポジトリの指示を別エージェントに渡したい」「現在の適用指示を可視化したい」など）
- 実行手順（`python scripts/export_instructions.py --out docs/report/...` のように）
- 失敗時の確認観点（AGENTS の探索順、skills 置き場、権限、サイズ）

## 5. 高リスク領域の注意（該当する場合のみ）
- 機密/個人情報の混入リスク: “まとめ出力”は、誤って `.env` や鍵ファイル名、個人パスを出力しやすいので、**収集対象をホワイトリスト化**し、最大サイズ/マスキング/除外ルールを必須にするのが安全です。
- ツール実行の安全: 生成スクリプトは読み取り中心にし、リポジトリを書き換える処理（自動整形や自動コミット等）を混ぜない方が事故が減ります。

## 6. 限界 / 未検証 / TODO
- 未検証: あなたの言う「このツール」が何を指すかが確定していない（常時適用か、合成出力か、別物か）。
- TODO: 次の情報が揃うと “再現可能性” を断定できる
  - ツール名/コマンド/入力例/出力例（保存先含む）
  - 使う目的（誰が/何のために/どの頻度で）
  - “Skills 一覧”に含めたい範囲（repo だけ or user も）
- Run 2の方向性案（必要なら）:
  - ツールが「常時適用」を狙っている場合: Skill での再現ではなく AGENTS.md の設計に寄せる形で整理し直す（どこに何を書くか、override の使い所）。
  - ツールが「合成出力」なら: 実物の出力要件に合わせてテンプレ/トリミング規則/スキャン対象を具体化する。

## 7. 参考文献（リンク）
- OpenAI Developer Platform: AGENTS.md（探索/優先順位/ホーム指示/サイズ制限）: https://developers.openai.com/codex/agents-md
- OpenAI Developer Platform: Skills overview（配置場所、オープン標準の位置づけ）: https://developers.openai.com/codex/skills
- OpenAI Developer Platform: Create custom skills（`SKILL.md` の必須項目、ロードモデル、上限など）: https://developers.openai.com/codex/skills/create-skill
- OpenAI Blog: Testing Agent Skills systematically with Evals（評価・テスト観点）: https://developers.openai.com/blog/eval-skills
- GitHub: openai/skills（公開 skills の例、導入/配布の入口）: https://github.com/openai/skills
- Agent Skills（オープン標準の概要）: https://agentskills.io
