# 音声文字起こしWebアプリ（STT Web App）

Streamlitを使用した音声文字起こしWebアプリ。複数のSTTモデルに対応し、Gemini Flash 2.5-liteによる自動構造化機能を搭載。

## 運用メモ
- 本番デプロイ先: Streamlit Community Cloud
- Tauri版（デスクトップアプリ）: `../stt-desktop`（`stt-suite/stt-desktop`。本リポジトリの隣にある別リポジトリ）

## 課題管理

- 顧客要望、不具合、対応方針、ステータスは、[室島精工様_課題管理表](https://docs.google.com/spreadsheets/d/1JHO7Rb57ivusmi8NYmu7iSd51pie-Jxa5Ec8AaFVwuU/edit?gid=0#gid=0)を正本として管理します。
- このリポジトリでは、主にカテゴリ「音声DB: Web」の課題を扱います。
- Windows版の課題は `stt-desktop` リポジトリで扱います。

## 機能

- マイク録音 / 複数ファイルの読み込み（作業録音・社長音声・業務記録の3カテゴリ）
- 5つのSTTモデル対応（OpenAI、Google Cloud、Amazon、Azure、ElevenLabs）
- Gemini Flash 2.5-liteによる文字起こしテキストの自動構造化
- Turso(libSQL)/SQLiteデータベース保存（本番はTursoに完全移行）
- Basic認証によるアクセス制限（オプション）

## クイックスタート

### 1. セットアップ

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストール
uv sync

# 環境変数を設定（.env.exampleをコピーして編集）
cp .env.example .env
nano .env
```

### 2. データベース設定

**Turso(libSQL) 使用（本番・推奨）**:
1. TursoでDBを作成しURL/トークンを取得
2. `.env` の `DATABASE_URL` を `sqlite+libsql://<db>-<org>.turso.io?secure=true&authToken=...` に設定
3. 初回起動時に `audio_transcription_chunks` のベクトル式インデックス（libsql_vector_idx）が自動作成されます

**ローカル開発**: 通常のSQLiteが自動使用されます（RAGは無効）

### 3. 起動と使用

```bash
# アプリ起動
./run_app.sh

# ブラウザで http://localhost:8501 を開く
```

**使い方**（タブ構成: 🎙️ 録音 / 🗄️ データベース / 💬 AIチャット）:
1. サイドバーでSTTモデルを選択（デフォルト: ElevenLabs）
2. 「録音」タブのサブタブ（作業録音 / 社長音声 / 業務記録）で音声入力。各画面は同じ構成:
   - 見出し・説明 → 取り込み情報（社長音声・業務記録のみ: タイトル/話者/録音日時）
   - 「マイク録音」（停止で自動処理） / 「ファイルを読み込む（複数可）」 → 「文字起こし開始」（作業録音）/「取り込み開始」（社長音声・業務記録）
   - 「処理キュー」「処理結果」で確認
3. 「データベース」タブで3カテゴリを1つの一覧で検索（カテゴリ / 期間 / タグ / 話者 / キーワード）・詳細表示・削除
4. 「AIチャット」タブで録音データに質問（検索ソース: 作業録音 / 社長音声 / 業務記録）

カテゴリ定義（キー `audio`/`ceo`/`work`、ラベル、タグ、テーブル）は `src/ui/categories.py` が正本。画面共通のUI部品は `src/ui/components.py`。

## 環境変数設定

### 必須APIキー

| 用途 | 環境変数 | 備考 |
|------|---------|------|
| **STTモデル** | 下記いずれか1つ | 選択したモデル用 |
| **構造化** | GEMINI_API_KEY | Gemini Flash 2.5-lite用 |
| **データベース** | DATABASE_URL | `sqlite+libsql://...`（Turso） |
| **Basic認証** | BASIC_AUTH_USERNAME<br>BASIC_AUTH_PASSWORD | オプション |

### STTモデル別環境変数

| サービス | 環境変数 | ファイルサイズ制限 |
|---------|---------|------------------|
| OpenAI | OPENAI_API_KEY | 25MB |
| Google Cloud | GOOGLE_CLOUD_PROJECT<br>GOOGLE_APPLICATION_CREDENTIALS | 10分（約10MB） |
| Amazon | AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY | 2GB（S3経由） |
| Azure | AZURE_SPEECH_KEY<br>AZURE_SPEECH_REGION | 100MB |
| ElevenLabs | ELEVENLABS_API_KEY | 1GB、4.5時間 |

### サンプル.env

```env
# データベース（例1: Turso/libSQL）
DATABASE_URL=sqlite+libsql://your-db-your-org.turso.io?secure=true&authToken=your-turso-token

# データベース（例2: ローカルSQLite）
# DATABASE_URL=sqlite:///./audio_transcriptions.db

# STTモデル（ElevenLabsの例）
ELEVENLABS_API_KEY=xi-xxxxxxxxxxxxxxxxxxxxx

# 構造化機能
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxx

# Basic認証（オプション）
BASIC_AUTH_USERNAME=admin
BASIC_AUTH_PASSWORD=secure-password
```

## 設定とデータベース

### 設定の永続化
- STTモデル選択、構造化機能、デバッグモードの設定は`.app_settings.json`に自動保存

### データベース利用時のポイント（Turso専用）
- `DATABASE_URL` に `sqlite+libsql://<db名>-<org>.turso.io?secure=true&authToken=...` を設定するとリモートTursoに接続可能
- `audio_transcription_chunks` の `libsql_vector_idx` 作成（アプリが初回自動作成）と `OPENAI_API_KEY` 設定でRAGタブが有効化

### データベーススキーマ（Turso）

| カラム名 | 型 | 説明 |
|---------|-----|------|
| 音声ID | SERIAL | 主キー |
| 音声ファイルpath | VARCHAR(500) | ファイル名 |
| 発言人数 | INTEGER | デフォルト: 1 |
| 録音時刻 | TIMESTAMP | 処理時刻 |
| 録音時間 | FLOAT | 秒 |
| 文字起こしテキスト | TEXT | 結果 |
| 構造化データ | JSONB | Gemini出力 |
| タグ | VARCHAR(200) | 自動生成 |

### Basic認証とCookie
- 環境変数で有効化、24時間有効なCookie認証トークン使用
- ログアウトボタンはサイドバーに表示

## 対応フォーマット
WAV、MP3、M4A、FLAC、OGG

## プロジェクト構造（主要ファイル）

```
stt/
├── src/
│   ├── app.py               # メインアプリ（3タブ: 録音 / データベース / AIチャット）
│   ├── ui/
│   │   ├── categories.py    # カテゴリ定義（作業録音 / 社長音声 / 業務記録）の正本
│   │   ├── components.py    # 画面共通UI部品（入力欄・処理キュー・処理結果・カテゴリ/期間フィルタ）
│   │   ├── sidebar.py
│   │   └── tabs/            # audio_tab（作業録音）/ ceo_tab（社長音声・業務記録共用）/ db_tab / rag_tab
│   ├── services/
│   │   ├── audio_processor.py  # 作業録音の処理（マイク/ファイル共通）
│   │   └── ceo_processor.py    # 社長音声・業務記録の処理
│   ├── stt_wrapper.py       # STT統一インターフェース
│   └── text_structurer.py   # Gemini構造化
├── scripts/                 # 各STT実装
├── database/                # DB関連
├── .env.example             # 環境変数サンプル
├── pyproject.toml           # 依存関係
└── run_app.sh               # 起動スクリプト
```

## トラブルシューティング

### よくある問題と対策

| 問題 | 対策 |
|------|------|
| APIキーエラー | 選択モデルの環境変数を確認 |
| .env変更が反映されない | ページリロードまたはサイドバーで手動再読み込み |
| モジュールエラー | `uv sync`で依存関係を再インストール |
| 音声処理失敗 | ファイル形式とサイズ制限を確認 |

### デバッグモード
サイドバーの「デバッグ設定」で有効化。`logs/`ディレクトリにログ出力:
- `streamlit_app.log`: アプリ全体
- `elevenlabs_debug.log`: ElevenLabs詳細

### 環境変数の確認
サイドバーの「環境変数の設定状況」で現在の設定を確認可能

## 開発

```bash
# パッケージ追加/削除
uv add package-name
uv remove package-name

# 依存関係更新
uv lock --upgrade
```


## 重要な注意事項
- **import-instruction-reminders**: 要求されたことのみ実行
- **既存ファイル優先**: 新規作成より既存ファイル編集を優先
- **ドキュメント作成制限**: 明示的に要求されない限り*.mdファイル作成禁止

- RAG機能は Turso(libSQL) 専用です（Postgres対応は削除）。
- `.env` では必須の `OPENAI_API_KEY` に加え、必要に応じて `EMBEDDING_MODEL` (既定: text-embedding-3-large、dimensions=1536で格納), `EMBEDDING_DIM`, `RAG_COMPLETION_MODEL`, `ENABLE_RAG` を設定可能。
- 新規保存分は自動でチャンク化・埋め込み登録。既存データをRAG対応させるには再保存やバックフィルスクリプトが必要。
- Streamlit UIのQAチャットは「💬 AIチャット」の1タブ。検索対象はソース選択pills（作業録音/社長音声/業務記録、既定は全ソース）で切り替える。
- Supabase関連の機能（Storage・移行ドキュメント等）は削除済みです。

## Agent Notes（RAG開発向けメモ）
- 本リポジトリはデータベースをTurso(libSQL)に完全移行済み。Postgres/pgvector対応はコードから削除済みです。関連依存（psycopg2, pgvector）も`pyproject.toml`から除外しました。
- QAチャット（「AIチャット」タブ）のアーキテクチャ:
  - 出口は1タブ（`ui/tabs/rag_tab.py`）で、検索対象はソース選択pills（audio/ceo/work、既定は全ソース）。会話ログ`rag_chat_logs.chat_kind`はデスクトップ版と同じ規則で記録（作業録音を含む検索="audio" / 含まない="ceo"、NULLは旧データ=audio扱い。参照ソースは`contexts[].source`に残る）
  - 新規保存分は保存時に即時索引化（作業録音: audio_processor、社長音声・業務記録: ceo_processor）。デスクトップ版等の外部保存分はQAタブ表示時に自動取り込み（20件以下は自動、超過時はボタン表示）
  - 検索モードは3種: `search`（ハイブリッド検索）/ `browse`（期間・要約だけが手がかりの質問。新しい順に最大`RAG_AGGREGATE_MAX_DOCS`=30件を薄く読む）/ `followup`（形式変更・メタ質問。再検索せず前回の参照録音を再利用）
  - `services/rag/query_cleaner.py`: 検索計画の判断材料（指示語除去・内容語判定・集約/追問判定・STT表記ゆれ同義語辞書）。同義語はprodコーパス走査で実在確認したもののみ登録（ヒケ=引け、ソリ=反り等）。指示語バイグラムはBM25を汚染するためFTSクエリから除外する（実測nDCG@6 0.47→0.64）
  - `services/rag/search_service.py`: 検索実行層。ベクトル検索（`vector_distance_cos`全走査+SQL日付フィルタ）/ キーワード検索（FTS5）/ 期間ブラウズの3操作。Phase 2（agentic search）ではこれらをLLMのツールとして公開する想定
  - `services/rag/tokenizer.py`: FTS5用の文字バイグラムトークナイザ。索引テーブルは`rag_fts_audio`/`rag_fts_ceo`（Python側で行を管理、トリガ無し）。現場用語・型番の完全一致検索を辞書非依存で保証
  - `services/rag/date_utils.py`: クエリからの日付範囲抽出（「7/28」「先月」に加え「X～Y」「XからYまで」の範囲も対応）。検索は正規化済み`recorded_date`列（JST, YYYY-MM-DD）へのSQL WHEREで行う（事後フィルタ禁止）
  - `services/rag/context_builder.py`: コンテキストは録音単位（短い録音は全文、長い録音はヒット周辺の結合）。チャンク断片をそのまま渡さない。browse/集約時は`per_doc_cap`で1件を薄くして件数を優先
  - `services/rag/reconcile.py` + `RAGService.reconcile()`: デスクトップ版保存分・社長音声の索引差分を補完。UIから自動/ボタン実行、CLIは`scripts/backfill_rag.py`
  - 期間拡大・部分参照などの検索側の事情は`build_chat_messages(notes=…)`でモデルに明示する（モデルが期間外の録音を「日付の矛盾」と誤解しないように）
- ハイブリッド検索の融合は重み付きRRF（Reciprocal Rank Fusion）。スコアの絶対値でのブレンドはキャリブレーション問題があるため禁止。`RAG_HYBRID_ALPHA`既定0.4（prod実データ評価でキーワード側が強いため）
- `created_at`はWeb版（naive UTC）とデスクトップ版（RFC3339 UTC）で形式が混在。日付判定には必ず`recorded_date`を使う
- QA検索タブの回答生成は「ストリーミングのみ」です。非ストリーミングAPIはコードから撤去済みです。
- 既定のRAGモデル: `EMBEDDING_MODEL=text-embedding-3-large`（`dimensions=1536`で格納、スキーマ変更不要）, `RAG_COMPLETION_MODEL=gpt-5.6-luna`。Responses APIを使用。
- 埋め込みモデルを変更すると、索引時モデルを記録する`rag_index_meta`マーカーとの不一致で全録音が自動的に再索引対象になる（QAタブのボタン1回 or `scripts/backfill_rag.py`で移行。旧ベクトルとの混在を防ぐため）。
- `EMBEDDING_DIM` を変更する場合はDB列定義が固定のため、再作成（既存チャンク削除→再インデックス）が必要。
- 検索品質の確認は `uv run python scripts/eval_rag.py "質問"`（検索計画と参照録音を表示。生成なし）。
