import logging
import time
import os
from array import array
from datetime import datetime, timezone
from typing import Sequence

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import make_url
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import deferred, relationship, sessionmaker
from sqlalchemy.types import UserDefinedType

# Postgres(pgvector)対応は廃止。libSQL専用。

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def utcnow_naive() -> datetime:
    """UTC現在時刻をnaiveで返す(created_at列の書き込み用)。

    created_at はDB全体で「naive UTC」に統一する。デスクトップ版はUTCの
    RFC3339文字列を書き込んでおり、日付判定(services/rag/reconcile.py の
    normalize_to_jst_date)はnaive値をUTCとして解釈する。bare datetime.now()
    だとサーバーのローカルTZに依存し、JSTマシンで動かすと9時間ズレるため、
    必ずこの関数を使う。aware値を書くとSQLiteの文字列形式が変わり
    デスクトップ版との互換性に影響するため、naiveへ落とす。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_libsql(url: str) -> bool:
    try:
        drivername = make_url(url).drivername
        return "libsql" in drivername
    except Exception:
        return False


# Postgres検出は不要になったため削除


def _extract_libsql_auth_token(url: str) -> str | None:
    """DATABASE_URL から authToken を抽出（sqlalchemy-libsql 0.2系向け）。"""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query or "")
        token = qs.get("authToken") or qs.get("authtoken") or qs.get("auth_token")
        if token and len(token) > 0:
            return token[0]
    except Exception:
        pass
    return None


def _strip_auth_token_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query or "")
        # auth に関係するキーを除去
        for k in ["authToken", "authtoken", "auth_token"]:
            if k in qs:
                qs.pop(k, None)
        new_q = urlencode(qs, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_q, parsed.fragment))
    except Exception:
        return url

class AudioTranscription(Base):
    __tablename__ = 'audio_transcriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)
    # 正規化済みの録音日(JST, 'YYYY-MM-DD')。created_atはWeb版(naive UTC)と
    # デスクトップ版(UTC RFC3339)で形式が異なるため、検索はこの列で行う。
    recorded_date = Column(String(10), nullable=True, index=True)
    duration_seconds = Column(Float, nullable=False)
    transcript = Column(Text, nullable=False)
    structured_json = Column(JSON, nullable=True)
    tags = Column(String(200), nullable=True)
    model_id = Column(String(100), nullable=True)
    language_code = Column(String(10), nullable=True)
    # 単語タイムスタンプ({text, start, end, type, speaker_id?}の配列)。
    # stt-desktop と同名・同形式。_json はSTTが返したまま(VAD後音声基準)、
    # _original_json はVAD前=元音声基準へ復元した値。サイズが大きいため
    # 通常のSELECTでは読み込まない(deferred)。
    word_timestamps_json = deferred(Column(JSON, nullable=True))
    word_timestamps_original_json = deferred(Column(JSON, nullable=True))
    chunks = relationship(
        "AudioTranscriptionChunk",
        back_populates="transcription",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<AudioTranscription(id={self.id}, file_path={self.file_path})>"


class CeoTranscription(Base):
    """社長音声の文字起こし結果。stt-desktop の ceo_transcriptions テーブルと互換。"""

    __tablename__ = 'ceo_transcriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(500), nullable=False)
    local_file_path = Column(String(500), nullable=True)
    source_file_path = Column(String(500), nullable=True)
    source_file_size_bytes = Column(Integer, nullable=True)
    source_file_modified_at = Column(String(64), nullable=True)
    source_file_hash = Column(String(64), nullable=True)
    source_app = Column(String(32), nullable=False, default="unknown", server_default=text("'unknown'"))
    input_method = Column(String(32), nullable=False, default="unknown", server_default=text("'unknown'"))
    title = Column(String(500), nullable=True)
    speaker = Column(String(200), nullable=True)
    recorded_at = Column(String(64), nullable=True)
    recorded_date = Column(String(10), nullable=True, index=True)
    model_id = Column(String(100), nullable=True)
    language_code = Column(String(10), nullable=True)
    transcript = Column(Text, nullable=False)
    structured_json = Column(JSON, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    tags = Column(String(200), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow_naive)
    # 単語タイムスタンプ(audio_transcriptionsと同形式)
    word_timestamps_json = deferred(Column(JSON, nullable=True))
    word_timestamps_original_json = deferred(Column(JSON, nullable=True))

    def __repr__(self):
        return f"<CeoTranscription(id={self.id}, title={self.title})>"


DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./audio_transcriptions.db')
IS_LIBSQL = _is_libsql(DATABASE_URL)


class LibSQLF32Vector(UserDefinedType):
    """libSQLのF32_BLOBカラムをSQLAlchemyで扱うための型。"""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **kw):  # type: ignore[override]
        return f"F32_BLOB({self.dimension})"

    def bind_processor(self, dialect):  # type: ignore[override]
        def process(value):
            if value is None:
                return None
            if isinstance(value, (bytes, bytearray, memoryview)):
                raw = bytes(value)
            else:
                raw = _vector_to_f32_blob(value, self.dimension)
            return raw

        return process

    def result_processor(self, dialect, coltype):  # type: ignore[override]
        def process(value):
            if value is None:
                return None
            return _blob_to_vector(value, self.dimension)

        return process

    @property
    def python_type(self):  # type: ignore[override]
        return list


def _vector_to_f32_blob(values: Sequence[float], dimension: int) -> bytes:
    arr = array('f', (float(v) for v in values))
    length = len(arr)
    if length != dimension:
        if length > dimension:
            arr = arr[:dimension]
        else:
            arr.extend((0.0,) * (dimension - length))
    return arr.tobytes()


def _blob_to_vector(blob: bytes, dimension: int) -> list[float]:
    if isinstance(blob, memoryview):
        data = blob.tobytes()
    else:
        data = bytes(blob)
    arr = array('f')
    arr.frombytes(data)
    if len(arr) > dimension:
        arr = arr[:dimension]
    return list(arr)


if IS_LIBSQL:
    VECTOR_BACKEND = "libsql"
else:
    VECTOR_BACKEND = None

USE_VECTOR = VECTOR_BACKEND is not None
LIBSQL_VECTOR_INDEX_NAME = "audio_transcription_chunks_embedding_idx"


class AudioTranscriptionChunk(Base):
    __tablename__ = 'audio_transcription_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcription_id = Column(
        Integer,
        ForeignKey('audio_transcriptions.id', ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    # チャンクの録音内時刻範囲(秒)。time_basisは時刻の基準:
    # "original"=VAD前の元音声基準 / "vad"=VAD後音声基準(元音声とズレの可能性)。
    # 単語タイムスタンプが無い録音ではNULL(従来動作)。
    start_sec = Column(Float, nullable=True)
    end_sec = Column(Float, nullable=True)
    time_basis = Column(String(16), nullable=True)
    if VECTOR_BACKEND == "libsql":
        embedding = Column(LibSQLF32Vector(EMBEDDING_DIM), nullable=False)
    else:
        # ローカルSQLite（非libSQL）の場合はRAG無効のためJSONで可
        embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive, nullable=False)

    transcription = relationship("AudioTranscription", back_populates="chunks")


class CeoTranscriptionChunk(Base):
    """社長音声のRAG索引用チャンク。audio側と同じ構造。"""

    __tablename__ = 'ceo_transcription_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcription_id = Column(
        Integer,
        ForeignKey('ceo_transcriptions.id', ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    # audio_transcription_chunks と同じ時刻範囲カラム
    start_sec = Column(Float, nullable=True)
    end_sec = Column(Float, nullable=True)
    time_basis = Column(String(16), nullable=True)
    if VECTOR_BACKEND == "libsql":
        embedding = Column(LibSQLF32Vector(EMBEDDING_DIM), nullable=False)
    else:
        embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow_naive, nullable=False)


class RAGChatLog(Base):
    __tablename__ = 'rag_chat_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=True, index=True)  # セッション管理用UUID
    chat_kind = Column(String(20), nullable=True)  # "audio"(作業録音) / "ceo"(社長音声)。NULLは旧データ=audio扱い
    created_at = Column(DateTime, default=utcnow_naive, nullable=False)
    user_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    contexts = Column(JSON, nullable=True)  # 参照したチャンクやスコアを保持
    used_hybrid = Column(Boolean, default=True, nullable=False)
    alpha = Column(Float, nullable=True)
    date_filter_applied = Column(Boolean, default=False, nullable=True)  # 日付フィルタ適用有無

def _tolerant_json_loads(value):
    """JSON列の読み出し。デスクトップ版が空文字列等の不正JSONを書き込む
    ことがあるため、パース失敗時はNoneを返してアプリを止めない。"""
    import json as _json

    try:
        return _json.loads(value)
    except (ValueError, TypeError):
        logger.warning("JSON列のパースに失敗したためNoneとして扱います: %r", value[:80] if isinstance(value, str) else value)
        return None


def _compact_json_dumps(value) -> str:
    """JSON列の書き込み。word_timestamps_json のような大きい配列で
    ASCIIエスケープ(\\uXXXX)がサイズを数倍にするため、UTF-8のまま
    コンパクトに書く(デスクトップ版 serde_json と同じ方針)。"""
    import json as _json

    return _json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# データベース接続設定
engine_kwargs = dict(
    echo=False,
    json_deserializer=_tolerant_json_loads,
    json_serializer=_compact_json_dumps,
)
if IS_LIBSQL:
    engine_kwargs["pool_pre_ping"] = True

if IS_LIBSQL:
    connect_args = {}
    token = _extract_libsql_auth_token(DATABASE_URL) or os.getenv("TURSO_AUTH_TOKEN") or os.getenv("LIBSQL_AUTH_TOKEN")
    if token:
        # sqlalchemy-libsql >=0.2.0 は connect_args の "auth_token" を推奨
        connect_args["auth_token"] = token
    url_for_engine = _strip_auth_token_from_url(DATABASE_URL)
    if connect_args:
        engine = create_engine(url_for_engine, connect_args=connect_args, **engine_kwargs)
    else:
        engine = create_engine(url_for_engine, **engine_kwargs)
else:
    engine = create_engine(DATABASE_URL, **engine_kwargs)

# Postgres向けの初期化は削除（Turso専用化）

# テーブル作成（所要時間をログ出力）
_t0 = time.time()
Base.metadata.create_all(bind=engine)
_t1 = time.time()
try:
    logger.debug("DB schema check/create completed in %.3fs", _t1 - _t0)
except Exception:
    pass


def _ensure_columns(table_name: str, columns: dict[str, str]) -> None:
    """既存SQLite/libSQLテーブルに不足カラムを追加する。

    SQLAlchemyのcreate_allは既存テーブルを変更しないため、desktop版の
    ensure_tables相当として互換カラムを明示的に補完する。
    """

    try:
        existing = {col["name"] for col in inspect(engine).get_columns(table_name)}
    except Exception as exc:
        logger.warning("%s のカラム確認に失敗: %s", table_name, exc)
        return

    missing = [(name, spec) for name, spec in columns.items() if name not in existing]
    if not missing:
        return

    try:
        with engine.begin() as connection:
            for name, spec in missing:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {spec}"))
                logger.info("%s.%s カラムを追加しました", table_name, name)
    except Exception as exc:
        logger.warning("%s のカラム追加に失敗: %s", table_name, exc)


_ensure_columns(
    "audio_transcriptions",
    {
        "recorded_date": "TEXT",
        # 単語タイムスタンプ(stt-desktop の ensure_tables と同名・同形式。
        # desktop側で作成済みのDBでは既存カラムのため何もしない=冪等)
        "word_timestamps_json": "TEXT",
        "word_timestamps_original_json": "TEXT",
    },
)

_ensure_columns("rag_chat_logs", {"chat_kind": "TEXT"})

_ensure_columns(
    "ceo_transcriptions",
    {
        "recorded_date": "TEXT",
        "file_path": "TEXT",
        "local_file_path": "TEXT",
        "source_file_path": "TEXT",
        "source_file_size_bytes": "INTEGER",
        "source_file_modified_at": "TEXT",
        "source_file_hash": "TEXT",
        "source_app": "TEXT NOT NULL DEFAULT 'unknown'",
        "input_method": "TEXT NOT NULL DEFAULT 'unknown'",
        "title": "TEXT",
        "speaker": "TEXT",
        "recorded_at": "TEXT",
        "model_id": "TEXT",
        "language_code": "TEXT",
        "transcript": "TEXT",
        "structured_json": "TEXT",
        "duration_seconds": "REAL",
        "tags": "TEXT",
        "created_at": "TEXT",
        "word_timestamps_json": "TEXT",
        "word_timestamps_original_json": "TEXT",
    },
)

# RAGチャンクの録音内時刻範囲(発言単位タイムスタンプのRAG利用)
for _chunk_table in ("audio_transcription_chunks", "ceo_transcription_chunks"):
    _ensure_columns(
        _chunk_table,
        {
            "start_sec": "REAL",
            "end_sec": "REAL",
            "time_basis": "TEXT",
        },
    )

# セッション作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# RAG検索用のFTS5テーブル名(バイグラム索引テキストをPython側で管理)
RAG_FTS_TABLES = {
    "audio": "rag_fts_audio",
    "ceo": "rag_fts_ceo",
}
CEO_VECTOR_INDEX_NAME = "ceo_transcription_chunks_embedding_idx"

if IS_LIBSQL:
    _t2 = time.time()
    try:
        with engine.begin() as connection:
            # 旧FTS(unicode61: 日本語でBM25が機能しない)とトリガを撤去
            for trig in (
                "audio_transcription_chunks_ai",
                "audio_transcription_chunks_ad",
                "audio_transcription_chunks_au",
            ):
                connection.execute(text(f"DROP TRIGGER IF EXISTS {trig}"))
            connection.execute(text("DROP TABLE IF EXISTS audio_transcription_chunks_fts"))

            # ベクトル式インデックス
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    f"{LIBSQL_VECTOR_INDEX_NAME} "
                    "ON audio_transcription_chunks(libsql_vector_idx(embedding))"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    f"{CEO_VECTOR_INDEX_NAME} "
                    "ON ceo_transcription_chunks(libsql_vector_idx(embedding))"
                )
            )

            # RAG用の補助インデックス
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_by_transcription "
                    "ON audio_transcription_chunks(transcription_id, chunk_index)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_ceo_chunks_by_transcription "
                    "ON ceo_transcription_chunks(transcription_id, chunk_index)"
                )
            )

            # FTS5(バイグラム方式)。rowid=チャンクid。行の追従はアプリ側の
            # インデックス処理・再同期処理(services/rag/reconcile.py)が行う。
            for fts_name in RAG_FTS_TABLES.values():
                connection.execute(
                    text(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_name} "
                        "USING fts5(tokens, tokenize='unicode61')"
                    )
                )
    except Exception as exc:  # pragma: no cover - 初期化時の警告
        logger.warning("libSQLの初期化（ベクトル/FTS）に失敗: %s", exc)
    finally:
        _t3 = time.time()
        try:
            logger.debug("libSQL init (indexes/FTS) completed in %.3fs", _t3 - _t2)
        except Exception:
            pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
