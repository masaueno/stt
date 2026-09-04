"""索引作成時のチャンク時刻付与(_index_generic)の統合テスト。

本物のprod DB(Turso)には接続せず、in-memory SQLiteで検証する。
埋め込みAPIはダミーに差し替える。
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402

from models import (  # noqa: E402
    AudioTranscription,
    AudioTranscriptionChunk,
    RAG_FTS_TABLES,
    SessionLocal,
    engine,
)
from services.rag_service import RAGService  # noqa: E402


TRANSCRIPT = "".join(f"作業{i}の記録がここにある。" for i in range(100))  # 1200文字 → 複数チャンク


def _words_for(transcript, sec_per_char=0.5, offset=0.0):
    return [
        {"text": ch, "start": round(offset + i * sec_per_char, 3),
         "end": round(offset + (i + 1) * sec_per_char, 3), "type": "word"}
        for i, ch in enumerate(transcript)
    ]


@pytest.fixture()
def db():
    # FTS5仮想テーブル(libsql環境でのみ自動作成)をテスト用に用意する
    try:
        with engine.begin() as conn:
            for fts in RAG_FTS_TABLES.values():
                conn.execute(
                    sql_text(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} "
                        "USING fts5(tokens, tokenize='unicode61')"
                    )
                )
    except Exception as exc:  # pragma: no cover - FTS5非対応のsqlite
        pytest.skip(f"FTS5が利用できないためスキップ: {exc}")
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def rag(monkeypatch):
    service = RAGService()
    # in-memory SQLite + APIキーなしでも索引処理を通す
    service._enabled = True
    monkeypatch.setattr(
        service, "_embed_texts", lambda texts: [[0.0, 0.0] for _ in texts]
    )
    return service


def _insert_recording(db, *, raw_json=None, original_json=None) -> int:
    record = AudioTranscription(
        file_path="rec.mp3",
        duration_seconds=600.0,
        transcript=TRANSCRIPT,
        word_timestamps_json=raw_json,
        word_timestamps_original_json=original_json,
    )
    db.add(record)
    db.flush()
    return record.id


class TestIndexChunkTimes:
    def test_original_words_used_with_original_basis(self, db, rag):
        words_vad = _words_for(TRANSCRIPT, sec_per_char=0.4)
        words_original = _words_for(TRANSCRIPT, sec_per_char=0.4, offset=120.0)
        tid = _insert_recording(db, raw_json=words_vad, original_json=words_original)

        assert rag._index_generic(db, "audio", tid, TRANSCRIPT)
        chunks = (
            db.query(AudioTranscriptionChunk)
            .filter_by(transcription_id=tid)
            .order_by(AudioTranscriptionChunk.chunk_index)
            .all()
        )
        assert len(chunks) >= 2
        assert all(c.time_basis == "original" for c in chunks)
        # originalはoffset=120秒から始まる時刻系
        assert chunks[0].start_sec == pytest.approx(120.0, abs=1.0)
        assert chunks[-1].end_sec > chunks[0].start_sec
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.start_sec >= prev.start_sec

    def test_desktop_row_with_raw_only_gets_vad_basis(self, db, rag):
        # デスクトップ保存分: word_timestamps_json のみ(JSON文字列)のケース
        words = _words_for(TRANSCRIPT, sec_per_char=0.3)
        tid = _insert_recording(db, raw_json=None, original_json=None)
        db.execute(
            sql_text(
                "UPDATE audio_transcriptions SET word_timestamps_json = :w WHERE id = :id"
            ),
            {"w": json.dumps(words, ensure_ascii=False), "id": tid},
        )

        assert rag._index_generic(db, "audio", tid, TRANSCRIPT)
        chunks = (
            db.query(AudioTranscriptionChunk)
            .filter_by(transcription_id=tid)
            .order_by(AudioTranscriptionChunk.chunk_index)
            .all()
        )
        assert all(c.time_basis == "vad" for c in chunks)
        assert chunks[0].start_sec == pytest.approx(0.0, abs=0.5)

    def test_row_without_words_keeps_null_times(self, db, rag):
        tid = _insert_recording(db)
        assert rag._index_generic(db, "audio", tid, TRANSCRIPT)
        chunks = db.query(AudioTranscriptionChunk).filter_by(transcription_id=tid).all()
        assert chunks
        assert all(c.start_sec is None and c.end_sec is None for c in chunks)
        assert all(c.time_basis is None for c in chunks)

    def test_reindex_replaces_chunks_and_times(self, db, rag):
        tid = _insert_recording(db, original_json=_words_for(TRANSCRIPT, sec_per_char=0.2))
        assert rag._index_generic(db, "audio", tid, TRANSCRIPT)
        first_count = db.query(AudioTranscriptionChunk).filter_by(transcription_id=tid).count()
        # 再索引しても二重登録されない(reconcileの再実行相当)
        assert rag._index_generic(db, "audio", tid, TRANSCRIPT)
        chunks = db.query(AudioTranscriptionChunk).filter_by(transcription_id=tid).all()
        assert len(chunks) == first_count
        assert len({c.chunk_index for c in chunks}) == len(chunks)
        assert all(c.time_basis == "original" for c in chunks)


class TestReconcileMaxItems:
    def test_max_items_limits_chunkless_rows_and_progresses(self, db, rag):
        from services.rag.reconcile import get_index_meta, set_index_meta
        from services.rag_service import EMBEDDING_MODEL

        # 索引が現行モデルで揃っているDBに、未索引の録音が3件増えた状態
        set_index_meta(db, "embedding_model", EMBEDDING_MODEL)
        ids = [_insert_recording(db) for _ in range(3)]
        db.commit()

        report = rag.reconcile(db, embed=True, max_items=2)
        assert report["indexed"]["audio"] == 2
        assert report["errors"] == 0
        indexed = {
            r[0]
            for r in db.execute(
                sql_text("SELECT DISTINCT transcription_id FROM audio_transcription_chunks")
            ).all()
        }
        assert indexed == set(ids[:2])
        assert rag.pending_counts(db) == {"audio": 1, "ceo": 0}

        # 2回目は残りだけを処理する(先頭からやり直さない)
        report = rag.reconcile(db, embed=True, max_items=2)
        assert report["indexed"]["audio"] == 1
        assert rag.pending_counts(db) == {"audio": 0, "ceo": 0}
        assert get_index_meta(db, "embedding_model") == EMBEDDING_MODEL

    def test_max_items_not_applied_to_model_migration(self, db, rag):
        from services.rag.reconcile import get_index_meta, set_index_meta
        from services.rag_service import EMBEDDING_MODEL

        # マーカーが旧モデル(=モデル変更)なら打ち切らず全件を再索引してマーカーを更新する
        set_index_meta(db, "embedding_model", "old-model")
        new_ids = [_insert_recording(db) for _ in range(3)]
        db.commit()
        total = db.execute(
            sql_text("SELECT COUNT(*) FROM audio_transcriptions WHERE transcript != ''")
        ).scalar()
        report = rag.reconcile(db, embed=True, max_items=2)
        assert report["indexed"]["audio"] == total
        assert total >= 3
        indexed = {
            r[0]
            for r in db.execute(
                sql_text("SELECT DISTINCT transcription_id FROM audio_transcription_chunks")
            ).all()
        }
        assert set(new_ids) <= indexed
        assert rag.pending_counts(db) == {"audio": 0, "ceo": 0}
        assert get_index_meta(db, "embedding_model") == EMBEDDING_MODEL
