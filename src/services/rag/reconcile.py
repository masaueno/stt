"""RAG索引の再同期処理。

Web版・デスクトップ版とも保存時にchunks/FTSを作成するが、保存時の索引失敗や
旧バージョンで保存した録音の取りこぼしを防ぐため、AIチャットを開いたときに
差分を検出して索引を補完する(デスクトップ版にも同じ処理 rag/indexer.rs がある)。
あわせて recorded_date の正規化と孤児データの掃除も行う。埋め込み生成
(OpenAI API)が必要な処理は rag_service側で本モジュールの検出結果を使って実行する。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from models import RAG_FTS_TABLES
from services.rag.date_utils import JST  # noqa: F401  (後方互換の再エクスポート)
from services.rag.tokenizer import to_fts_text

logger = logging.getLogger(__name__)

# 小数秒が6桁を超えるとdatetime.fromisoformatが失敗するため丸める
_FRACTION_RE = re.compile(r"(\.\d{1,6})\d*")

_CHUNK_TABLES = {
    "audio": ("audio_transcription_chunks", "audio_transcriptions"),
    "ceo": ("ceo_transcription_chunks", "ceo_transcriptions"),
}


def normalize_to_jst_date(value) -> Optional[str]:
    """created_at/recorded_at をJSTの 'YYYY-MM-DD' に正規化する。

    - naive文字列/naive datetimeはUTCとみなす(本番のWeb版・デスクトップ版とも
      UTC時刻で書き込んでいることを実データの時間帯分布で確認済み)
    - TZ付き(+00:00 / Z)はそのままJSTへ変換
    """
    if value is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        s = s.replace("Z", "+00:00")
        s = _FRACTION_RE.sub(r"\1", s)
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # 先頭10文字が日付形式ならそれを使う
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
            if m:
                return m.group(0)
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST).strftime("%Y-%m-%d")


def fill_recorded_dates(db: Session) -> Dict[str, int]:
    """recorded_date が未設定の行を正規化して埋める。"""
    filled = {"audio": 0, "ceo": 0}

    rows = db.execute(
        text("SELECT id, created_at FROM audio_transcriptions WHERE recorded_date IS NULL OR recorded_date = ''")
    ).mappings().all()
    for r in rows:
        d = normalize_to_jst_date(r["created_at"])
        if d:
            db.execute(
                text("UPDATE audio_transcriptions SET recorded_date = :d WHERE id = :id"),
                {"d": d, "id": r["id"]},
            )
            filled["audio"] += 1

    rows = db.execute(
        text(
            "SELECT id, recorded_at, created_at FROM ceo_transcriptions "
            "WHERE recorded_date IS NULL OR recorded_date = ''"
        )
    ).mappings().all()
    for r in rows:
        d = normalize_to_jst_date(r["recorded_at"]) or normalize_to_jst_date(r["created_at"])
        if d:
            db.execute(
                text("UPDATE ceo_transcriptions SET recorded_date = :d WHERE id = :id"),
                {"d": d, "id": r["id"]},
            )
            filled["ceo"] += 1

    return filled


def sync_fts(db: Session) -> Dict[str, int]:
    """チャンクとFTSの差分を同期する(追加・孤児削除)。"""
    report = {"added": 0, "removed": 0}
    for source, (chunk_table, _parent) in _CHUNK_TABLES.items():
        fts = RAG_FTS_TABLES[source]

        missing = db.execute(
            text(
                f"SELECT c.id, c.chunk_text FROM {chunk_table} c "
                f"WHERE c.id NOT IN (SELECT rowid FROM {fts})"
            )
        ).mappings().all()
        for r in missing:
            db.execute(
                text(f"INSERT INTO {fts}(rowid, tokens) VALUES (:id, :tokens)"),
                {"id": r["id"], "tokens": to_fts_text(r["chunk_text"] or "")},
            )
        report["added"] += len(missing)

        orphans = db.execute(
            text(f"SELECT rowid AS id FROM {fts} WHERE rowid NOT IN (SELECT id FROM {chunk_table})")
        ).mappings().all()
        for r in orphans:
            db.execute(text(f"DELETE FROM {fts} WHERE rowid = :id"), {"id": r["id"]})
        report["removed"] += len(orphans)

    return report


def cleanup_orphan_chunks(db: Session) -> int:
    """親レコードが削除されたチャンクを掃除する(FK未強制環境対策)。"""
    removed = 0
    for source, (chunk_table, parent) in _CHUNK_TABLES.items():
        fts = RAG_FTS_TABLES[source]
        orphan_ids = [
            r["id"]
            for r in db.execute(
                text(
                    f"SELECT c.id AS id FROM {chunk_table} c "
                    f"WHERE c.transcription_id NOT IN (SELECT id FROM {parent})"
                )
            ).mappings().all()
        ]
        for cid in orphan_ids:
            db.execute(text(f"DELETE FROM {fts} WHERE rowid = :id"), {"id": cid})
            db.execute(text(f"DELETE FROM {chunk_table} WHERE id = :id"), {"id": cid})
        removed += len(orphan_ids)
    return removed


def get_index_meta(db: Session, key: str) -> Optional[str]:
    """索引メタ情報(埋め込みモデル名等)を読む。テーブル未作成ならNone。"""
    try:
        row = db.execute(
            text("SELECT value FROM rag_index_meta WHERE key = :key"), {"key": key}
        ).first()
        return row[0] if row else None
    except Exception:
        return None


def set_index_meta(db: Session, key: str, value: str) -> None:
    """索引メタ情報を保存する(テーブルは必要時に作成)。"""
    db.execute(
        text("CREATE TABLE IF NOT EXISTS rag_index_meta (key TEXT PRIMARY KEY, value TEXT)")
    )
    db.execute(
        text("INSERT OR REPLACE INTO rag_index_meta(key, value) VALUES (:key, :value)"),
        {"key": key, "value": value},
    )


def find_unindexed(db: Session, embedding_model: Optional[str] = None) -> Dict[str, List[int]]:
    """チャンク未作成の文字起こしID一覧(埋め込み生成が必要な差分)。

    embedding_modelを渡した場合、索引時のモデル(rag_index_metaに記録)と
    不一致なら全録音を再索引対象として返す。埋め込みモデル変更時に旧ベクトルと
    新ベクトルが混在すると距離計算が壊れるため、全量再作成へ誘導する。
    """
    model_changed = (
        embedding_model is not None
        and get_index_meta(db, "embedding_model") != embedding_model
    )
    result: Dict[str, List[int]] = {}
    for source, (chunk_table, parent) in _CHUNK_TABLES.items():
        if model_changed:
            rows = db.execute(
                text(
                    f"SELECT t.id AS id FROM {parent} t "
                    f"WHERE t.transcript IS NOT NULL AND t.transcript != '' ORDER BY t.id"
                )
            ).mappings().all()
        else:
            rows = db.execute(
                text(
                    f"SELECT t.id AS id FROM {parent} t "
                    f"WHERE t.transcript IS NOT NULL AND t.transcript != '' "
                    f"AND NOT EXISTS (SELECT 1 FROM {chunk_table} c WHERE c.transcription_id = t.id) "
                    f"ORDER BY t.id"
                )
            ).mappings().all()
        result[source] = [r["id"] for r in rows]
    return result
