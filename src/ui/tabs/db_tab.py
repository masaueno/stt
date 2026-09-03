"""データベースタブ。

作業録音(audio_transcriptions)と社長音声 / 業務記録(ceo_transcriptions)を
1つの一覧で表示・絞り込み・詳細表示・削除する。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from sqlalchemy import delete

from models import AudioTranscription, CeoTranscription, get_db
from services.ceo_time_utils import JST, created_at_to_jst, recorded_at_to_jst
from services.cloudflare_r2 import (
    build_object_key_for_filename,
    build_public_url_for_key,
    generate_presigned_get_url,
    load_r2_config_from_env,
    object_exists_in_r2,
)
from ui.categories import AUDIO, category_label, ceo_category_key
from ui.components import (
    LABEL_DELETE,
    LABEL_LOAD,
    LABEL_RELOAD,
    render_category_pills,
    render_date_range,
)

SOURCE_APP_LABELS = {"web": "Web版", "desktop": "アプリ版"}
INPUT_METHOD_LABELS = {"mic": "マイク録音", "file_import": "ファイル読み込み"}
_COLUMNS = ["ID", "カテゴリ", "録音日時", "タイトル", "話者", "長さ(s)", "タグ", "テキスト"]
_PREVIEW_LEN = 50
_ALL = "すべて"


def _label(value, labels: dict[str, str]) -> str:
    raw = str(value or "").strip()
    if not raw or raw == "unknown":
        return "-"
    return labels.get(raw, raw)


def _file_name(path: Optional[str]) -> str:
    """パス末尾のファイル名(Windows/Unix両区切りに対応)。"""
    import re
    parts = re.split(r"[\\/]", path or "")
    return parts[-1] if parts and parts[-1] else ""


def _fmt(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "-"


def _state() -> dict:
    st.session_state.setdefault("r2_exists_cache", {})
    return st.session_state.setdefault("db_tab_state", {"loaded": False, "records": []})


# ---------- 読み込み ----------

def _r2_download_url(file_path: str, r2_cfg, signed_exp: int, cache: dict) -> Optional[str]:
    if r2_cfg is None:
        return None
    key = build_object_key_for_filename(file_path, r2_cfg)
    if not key:
        return None
    exists = cache.get(key)
    if exists is None:
        exists = object_exists_in_r2(key, r2_cfg)
        cache[key] = exists
    if not exists:
        return None
    return build_public_url_for_key(key, r2_cfg) or generate_presigned_get_url(
        key, expires_in=signed_exp, cfg=r2_cfg
    )


def _audio_row(record: AudioTranscription) -> dict:
    created = created_at_to_jst(record.created_at)
    return {
        "id": record.id,
        "category": AUDIO.key,
        "dt": created,
        "recorded_at": _fmt(created),
        "created_at": _fmt(created),
        "recorded_date": record.recorded_date or (created.date().isoformat() if created else None),
        "title": _file_name(record.file_path) or "-",
        "speaker": "-",
        "duration": record.duration_seconds,
        "tags": record.tags or "",
        "transcript": record.transcript or "",
        "structured_json": record.structured_json,
        "model": record.model_id,
        "language": record.language_code,
        "source_app": "-",
        "input_method": "-",
        "file": record.file_path,
        "local_path": None,
        # R2の存在確認は行数分のHEADリクエストになるため、詳細表示時に遅延評価する
        "cloud_url": None,
        "download_url": None,
        "_r2_checked": False,
    }


def _ceo_row(record: CeoTranscription) -> dict:
    recorded = recorded_at_to_jst(record.recorded_at)
    created = created_at_to_jst(record.created_at)
    dt = recorded or created
    file_path = record.file_path or ""
    return {
        "id": record.id,
        "category": ceo_category_key(record.tags),
        "dt": dt,
        "recorded_at": _fmt(dt),
        "created_at": _fmt(created),
        "recorded_date": record.recorded_date or (dt.date().isoformat() if dt else None),
        "title": record.title or (_file_name(record.source_file_path or file_path) or "-"),
        "speaker": record.speaker or "-",
        "duration": record.duration_seconds,
        "tags": record.tags or "",
        "transcript": record.transcript or "",
        "structured_json": record.structured_json,
        "model": record.model_id,
        "language": record.language_code,
        "source_app": _label(record.source_app, SOURCE_APP_LABELS),
        "input_method": _label(record.input_method, INPUT_METHOD_LABELS),
        "file": record.source_file_path or file_path or None,
        "local_path": record.local_file_path,
        "cloud_url": file_path if file_path.startswith(("http://", "https://")) else None,
        "download_url": None,
    }


def _sort_key(row: dict):
    dt = row["dt"]
    ts = dt.replace(tzinfo=JST).timestamp() if dt else None
    return (ts is None, -(ts or 0), -(row["id"] or 0))


def _load_records() -> list[dict]:
    db = next(get_db())
    try:
        audio_records = db.query(AudioTranscription).all()
        ceo_records = db.query(CeoTranscription).all()
    finally:
        db.close()

    rows = [_audio_row(r) for r in audio_records]
    rows.extend(_ceo_row(r) for r in ceo_records)
    rows.sort(key=_sort_key)
    return rows


def _delete_record(category: str, record_id: int) -> None:
    model = AudioTranscription if category == AUDIO.key else CeoTranscription
    db = next(get_db())
    try:
        db.execute(delete(model).where(model.id == record_id))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------- 絞り込み ----------

def _apply_filters(records, categories, date_range, tag, speaker, keyword) -> list[dict]:
    start = date_range[0].isoformat() if date_range else None
    end = date_range[1].isoformat() if date_range else None
    q = keyword.strip().lower()
    out = []
    for r in records:
        if r["category"] not in categories:
            continue
        if start and end:
            d = r["recorded_date"]
            if not d or d < start or d > end:
                continue
        if tag != _ALL and r["tags"] != tag:
            continue
        if speaker != _ALL and r["speaker"] != speaker:
            continue
        if q and not any(
            q in str(v or "").lower()
            for v in (r["title"], r["speaker"], r["file"], r["transcript"])
        ):
            continue
        out.append(r)
    return out


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": r["id"],
                "カテゴリ": category_label(r["category"]),
                "録音日時": r["recorded_at"],
                "タイトル": r["title"],
                "話者": r["speaker"],
                "長さ(s)": r["duration"],
                "タグ": r["tags"] or "-",
                "テキスト": r["transcript"][:_PREVIEW_LEN]
                + ("…" if len(r["transcript"]) > _PREVIEW_LEN else ""),
            }
            for r in rows
        ],
        columns=_COLUMNS,
    )


# ---------- 詳細 ----------

def _resolve_audio_download_url(rec: dict) -> None:
    """作業録音のR2ダウンロードURLを詳細表示時にだけ解決する(結果は行に保持)。"""
    if rec["category"] != AUDIO.key or rec.get("_r2_checked"):
        return
    r2_cfg = load_r2_config_from_env()
    signed_exp = int(os.getenv("R2_SIGNED_URL_EXPIRES", "900"))
    url = _r2_download_url(rec["file"] or "", r2_cfg, signed_exp, st.session_state["r2_exists_cache"])
    rec["download_url"] = url
    rec["cloud_url"] = url
    rec["_r2_checked"] = True


def _render_detail(rec: dict, state: dict) -> None:
    _resolve_audio_download_url(rec)
    st.subheader(f"ID: {rec['id']}")
    duration = f"{rec['duration']:.1f} 秒" if rec["duration"] is not None else "-"
    left = [
        ("カテゴリ", category_label(rec["category"])),
        ("録音日時", rec["recorded_at"]),
        ("登録日時", rec["created_at"]),
        ("タイトル", rec["title"]),
        ("話者", rec["speaker"]),
        ("長さ", duration),
        ("タグ", rec["tags"]),
    ]
    right = [
        ("モデル", rec["model"]),
        ("言語", rec["language"]),
        ("登録元", rec["source_app"]),
        ("取り込み方法", rec["input_method"]),
        ("ファイル", rec["file"]),
        ("ローカルパス", rec["local_path"]),
        ("クラウドURL", rec["cloud_url"]),
    ]
    c1, c2 = st.columns(2)
    for col, items in ((c1, left), (c2, right)):
        with col:
            for label, value in items:
                st.write(f"**{label}:** {value or '-'}")

    st.markdown("**文字起こし**")
    st.text_area(
        "文字起こし",
        rec["transcript"],
        height=240,
        key=f"db_detail_text_{rec['category']}_{rec['id']}",
        label_visibility="collapsed",
    )
    if rec["structured_json"]:
        st.markdown("**構造化データ**")
        st.json(rec["structured_json"])
    if rec["download_url"]:
        st.link_button("Cloudflare R2 からダウンロード", rec["download_url"])

    with st.expander(f"⚠️ {LABEL_DELETE}", expanded=False):
        confirm = st.checkbox(
            "削除に同意します（取り消し不可）",
            key=f"db_confirm_delete_{rec['category']}_{rec['id']}",
        )
        if st.button(
            LABEL_DELETE,
            disabled=not confirm,
            key=f"db_delete_{rec['category']}_{rec['id']}",
        ):
            _delete_record(rec["category"], rec["id"])
            state["loaded"] = False
            st.success(f"ID {rec['id']} を削除しました。")
            st.rerun()


# ---------- main ----------

def run_db_tab() -> None:
    st.header("データベース")
    state = _state()

    c_load, c_count = st.columns([1, 4])
    with c_load:
        if st.button(LABEL_RELOAD if state["loaded"] else LABEL_LOAD, key="db_tab_load"):
            state["records"] = _load_records()
            state["loaded"] = True
    with c_count:
        if state["loaded"]:
            st.caption(f"{len(state['records'])} 件")

    if not state["loaded"]:
        st.info(f"「{LABEL_LOAD}」でデータベースを表示します。")
        return

    records: list[dict] = state["records"]
    if not records:
        st.info("レコードがありません。")
        return

    # --- フィルタ: カテゴリ → 期間 → タグ / 話者 / キーワード ---
    categories = render_category_pills("db_categories")
    tags = [_ALL] + sorted({r["tags"] for r in records if r["tags"]})
    speakers = [_ALL] + sorted({r["speaker"] for r in records if r["speaker"] and r["speaker"] != "-"})
    f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.6])
    with f1:
        date_range = render_date_range("db", unset_label="📅 期間: すべて", clear_label="解除")
    with f2:
        tag = st.selectbox("タグ", tags, key="db_tag_filter")
    with f3:
        speaker = st.selectbox("話者", speakers, key="db_speaker_filter")
    with f4:
        keyword = st.text_input(
            "キーワード",
            key="db_keyword_filter",
            placeholder="タイトル / 話者 / ファイル名 / 文字起こし",
        )

    if not categories:
        st.info("カテゴリを1つ以上選択してください。")
        return

    filtered = _apply_filters(records, categories, date_range, tag, speaker, keyword)
    event = st.dataframe(
        _to_frame(filtered),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="db_table",
    )
    if not filtered:
        return

    selected_rows = event.selection.rows if event and event.selection else []
    if not selected_rows or selected_rows[0] >= len(filtered):
        st.caption("行を選ぶと詳細を表示します。")
        return
    _render_detail(filtered[selected_rows[0]], state)
