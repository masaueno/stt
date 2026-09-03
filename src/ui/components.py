"""画面間で共有するUI部品。

録音3画面(作業録音 / 社長音声 / 業務記録)、データベース、AIチャットで
同じ見た目・同じ文言を保つため、セクション見出し・入力欄・処理キュー・
処理結果カード・カテゴリ選択・期間指定はすべてここから描画する。
"""

from __future__ import annotations

import html
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

from ui.categories import CATEGORY_KEYS, CATEGORY_LABELS, Category

# ---------- 固定文言 ----------

LABEL_MIC = "マイク録音"
LABEL_FILE = "ファイルを読み込む（複数可）"
LABEL_IMPORT_INFO = "取り込み情報"
LABEL_QUEUE = "処理キュー"
LABEL_RESULT = "処理結果"
LABEL_LOAD = "データを読み込む"
LABEL_RELOAD = "再読み込み"
LABEL_DELETE = "削除"

AUDIO_FILE_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]

# 処理キューの「種別」
KIND_LABELS = {"mic": "マイク録音", "file": "ファイル"}

# 処理キュー・処理結果の状態表示(文言と色)
_STATUS = {
    "processing": ("処理中", "#2563eb", "#e8f0fe"),
    "waiting": ("待機", "#6b7280", "#f3f4f6"),
    "ok": ("完了", "#16a34a", "#e8f8ef"),
    "skipped_duplicate": ("重複", "#6b7280", "#f3f4f6"),
    "error": ("失敗", "#dc2626", "#fdecea"),
}


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


# ---------- セクションカード ----------

@contextmanager
def section_card(title: str, right_html: Optional[str] = None):
    """枠付きコンテナ + 見出し(左) + 任意の右寄せ表示(カウンタ・バッジ)。"""
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"### {title}")
        with right:
            if right_html:
                st.markdown(
                    f"<div style='text-align:right;padding-top:14px;'>{right_html}</div>",
                    unsafe_allow_html=True,
                )
        yield


# ---------- 入力部品 ----------

def render_import_info(category: Category, key_prefix: str) -> Tuple[str, str, str]:
    """取り込み情報(タイトル / 話者 / 録音日時)。戻り値はいずれも strip 済み。"""
    st.markdown(f"**{LABEL_IMPORT_INFO}**")
    c_title, c_speaker, c_time = st.columns([2, 1, 2])
    with c_title:
        title = st.text_input(
            "タイトル",
            value="",
            key=f"{key_prefix}_title",
            placeholder="空欄ならファイル名",
        )
    with c_speaker:
        speaker = st.text_input(
            "話者",
            value=category.default_speaker,
            key=f"{key_prefix}_speaker",
        )
    with c_time:
        recorded_at = st.text_input(
            "録音日時",
            value="",
            key=f"{key_prefix}_recorded_at",
            placeholder="空欄なら録音完了時刻",
            help="ISO 8601形式。例: 2026-05-21T10:00:00",
        )
    return title.strip(), speaker.strip(), recorded_at.strip()


def render_mic_input(key: str, *, sample_rate: Optional[int] = None):
    """マイク録音。停止すると呼び出し側が自動で処理する。"""
    kwargs = {"sample_rate": sample_rate} if sample_rate else {}
    return st.audio_input(
        LABEL_MIC,
        key=key,
        help="停止すると自動で処理を開始します。",
        **kwargs,
    )


def render_file_input(key_prefix: str, button_label: str, *, disabled: bool = False):
    """ファイルを読み込む（複数可）+ 開始ボタン。戻り値は (files, clicked)。"""
    files = st.file_uploader(
        LABEL_FILE,
        type=AUDIO_FILE_TYPES,
        accept_multiple_files=True,
        key=f"{key_prefix}_files",
    )
    clicked = st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=disabled or not files,
        key=f"{key_prefix}_start",
    )
    return files or [], clicked


def audio_digest(audio_value) -> Optional[str]:
    try:
        raw = audio_value.getvalue() if hasattr(audio_value, "getvalue") else audio_value
        return sha256(raw).hexdigest()
    except Exception:
        return None


def should_process_recording(audio_value, key_prefix: str) -> Tuple[bool, Optional[str]]:
    """新しい録音なら (True, digest)。処理済みの録音は再処理ボタンを押したときだけ True。"""
    digest = audio_digest(audio_value)
    if not digest:
        st.error("録音データの確認に失敗しました。もう一度録音してください。")
        return False, None
    if st.session_state.get(f"{key_prefix}_mic_processing"):
        return False, digest
    if st.session_state.get(f"{key_prefix}_mic_last_digest") != digest:
        return True, digest
    retry = st.button("同じ録音を再処理", key=f"{key_prefix}_mic_retry")
    return retry, digest


# ---------- 処理キュー ----------

@dataclass
class QueueItem:
    status: str          # processing / waiting / ok / skipped_duplicate / error
    kind: str            # mic / file
    file_name: str
    memo: str = ""


def render_queue(items: Sequence[QueueItem]) -> None:
    counts = {k: 0 for k in ("processing", "waiting", "ok", "error")}
    for it in items:
        if it.status in counts:
            counts[it.status] += 1
    counter = (
        "<span style='font-size:12px;color:#555;'>"
        f"処理中: {counts['processing']} ・ 待機: {counts['waiting']} ・ "
        f"完了: {counts['ok']} ・ 失敗: {counts['error']}</span>"
    )
    with section_card(LABEL_QUEUE, counter):
        if not items:
            st.caption("まだ処理した録音はありません。")
            return
        rows = []
        for it in items:
            label, color, _ = _STATUS.get(it.status, ("-", "#555", "#f3f4f6"))
            rows.append(
                "<tr style='border-bottom:1px solid #f0f2f5;'>"
                f"<td style='padding:6px 8px;color:{color};white-space:nowrap;'>● {_esc(label)}</td>"
                f"<td style='padding:6px 8px;white-space:nowrap;'>{_esc(KIND_LABELS.get(it.kind, it.kind))}</td>"
                f"<td style='padding:6px 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:320px;' title='{_esc(it.file_name)}'>{_esc(it.file_name)}</td>"
                f"<td style='padding:6px 8px;color:#555;'>{_esc(it.memo)}</td>"
                "</tr>"
            )
        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr style='background:#f4f6f8;'>"
            "<th style='text-align:left;padding:6px 8px;width:80px;'>状態</th>"
            "<th style='text-align:left;padding:6px 8px;width:100px;'>種別</th>"
            "<th style='text-align:left;padding:6px 8px;'>ファイル名</th>"
            "<th style='text-align:left;padding:6px 8px;'>メモ</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
            unsafe_allow_html=True,
        )


# ---------- 処理結果 ----------

@contextmanager
def result_card(status: Optional[str]):
    """処理結果カード。status が None なら「待機中」バッジ。本文は呼び出し側で描画する。"""
    if status is None:
        text, color, bg = "待機中", "#666", "#eee"
    else:
        text, color, bg = _STATUS.get(status, ("-", "#555", "#f3f4f6"))
    badge = (
        f"<span style='font-size:12px;padding:2px 10px;border-radius:999px;"
        f"background:{bg};color:{color};'>{_esc(text)}</span>"
    )
    with section_card(LABEL_RESULT, badge):
        yield


# ---------- カテゴリ選択・期間指定(データベース / AIチャット共通) ----------

def render_category_pills(key: str, label: str = "カテゴリ") -> Tuple[str, ...]:
    """カテゴリのトグル(既定は全選択)。戻り値は正規順のキー。"""
    selected = st.pills(
        label,
        options=list(CATEGORY_KEYS),
        format_func=CATEGORY_LABELS.get,
        selection_mode="multi",
        default=list(CATEGORY_KEYS),
        key=key,
        label_visibility="collapsed",
    )
    return tuple(k for k in CATEGORY_KEYS if k in (selected or ()))


def render_date_range(
    key_prefix: str,
    *,
    unset_label: str,
    clear_label: str,
    caption: Optional[str] = None,
) -> Optional[Tuple[date, date]]:
    """期間(開始日 / 終了日)のポップオーバー。適用済みなら (start, end) を返す。"""
    from_key, to_key = f"{key_prefix}_date_from", f"{key_prefix}_date_to"
    d_from = st.session_state.get(from_key)
    d_to = st.session_state.get(to_key)
    label = f"📅 {d_from} 〜 {d_to}" if (d_from and d_to) else unset_label
    with st.popover(label, use_container_width=True):
        if caption:
            st.caption(caption)
        c1, c2 = st.columns(2)
        with c1:
            d_from = st.date_input("開始日", value=d_from, key=f"{from_key}_input", format="YYYY-MM-DD")
        with c2:
            d_to = st.date_input("終了日", value=d_to, key=f"{to_key}_input", format="YYYY-MM-DD")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("適用", use_container_width=True, key=f"{key_prefix}_date_apply"):
                st.session_state[from_key] = d_from
                st.session_state[to_key] = d_to
                st.rerun()
        with b2:
            if st.button(clear_label, use_container_width=True, key=f"{key_prefix}_date_clear"):
                st.session_state[from_key] = None
                st.session_state[to_key] = None
                st.rerun()
    if st.session_state.get(from_key) and st.session_state.get(to_key):
        return st.session_state[from_key], st.session_state[to_key]
    return None


def cleanup_paths(paths: Iterable[Optional[str]]) -> None:
    """一時ファイルをまとめて削除する(失敗は無視)。"""
    for p in paths:
        if not p:
            continue
        try:
            if os.path.isdir(p):
                os.rmdir(p)
            else:
                os.unlink(p)
        except Exception:
            pass
