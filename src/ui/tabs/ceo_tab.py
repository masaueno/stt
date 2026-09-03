"""社長音声 / 業務記録タブ(共用)。

カテゴリ(ui.categories.CEO / WORK)を引数に取り、マイク録音とファイル読み込みを
`ceo_transcriptions` に保存する。カテゴリの違いは tags・話者の既定値・
session_state のキーだけで、画面構成は同一。
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Optional

import streamlit as st

from services.ceo_processor import (
    DEFAULT_CEO_MODEL,
    CeoProcessResult,
    process_ceo_uploaded_path,
)
from services.ceo_time_utils import recorded_at_to_jst
from ui.categories import Category
from ui.components import (
    QueueItem,
    cleanup_paths,
    render_file_input,
    render_import_info,
    render_mic_input,
    render_queue,
    result_card,
    should_process_recording,
)

_START_LABEL = "取り込み開始"
DEFAULT_VAD_ENABLED = True
DEFAULT_VAD_AGGRESSIVENESS = 2


# ---------- state ----------

def _state(category: Category) -> dict:
    return st.session_state.setdefault(
        f"ceo_tab_state_{category.key}",
        {"results": [], "active_idx": 0},  # results: CeoProcessResult(全件) / active_idx: 処理結果で表示する結果
    )


def _vad_settings() -> tuple[bool, int]:
    app_settings = st.session_state.get("settings")
    use_vad = bool(getattr(app_settings, "get_use_vad", lambda: DEFAULT_VAD_ENABLED)())
    vad_aggr = int(getattr(app_settings, "get_vad_aggressiveness", lambda: DEFAULT_VAD_AGGRESSIVENESS)())
    return use_vad, vad_aggr


# ---------- 一時ファイル ----------

def _suffix_from_audio_upload(uf, default: str = ".wav") -> str:
    name_suffix = Path(getattr(uf, "name", "") or "").suffix
    if name_suffix:
        return name_suffix
    content_type = (getattr(uf, "type", "") or "").lower()
    if "wav" in content_type:
        return ".wav"
    if "webm" in content_type:
        return ".webm"
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    if "ogg" in content_type:
        return ".ogg"
    return default


def _persist_uploaded_file(
    uf,
    *,
    source_kind: str,
    file_name_override: Optional[str] = None,
    temp_prefix: str = "stt_ceo_",
) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_prefix))
    file_name = Path(file_name_override or getattr(uf, "name", "") or "uploaded_audio").name
    suffix = Path(file_name).suffix or ".audio"
    temp_path = temp_dir / f"source{suffix}"
    digest = sha256()
    size = 0

    if hasattr(uf, "seek"):
        uf.seek(0)
    with open(temp_path, "wb") as out:
        while True:
            chunk = uf.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)

    return {
        "source_kind": source_kind,
        "temp_file_path": str(temp_path),
        "temp_dir": str(temp_dir),
        "file_name": file_name,
        "size_bytes": size,
        "modified_at": None,
        "recorded_at": None,
        "source_file_hash": digest.hexdigest(),
    }


def _persist_mic_recording(audio_value, category: Category) -> dict:
    timestamp = datetime.now()
    suffix = _suffix_from_audio_upload(audio_value)
    file_name = f"{category.file_prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}{suffix}"
    entry = _persist_uploaded_file(
        audio_value,
        source_kind="mic",
        file_name_override=file_name,
        temp_prefix=f"stt_{category.key}_mic_",
    )
    entry["modified_at"] = timestamp.isoformat(timespec="seconds")
    entry["recorded_at"] = timestamp.isoformat(timespec="seconds")
    return entry


# ---------- 処理 ----------

def _process_entries(
    files: list[dict],
    *,
    category: Category,
    title_override: str,
    speaker: str,
    recorded_at_override: Optional[str],
    selected_model: str,
    logger,
) -> list[CeoProcessResult]:
    results: list[CeoProcessResult] = []
    total = len(files)
    if total == 0:
        return results
    progress = st.progress(0.0)
    status = st.empty()
    use_vad, vad_aggressiveness = _vad_settings()

    for idx, f in enumerate(files):
        status.text(f"処理中: {f['file_name']} ({idx + 1}/{total})")
        effective_recorded_at = (
            recorded_at_override
            or f.get("recorded_at")
            or f.get("modified_at")
            or datetime.now().isoformat(timespec="seconds")
        )
        logger.info(
            "%s 処理開始 (%s): name=%s size=%s hash=%s",
            category.label, f["source_kind"], f["file_name"], f["size_bytes"], f.get("source_file_hash"),
        )
        result = process_ceo_uploaded_path(
            file_name=f["file_name"],
            temp_file_path=f["temp_file_path"],
            title=title_override or Path(f["file_name"]).stem,
            speaker=speaker or category.default_speaker,
            recorded_at=effective_recorded_at,
            source_file_size_bytes=f["size_bytes"],
            source_file_modified_at=f.get("modified_at"),
            source_file_hash=f.get("source_file_hash"),
            selected_model=selected_model,
            use_vad=use_vad,
            vad_aggressiveness=vad_aggressiveness,
            cleanup_source=True,
            tags=category.tag,
            input_method="mic" if f["source_kind"] == "mic" else "file_import",
        )
        results.append(result)
        progress.progress((idx + 1) / total)

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped_duplicate")
    err = sum(1 for r in results if r.status == "error")
    status.text(f"完了: 成功 {ok} / 重複スキップ {skipped} / 失敗 {err}")
    return results


def _run_batch(
    files: list[dict],
    *,
    category: Category,
    import_info: tuple[str, str, str],
    selected_model: str,
    logger,
) -> None:
    state = _state(category)
    title, speaker, recorded_at = import_info
    try:
        with st.spinner(f"{category.label}として文字起こし中..."):
            results = _process_entries(
                files,
                category=category,
                title_override=title,
                speaker=speaker,
                recorded_at_override=recorded_at or None,
                selected_model=selected_model or DEFAULT_CEO_MODEL,
                logger=logger,
            )
        state["results"].extend(results)
        state["active_idx"] = max(0, len(state["results"]) - 1)
        ok = sum(1 for r in results if r.status == "ok")
        skipped = sum(1 for r in results if r.status == "skipped_duplicate")
        err = sum(1 for r in results if r.status == "error")
        if ok:
            st.success(f"{category.label}として保存しました。")
        if skipped:
            st.info("既に保存済みの音声は重複としてスキップしました。")
        if err:
            st.error("取り込みに失敗した音声があります。処理結果を確認してください。")
    except Exception as exc:
        st.error(f"{category.label}の取り込みに失敗しました: {exc}")
        logger.exception("%s processing failed", category.key)
    finally:
        cleanup_paths([f.get("temp_file_path") for f in files])
        cleanup_paths([f.get("temp_dir") for f in files])


def _handle_mic(category: Category, import_info, selected_model: str, logger) -> None:
    prefix = f"{category.key}"
    audio_value = render_mic_input(f"{prefix}_mic_audio", sample_rate=16000)
    if not audio_value:
        return
    go, digest = should_process_recording(audio_value, prefix)
    if not go:
        return

    st.session_state[f"{prefix}_mic_processing"] = True
    files: list[dict] = []
    try:
        files = [_persist_mic_recording(audio_value, category)]
        first = files[0]
        logger.info(
            "%s 録音受領: name=%s size=%s type=%s hash=%s",
            category.label, first["file_name"], first["size_bytes"],
            getattr(audio_value, "type", ""), first["source_file_hash"],
        )
        _run_batch(files, category=category, import_info=import_info, selected_model=selected_model, logger=logger)
    except Exception as exc:
        st.error(f"録音データの保存に失敗しました: {exc}")
        logger.exception("%s mic recording failed", category.key)
        cleanup_paths([f.get("temp_file_path") for f in files])
        cleanup_paths([f.get("temp_dir") for f in files])
    finally:
        st.session_state[f"{prefix}_mic_last_digest"] = digest
        st.session_state[f"{prefix}_mic_processing"] = False


def _handle_files(category: Category, import_info, selected_model: str, logger) -> None:
    uploads, clicked = render_file_input(category.key, _START_LABEL)
    if not clicked or not uploads:
        return
    files: list[dict] = []
    try:
        files = [
            _persist_uploaded_file(uf, source_kind="file", temp_prefix=f"stt_{category.key}_file_")
            for uf in uploads
        ]
    except Exception as exc:
        st.error(f"ファイルの読み込みに失敗しました: {exc}")
        logger.exception("%s file import failed", category.key)
        cleanup_paths([f.get("temp_file_path") for f in files])
        cleanup_paths([f.get("temp_dir") for f in files])
        return
    _run_batch(files, category=category, import_info=import_info, selected_model=selected_model, logger=logger)


# ---------- 処理キュー / 処理結果 ----------

def _queue_items(results: list[CeoProcessResult]) -> list[QueueItem]:
    items = []
    for r in results:
        if r.status == "error":
            memo = r.error or "処理に失敗しました"
        elif r.status == "skipped_duplicate":
            memo = f"既存ID: {r.matched_existing_id}" if r.matched_existing_id else "重複"
        else:
            memo = "完了"
        items.append(QueueItem(status=r.status, kind=r.source_kind, file_name=r.file_name, memo=memo))
    return items


def _format_duration(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f} 秒"
    except Exception:
        return str(value)


def _format_recorded_at(value: Optional[str]) -> str:
    dt = recorded_at_to_jst(value)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else (value or "-")


def _render_result_panel(category: Category) -> None:
    state = _state(category)
    results: list[CeoProcessResult] = state["results"]
    active: Optional[CeoProcessResult] = None
    active_idx = 0
    if results:
        active_idx = max(0, min(state.get("active_idx", 0), len(results) - 1))
        active = results[active_idx]

    with result_card(active.status if active else None):
        if len(results) > 1:
            selected = st.selectbox(
                "結果を選択",
                options=list(range(len(results))),
                index=active_idx,
                format_func=lambda i: f"{i + 1}. {results[i].file_name}",
                key=f"{category.key}_active_result_select",
            )
            if selected != active_idx:
                state["active_idx"] = selected
                st.rerun()

        cols = st.columns(3)
        with cols[0]:
            st.markdown("**ファイル**")
            st.write(active.file_name if active else "-")
            st.markdown("**タイトル**")
            st.write(active.title if active and active.title else "-")
            st.markdown("**話者**")
            st.write(active.speaker if active and active.speaker else "-")
        with cols[1]:
            st.markdown("**録音日時**")
            st.write(_format_recorded_at(active.recorded_at) if active else "-")
            st.markdown("**長さ**")
            st.write(_format_duration(active.duration_seconds) if active else "-")
            st.markdown("**DB保存**")
            if active is None:
                st.write("-")
            elif active.status == "ok":
                st.write(f"✅ 完了 (ID: {active.record_id})")
            elif active.status == "skipped_duplicate":
                st.write(f"⏭️ 既存 (ID: {active.matched_existing_id or '-'})")
            else:
                st.write("未保存")
        with cols[2]:
            st.markdown("**VAD保存先**")
            st.write(active.saved_path if active and active.saved_path else "-")

        if active and active.vad_note:
            st.caption(active.vad_note)
        if active and active.warning:
            st.warning(active.warning)
        if active and active.error:
            st.error(active.error)

        st.markdown("**文字起こし**")
        st.text_area(
            "文字起こし",
            value=(active.transcript or "") if active else "",
            height=180,
            placeholder="ここに結果が表示されます",
            key=f"{category.key}_result_text_{active_idx if active else 'placeholder'}",
            label_visibility="collapsed",
        )


# ---------- main ----------

def run_ceo_tab(category: Category, selected_model: str, logger) -> None:
    st.header(category.label)
    st.caption(category.description)

    import_info = render_import_info(category, category.key)
    model = selected_model or DEFAULT_CEO_MODEL
    _handle_mic(category, import_info, model, logger)
    _handle_files(category, import_info, model, logger)

    render_queue(_queue_items(_state(category)["results"]))
    _render_result_panel(category)
