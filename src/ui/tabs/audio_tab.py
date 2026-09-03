"""作業録音タブ。

マイク録音とファイル読み込みの両方を `audio_transcriptions` に保存する。
処理本体は services/audio_processor.py に共通化してあり、ここは入力・表示のみ。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from services.audio_processor import (
    DEFAULT_TAGS_FILE,
    DEFAULT_TAGS_MIC,
    AudioProcessResult,
    prepare_mic_recording,
    process_audio_path,
)
from services.ceo_time_utils import created_at_to_jst
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from ui.categories import AUDIO
from ui.components import (
    QueueItem,
    render_file_input,
    render_mic_input,
    render_queue,
    result_card,
    should_process_recording,
)

_PREFIX = "audio"
_START_LABEL = "文字起こし開始"


def _state() -> dict:
    return st.session_state.setdefault(
        "audio_tab_state",
        {"queue": [], "results": []},  # queue: QueueItem / results: AudioProcessResult(成功分)
    )


def _vad_settings() -> tuple[bool, int]:
    app_settings = st.session_state.get("settings")
    use_vad = bool(getattr(app_settings, "get_use_vad", lambda: True)())
    vad_aggr = int(getattr(app_settings, "get_vad_aggressiveness", lambda: 2)())
    return use_vad, vad_aggr


def _init_engines(selected_model: str, use_structuring: bool):
    try:
        return STTModelWrapper(selected_model), (TextStructurer() if use_structuring else None)
    except Exception as exc:
        st.error(f"初期化エラー: {exc}")
        return None, None


def _show_error(res: AudioProcessResult) -> None:
    st.error(f"❌ {res.file_name} の文字起こしに失敗しました")
    low = (res.error or "").lower()
    if "invalid_api_key" in low:
        st.error("🔑 APIキーが無効です")
        st.info("💡 別のSTTモデルに切り替えるか、APIキーを確認してください。")
    elif "internal server error" in low:
        st.error("🔧 サーバーで一時的な問題が発生しました")
        st.info("💡 数分後に再試行するか、別のSTTモデルに切り替えてください。")
    elif res.error:
        st.error(f"エラー詳細: {res.error}")


def _record(res: AudioProcessResult) -> None:
    state = _state()
    if res.status == "ok":
        memo = "完了" if not res.warning else f"完了（{res.warning}）"
        state["results"].append(res)
    else:
        memo = res.error or "処理に失敗しました"
        _show_error(res)
    state["queue"].append(
        QueueItem(status=res.status, kind=res.source_kind, file_name=res.file_name, memo=memo)
    )
    if res.vad_note:
        st.info(res.vad_note)
    if res.warning:
        st.warning(res.warning)


# ---------- 入力 ----------

def _handle_mic(selected_model: str, use_structuring: bool, logger) -> None:
    audio_value = render_mic_input(f"{_PREFIX}_mic_audio")
    if not audio_value:
        return
    go, digest = should_process_recording(audio_value, _PREFIX)
    if not go:
        return

    stt_wrapper, text_structurer = _init_engines(selected_model, use_structuring)
    if stt_wrapper is None:
        return
    use_vad, vad_aggr = _vad_settings()
    st.session_state[f"{_PREFIX}_mic_processing"] = True
    try:
        with st.spinner("文字起こし中..."):
            prep = prepare_mic_recording(audio_value, selected_model, logger)
            res = process_audio_path(
                file_name=prep.file_name,
                input_path=prep.input_path,
                source_kind="mic",
                selected_model=selected_model,
                stt_wrapper=stt_wrapper,
                text_structurer=text_structurer,
                use_vad=use_vad,
                vad_aggressiveness=vad_aggr,
                default_tags=DEFAULT_TAGS_MIC,
                duration=prep.duration,
                created_at=prep.created_at,
                save_path=prep.save_path,
                save_to_r2=prep.settings.save_to_r2,
                cleanup_input=prep.save_path is None,
                logger=logger,
            )
        _record(res)
        if res.status == "ok":
            st.success("✅ 文字起こし完了")
    except Exception as exc:
        st.error(f"マイク録音処理エラー: {exc}")
        logger.error("マイク録音処理エラー: %s", exc, exc_info=True)
    finally:
        st.session_state[f"{_PREFIX}_mic_last_digest"] = digest
        st.session_state[f"{_PREFIX}_mic_processing"] = False


def _handle_files(selected_model: str, use_structuring: bool, logger) -> None:
    files, clicked = render_file_input(_PREFIX, _START_LABEL)
    if not clicked or not files:
        return

    stt_wrapper, text_structurer = _init_engines(selected_model, use_structuring)
    if stt_wrapper is None:
        return
    use_vad, vad_aggr = _vad_settings()

    progress = st.progress(0.0)
    status = st.empty()
    total = len(files)
    for idx, uf in enumerate(files):
        status.text(f"処理中: {uf.name} ({idx + 1}/{total})")
        logger.info("処理開始: %s", uf.name)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            res = process_audio_path(
                file_name=uf.name,
                input_path=tmp_path,
                source_kind="file",
                selected_model=selected_model,
                stt_wrapper=stt_wrapper,
                text_structurer=text_structurer,
                use_vad=use_vad,
                vad_aggressiveness=vad_aggr,
                default_tags=DEFAULT_TAGS_FILE,
                logger=logger,
            )
            _record(res)
        except Exception as exc:
            st.error(f"処理エラー ({uf.name}): {exc}")
            logger.error("処理エラー (%s): %s", uf.name, exc, exc_info=True)
        progress.progress((idx + 1) / total)
    status.text("✅ すべての処理が完了しました")


# ---------- 処理結果 ----------

def _format_created_at(value) -> str:
    dt = created_at_to_jst(value)
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "-"


def _render_storage_info(res: AudioProcessResult) -> None:
    st.markdown("**保存情報**")
    if res.save_path:
        st.caption(f"ローカル: {res.save_path}")
    if res.r2_ok:
        st.caption(f"R2: s3://{res.r2_bucket}/{res.r2_key}")
        if res.r2_url:
            st.caption(f"公開URL: {res.r2_url}")


def _render_results(results: list[AudioProcessResult], latest_status: Optional[str]) -> None:
    with result_card(latest_status):
        if not results:
            st.caption("ここに結果が表示されます。")
            return
        for idx, res in enumerate(results):
            with st.expander(f"📁 {res.file_name}", expanded=(idx == len(results) - 1)):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**録音日時:** {_format_created_at(res.created_at)}")
                    st.write(f"**長さ:** {res.duration_seconds:.1f}秒")
                    st.write(f"**タグ:** {res.tags or '-'}")
                    st.markdown("**文字起こし**")
                    st.text_area(
                        "文字起こし",
                        res.transcript or "",
                        height=200,
                        key=f"{_PREFIX}_result_text_{idx}",
                        label_visibility="collapsed",
                    )
                with col2:
                    if res.structured_json:
                        st.markdown("**構造化データ**")
                        st.json(res.structured_json)
                    else:
                        st.caption("構造化データなし")
                    if res.source_kind == "mic":
                        _render_storage_info(res)


# ---------- main ----------

def run_audio_tab(selected_model: str, use_structuring: bool, logger) -> None:
    st.header(AUDIO.label)
    st.caption(AUDIO.description)

    _handle_mic(selected_model, use_structuring, logger)
    _handle_files(selected_model, use_structuring, logger)

    state = _state()
    render_queue(state["queue"])
    latest_status = state["queue"][-1].status if state["queue"] else None
    _render_results(state["results"], latest_status)
