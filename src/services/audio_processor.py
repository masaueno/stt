"""作業録音(audio_transcriptions)の処理サービス。

マイク録音・ファイル読み込みの両経路で共通の
VAD → STT → 構造化 → 単語タイムスタンプ → DB保存 → RAG索引 → (任意) R2アップロード
を1関数にまとめる。UI(ui/tabs/audio_tab.py)は一時ファイルの用意と表示だけを担う。
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa

from models import AudioTranscription, get_db, utcnow_naive
from services.audio_utils import (
    convert_webm_to_wav,
    get_audio_duration,
    should_convert_to_wav,
)
from services.cloudflare_r2 import load_r2_config_from_env, upload_file_to_r2
from services.rag_service import get_rag_service
from services.vad import trim_non_speech
from services.word_timestamps import build_word_timestamp_columns
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer

_module_logger = logging.getLogger(__name__)

DEFAULT_TAGS_FILE = "未分類"
DEFAULT_TAGS_MIC = "マイク録音"


@dataclass
class MicStorageSettings:
    save_local: bool
    save_dir: str
    save_to_r2: bool


def mic_storage_settings() -> MicStorageSettings:
    return MicStorageSettings(
        save_local=os.getenv("SAVE_MIC_AUDIO_LOCAL", "true").lower() == "true",
        save_dir=os.getenv("MIC_AUDIO_SAVE_DIR", "data/recordings"),
        save_to_r2=os.getenv("SAVE_MIC_AUDIO_TO_R2", "false").lower() == "true",
    )


@dataclass
class AudioProcessResult:
    """1ファイルの処理結果(処理キュー・処理結果の表示に使う)。"""

    file_name: str
    status: str = "error"          # ok | error
    source_kind: str = "file"      # mic | file
    record_id: Optional[int] = None
    created_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    transcript: Optional[str] = None
    structured_json: Optional[dict] = None
    tags: Optional[str] = None
    save_path: Optional[str] = None
    r2_url: Optional[str] = None
    r2_bucket: Optional[str] = None
    r2_key: Optional[str] = None
    r2_ok: bool = False
    vad_note: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PreparedMicRecording:
    """マイク録音を一時ファイル化(必要ならWAV変換・ローカル保存)した結果。"""

    file_name: str
    input_path: str
    duration: float
    created_at: datetime
    save_path: Optional[str]
    settings: MicStorageSettings


def prepare_mic_recording(
    audio_value,
    selected_model: str,
    logger: logging.Logger = _module_logger,
) -> PreparedMicRecording:
    """`st.audio_input` の録音を処理用の一時ファイルにする。

    - 必要なモデルのみ WebM → WAV(16kHz) へ変換
    - SAVE_MIC_AUDIO_LOCAL=true なら MIC_AUDIO_SAVE_DIR に `mic_YYYYMMDD_HHMMSS.*` で保存
    """
    settings = mic_storage_settings()
    raw = audio_value.getvalue() if hasattr(audio_value, "getvalue") else audio_value
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(raw)
        webm_path = tmp.name
    logger.info("マイク録音処理開始: %s", webm_path)

    tmp_path = webm_path
    duration = 0.0
    if should_convert_to_wav(selected_model):
        try:
            wav_path, duration = convert_webm_to_wav(webm_path, target_sr=16000)
            os.unlink(webm_path)
            tmp_path = wav_path
            logger.info("音声変換完了: WebM → WAV (%s)", wav_path)
        except Exception as exc:
            duration = get_audio_duration(webm_path)
            logger.warning("音声変換失敗（WebMで処理継続）: %s", exc)
    else:
        duration = get_audio_duration(tmp_path)

    # created_at・ファイル名ともnaive UTCで統一(日付判定はUTC解釈のため)
    created_at = utcnow_naive()
    ext = ".wav" if tmp_path.endswith(".wav") else ".webm"
    file_name = f"mic_{created_at.strftime('%Y%m%d_%H%M%S')}{ext}"

    save_path: Optional[str] = None
    if settings.save_local:
        try:
            Path(settings.save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(settings.save_dir) / file_name)
            os.replace(tmp_path, save_path)
            tmp_path = save_path
            logger.info("ローカル保存: %s", save_path)
        except Exception as exc:
            logger.warning("ローカル保存に失敗（処理は継続）: %s", exc)
            save_path = None

    return PreparedMicRecording(
        file_name=file_name,
        input_path=tmp_path,
        duration=duration,
        created_at=created_at,
        save_path=save_path,
        settings=settings,
    )


def _upload_to_r2(
    result: AudioProcessResult,
    *,
    stt_input_path: str,
    vad_applied: bool,
    fallback_path: str,
    logger: logging.Logger,
) -> None:
    cfg = load_r2_config_from_env()
    if cfg is None:
        logger.error("R2設定が不足しています。R2_* 環境変数を確認してください。")
        return
    try:
        # VAD ONならトリム後を優先してアップロード
        if vad_applied and os.path.exists(stt_input_path):
            source_path = stt_input_path
            key = f"{Path(result.file_name).stem}_vad.wav"
        else:
            source_path = fallback_path
            key = result.file_name
        info = upload_file_to_r2(source_path, key, cfg)
        result.r2_ok = True
        result.r2_url = info.get("url")
        result.r2_bucket = info.get("bucket")
        result.r2_key = info.get("key")
        logger.info("R2アップロード成功: s3://%s/%s", info.get("bucket"), info.get("key"))
    except Exception as exc:
        logger.error("R2アップロード失敗: %s", exc)


def process_audio_path(
    *,
    file_name: str,
    input_path: str,
    source_kind: str,
    selected_model: str,
    stt_wrapper: STTModelWrapper,
    text_structurer: Optional[TextStructurer],
    use_vad: bool,
    vad_aggressiveness: int,
    default_tags: str = DEFAULT_TAGS_FILE,
    duration: Optional[float] = None,
    created_at: Optional[datetime] = None,
    save_path: Optional[str] = None,
    save_to_r2: bool = False,
    cleanup_input: bool = True,
    logger: logging.Logger = _module_logger,
) -> AudioProcessResult:
    """音声ファイル1件を文字起こしして `audio_transcriptions` に保存する。

    `logger` には画面側のロガー(logs/streamlit_app.log へ出力)を渡す。
    """

    result = AudioProcessResult(
        file_name=file_name,
        source_kind=source_kind,
        save_path=save_path,
        created_at=created_at or utcnow_naive(),
    )
    stt_input_path = input_path
    try:
        if duration is None:
            audio_data, sr = librosa.load(input_path, sr=None)
            duration = len(audio_data) / sr
            logger.debug("音声ファイル情報: 時間=%.2f秒, サンプリングレート=%sHz", duration, sr)
        result.duration_seconds = duration

        # VAD前処理(任意)
        vad_applied = False
        vad_kept_ranges = None
        if use_vad:
            try:
                vad_res = trim_non_speech(input_path, enabled=True, aggressiveness=vad_aggressiveness)
                stt_input_path = vad_res.output_path
                vad_applied = True
                vad_kept_ranges = vad_res.kept_ranges
                reduced = 0.0
                if vad_res.orig_sec > 0:
                    reduced = max(0.0, 1.0 - (vad_res.out_sec / vad_res.orig_sec)) * 100.0
                result.vad_note = (
                    f"VAD有効: 元{vad_res.orig_sec:.2f}s → 送信{vad_res.out_sec:.2f}s "
                    f"(−{reduced:.1f}%) [{vad_res.method}]"
                )
                logger.info(result.vad_note)
            except Exception as exc:
                logger.warning("VAD前処理に失敗したためスキップ: %s", exc)
                result.warning = "VAD前処理に失敗したため、元音声を使用します。"
                stt_input_path = input_path

        # STT
        logger.info("文字起こし実行中: %s (モデル: %s)", file_name, selected_model)
        stt_result = stt_wrapper.transcribe_detailed(stt_input_path)
        transcription = stt_result.text
        error_msg = stt_result.error
        if error_msg:
            transcription = None
            logger.error("文字起こしエラー: %s", error_msg)
        if not transcription:
            result.error = error_msg or "文字起こし結果が空でした"
            logger.error("文字起こし失敗: %s, %s", file_name, result.error)
            return result

        # 構造化
        structured_data = None
        tags = default_tags
        if text_structurer is not None:
            structured_data = text_structurer.structure_text(transcription)
            if structured_data:
                tags = text_structurer.extract_tags(structured_data)

        # 単語タイムスタンプ: STTが返したまま(VAD後基準)と、
        # VAD保持区間から元音声基準へ復元した値の両方を保存する
        word_ts, word_ts_original = build_word_timestamp_columns(
            stt_result.words,
            vad_applied=vad_applied,
            vad_ranges=vad_kept_ranges,
        )

        if save_to_r2:
            _upload_to_r2(
                result,
                stt_input_path=stt_input_path,
                vad_applied=vad_applied,
                fallback_path=save_path or input_path,
                logger=logger,
            )

        db = next(get_db())
        try:
            record = AudioTranscription(
                file_path=file_name,
                created_at=result.created_at,
                duration_seconds=duration,
                transcript=transcription,
                structured_json=structured_data,
                tags=tags,
                word_timestamps_json=word_ts,
                word_timestamps_original_json=word_ts_original,
            )
            db.add(record)
            db.flush()

            rag_service = get_rag_service()
            if rag_service.enabled:
                try:
                    rag_service.index_transcription(db, record.id, transcription)
                except Exception as exc:  # pragma: no cover - API例外
                    logger.error("RAG埋め込みの生成に失敗: %s", exc, exc_info=True)

            db.commit()
            result.record_id = record.id
            logger.info("文字起こし結果をデータベースに保存: %s (ID: %s)", file_name, record.id)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        result.status = "ok"
        result.transcript = transcription
        result.structured_json = structured_data
        result.tags = tags
        return result

    except Exception as exc:
        logger.error("処理エラー (%s): %s", file_name, exc, exc_info=True)
        result.error = str(exc)
        return result

    finally:
        if cleanup_input:
            try:
                os.unlink(input_path)
                logger.debug("一時ファイル削除: %s", input_path)
            except Exception:
                pass
        if stt_input_path != input_path:
            try:
                os.unlink(stt_input_path)
                logger.debug("VAD一時ファイル削除: %s", stt_input_path)
            except Exception:
                pass
