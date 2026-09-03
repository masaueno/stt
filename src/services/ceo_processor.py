"""社長音声（CEO）処理サービス。

stt-desktop の `process_ceo_recording` 相当の処理を Streamlit Web 版向けに移植したもの。

- VAD で非音声区間をカット（既存 services.vad を流用）
- 任意の STT モデル（既定 ElevenLabs）で文字起こし
- `ceo_transcriptions` テーブルに保存
- 同じ source（録音ファイルhash）が既に登録済みなら重複としてスキップ

stt-desktop と同じ DATABASE_URL を共有すれば、両アプリで `ceo_transcriptions` を
透過的に共有できる。
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Optional

from sqlalchemy import or_

from models import CeoTranscription, get_db, utcnow_naive
from services.audio_utils import get_audio_duration, get_audio_duration_metadata
from services.rag_service import get_rag_service
from services.vad import trim_non_speech
from services.word_timestamps import build_word_timestamp_columns
from stt_wrapper import STTModelWrapper


logger = logging.getLogger(__name__)


DEFAULT_CEO_MODEL = "ElevenLabs"
DEFAULT_CEO_SPEAKER = "社長"
DEFAULT_CEO_VAD_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_CEO_VAD_MAX_DURATION_SECONDS = 15 * 60
DEFAULT_CEO_DURATION_DECODE_MAX_BYTES = 25 * 1024 * 1024


@dataclass
class CeoProcessResult:
    """1ファイルの処理結果。"""

    file_name: str
    status: str  # "ok" | "skipped_duplicate" | "error"
    source_kind: str = "mic"
    record_id: Optional[int] = None
    transcript: Optional[str] = None
    duration_seconds: Optional[float] = None
    title: Optional[str] = None
    speaker: Optional[str] = None
    recorded_at: Optional[str] = None
    saved_path: Optional[str] = None
    vad_note: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    matched_existing_id: Optional[int] = None


def _require_configured_database() -> None:
    """社長音声はdesktopと共有するDBが前提なので、暗黙SQLiteへ逃がさない。"""

    if not os.getenv("DATABASE_URL", "").strip():
        raise RuntimeError(
            "社長音声機能では DATABASE_URL が必須です。"
            "Turso/libSQL など desktop と共有するDBを設定してください。"
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s が整数ではないため既定値を使用します: %s", name, raw)
        return default


def _sha256_file(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(file_name: str, source_file_hash: str, input_method: str = "mic") -> str:
    """Web経由の取り込み元を表す疑似パス(重複判定のキー)。"""
    prefix = "mic" if input_method == "mic" else "upload"
    return f"{prefix}:{source_file_hash}:{Path(file_name).name}"


def _append_warning(result: CeoProcessResult, message: str) -> None:
    if result.warning:
        result.warning = f"{result.warning}\n{message}"
    else:
        result.warning = message


def _should_apply_vad(
    *,
    file_path: str,
    size_bytes: Optional[int],
    result: CeoProcessResult,
) -> bool:
    max_bytes = _env_int("CEO_VAD_MAX_BYTES", DEFAULT_CEO_VAD_MAX_BYTES)
    max_seconds = _env_int("CEO_VAD_MAX_DURATION_SECONDS", DEFAULT_CEO_VAD_MAX_DURATION_SECONDS)

    if max_bytes and size_bytes is not None and size_bytes > max_bytes:
        _append_warning(
            result,
            f"VADはスキップしました: ファイルサイズが上限 "
            f"{max_bytes / (1024 * 1024):.0f}MB を超えています。",
        )
        return False

    duration = get_audio_duration_metadata(file_path)
    if max_seconds and duration and duration > max_seconds:
        _append_warning(
            result,
            f"VADはスキップしました: 音声長が上限 {max_seconds / 60:.0f}分を超えています。",
        )
        return False

    return True


def _safe_duration(file_path: str, fallback_path: Optional[str] = None) -> Optional[float]:
    for candidate in (file_path, fallback_path):
        if not candidate:
            continue
        duration = get_audio_duration_metadata(candidate)
        if duration:
            return duration
    for candidate in (file_path, fallback_path):
        if not candidate:
            continue
        try:
            size_bytes = os.path.getsize(candidate)
        except OSError:
            size_bytes = 0
        if size_bytes and size_bytes > _env_int("CEO_DURATION_DECODE_MAX_BYTES", DEFAULT_CEO_DURATION_DECODE_MAX_BYTES):
            continue
        duration = get_audio_duration(candidate)
        if duration:
            return duration
    return None


@dataclass
class CeoBatchSummary:
    """複数ファイルの一括処理サマリ。"""

    results: list[CeoProcessResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped_duplicate")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == "error")


def _strip_vad_suffix(stem: str) -> str:
    """`*_vad` / `*_vad_<n>` のサフィックスを取り除く。"""

    if stem.endswith("_vad"):
        return stem[: -len("_vad")]
    if "_vad_" in stem:
        base, _, tail = stem.rpartition("_vad_")
        if base and tail.isdigit():
            return base
    return stem


def _normalize_path_for_match(raw: Optional[str]) -> Optional[str]:
    value = (raw or "").strip()
    if not value or value.startswith(("http://", "https://", "upload:")):
        return None
    normalized = value.replace("\\", "/")
    if os.name == "nt":
        normalized = normalized.lower()
    return normalized


def _canonical_source_key(raw: Optional[str]) -> Optional[str]:
    normalized = _normalize_path_for_match(raw)
    if not normalized:
        return None
    path = Path(normalized)
    stem = _strip_vad_suffix(path.stem)
    return str(path.parent / stem).replace("\\", "/")


def _parse_ts(value: Optional[str]) -> Optional[float]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _modified_at_matches(expected: Optional[str], actual: Optional[str]) -> bool:
    if expected and not actual:
        return False
    if not expected or not actual:
        return True
    if expected == actual:
        return True
    left = _parse_ts(expected)
    right = _parse_ts(actual)
    if left is None or right is None:
        return False
    return abs(left - right) < 1.0


def _signature_matches(
    candidate: CeoTranscription,
    *,
    size_bytes: Optional[int],
    modified_at: Optional[str],
) -> bool:
    size_match = (
        candidate.source_file_size_bytes is None
        or size_bytes is None
        or candidate.source_file_size_bytes == size_bytes
    )
    return size_match and _modified_at_matches(candidate.source_file_modified_at, modified_at)


def is_generated_vad_file(file_name: str) -> bool:
    """`_vad.wav` / `_vad_<n>.wav` を VAD 生成物として判定。"""

    path = Path(file_name)
    if path.suffix.lower() != ".wav":
        return False
    stem = path.stem
    if stem.endswith("_vad"):
        return True
    if "_vad_" in stem:
        _, _, tail = stem.rpartition("_vad_")
        return tail.isdigit()
    return False


def find_duplicate(
    db,
    *,
    source_file_path: str,
    size_bytes: Optional[int],
    modified_at: Optional[str],
    source_file_hash: Optional[str] = None,
) -> Optional[CeoTranscription]:
    """既に登録済みの社長音声レコードを探す。

    stt-desktop と同じく、source_file_path をキーに size / modified_at が一致する
    レコードを優先する。後方互換のため、local_file_path や file_path の一致も
    フォールバックとして見る。
    """

    if source_file_hash:
        existing_by_hash = (
            db.query(CeoTranscription)
            .filter(CeoTranscription.source_file_hash == source_file_hash)
            .first()
        )
        if existing_by_hash is not None:
            return existing_by_hash

    if not source_file_path:
        return None

    query = db.query(CeoTranscription).filter(
        or_(
            CeoTranscription.source_file_path == source_file_path,
            CeoTranscription.local_file_path == source_file_path,
            CeoTranscription.file_path == source_file_path,
        )
    )

    for candidate in query.all():
        if _signature_matches(candidate, size_bytes=size_bytes, modified_at=modified_at):
            return candidate

    requested_key = _canonical_source_key(source_file_path)
    if requested_key:
        for candidate in db.query(CeoTranscription).all():
            for legacy_path in (candidate.local_file_path, candidate.file_path):
                if _canonical_source_key(legacy_path) == requested_key:
                    return candidate

    return None


def _resolve_vad_output_dir() -> Path:
    """アップロード経由（元フォルダが分からない場合）の VAD 後 wav の保存先ディレクトリ。

    desktop 版は「元音声と同じフォルダ」に出すが、Web アップロード経由ではフルパスが
    取得できないため、desktop の既定保存先と同じ `~/Downloads` を使う。
    `CEO_VAD_OUTPUT_DIR` でサーバー側パスを上書き可。
    """

    raw = os.getenv("CEO_VAD_OUTPUT_DIR", "").strip()
    if raw:
        out_dir = Path(raw).expanduser()
    else:
        out_dir = Path.home() / "Downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_vad_output(source_vad_path: str, base_file_name: str) -> str:
    """VAD 出力ファイルを `CEO_VAD_OUTPUT_DIR` 配下に重複しない名前で保存し、
    保存先パスを返す。
    """

    out_dir = _resolve_vad_output_dir()
    stem = _strip_vad_suffix(Path(base_file_name).stem) or "ceo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for attempt in range(1000):
        suffix = "_vad" if attempt == 0 else f"_vad_{attempt}"
        candidate = out_dir / f"{stem}_{timestamp}{suffix}.wav"
        if not candidate.exists():
            try:
                shutil.move(source_vad_path, candidate)
            except Exception:
                # 同一 FS でない場合はコピー → 一時ファイルは _process_single 側で削除
                shutil.copyfile(source_vad_path, candidate)
                try:
                    os.unlink(source_vad_path)
                except OSError:
                    pass
            return str(candidate)
    raise RuntimeError("VAD 出力ファイル名を決定できませんでした")


def process_ceo_uploaded_path(
    *,
    file_name: str,
    temp_file_path: str,
    title: Optional[str],
    speaker: Optional[str],
    recorded_at: Optional[str],
    source_file_size_bytes: Optional[int] = None,
    source_file_modified_at: Optional[str] = None,
    source_file_hash: Optional[str] = None,
    selected_model: str = DEFAULT_CEO_MODEL,
    use_vad: bool = True,
    vad_aggressiveness: int = 2,
    cleanup_source: bool = False,
    tags: str = "社長音声",
    input_method: str = "mic",
) -> CeoProcessResult:
    """ブラウザのマイク録音・ファイル読み込みの一時ファイルを処理して
    `ceo_transcriptions` に保存する。

    `st.audio_input` / `st.file_uploader` の戻り値を session_state にbytesで
    保持しないため、UI側で一時ファイルへ退避したパスを受け取る。
    `tags` はカテゴリのタグ(社長音声 / 業務記録)、`input_method` は
    "mic"(マイク録音) / "file_import"(ファイル読み込み)。
    """

    _require_configured_database()
    src = Path(temp_file_path)
    if source_file_size_bytes is None:
        try:
            source_file_size_bytes = int(src.stat().st_size)
        except OSError:
            source_file_size_bytes = None
    if source_file_hash is None:
        source_file_hash = _sha256_file(str(src))
    source_file_path = _source_key(file_name, source_file_hash, input_method)
    tags = (tags or "社長音声").strip() or "社長音声"

    title = (title or Path(file_name).stem or tags).strip()
    speaker = (speaker or DEFAULT_CEO_SPEAKER).strip() or DEFAULT_CEO_SPEAKER
    recorded_at = (recorded_at or "").strip() or None

    result = CeoProcessResult(
        file_name=file_name,
        status="error",
        source_kind="mic" if input_method == "mic" else "file",
        title=title,
        speaker=speaker,
        recorded_at=recorded_at,
    )

    # 1. VAD 生成物として明らかなファイル名は弾く
    if is_generated_vad_file(file_name):
        result.status = "skipped_duplicate"
        result.warning = "VAD 生成物（*_vad.wav）と判定したためスキップしました。"
        if cleanup_source and src.exists():
            try:
                src.unlink()
            except OSError:
                pass
        return result

    # 2. 重複判定
    db = next(get_db())
    try:
        existing = find_duplicate(
            db,
            source_file_path=source_file_path,
            size_bytes=source_file_size_bytes,
            modified_at=source_file_modified_at,
            source_file_hash=source_file_hash,
        )
        if existing is not None:
            result.status = "skipped_duplicate"
            result.matched_existing_id = existing.id
            result.record_id = existing.id
            result.transcript = existing.transcript
            result.duration_seconds = existing.duration_seconds
            if cleanup_source and src.exists():
                try:
                    src.unlink()
                except OSError:
                    pass
            return result
    finally:
        db.close()

    vad_path: Optional[str] = None
    saved_vad_path: Optional[str] = None
    try:
        original_duration: Optional[float] = None

        # 3. VAD（任意）
        stt_input_path = str(src)
        vad_applied = False  # STT入力がVADトリム後音声か
        vad_kept_ranges = None
        if use_vad and _should_apply_vad(
            file_path=str(src),
            size_bytes=source_file_size_bytes,
            result=result,
        ):
            try:
                vad_res = trim_non_speech(
                    str(src),
                    enabled=True,
                    aggressiveness=int(vad_aggressiveness),
                )
                vad_path = vad_res.output_path
                stt_input_path = vad_path
                vad_applied = True
                vad_kept_ranges = vad_res.kept_ranges
                original_duration = vad_res.orig_sec
                if vad_res.orig_sec > 0:
                    reduced = max(0.0, 1.0 - (vad_res.out_sec / vad_res.orig_sec)) * 100.0
                    result.vad_note = (
                        f"VAD: {vad_res.orig_sec:.2f}s → {vad_res.out_sec:.2f}s "
                        f"(−{reduced:.1f}%) [{vad_res.method}]"
                    )
                else:
                    result.vad_note = f"VAD: 入力長 0 秒 [{vad_res.method}]"
            except Exception as exc:
                logger.warning("CEO VAD 前処理に失敗したためスキップ: %s", exc)
                _append_warning(result, f"VAD 前処理に失敗したため元音声を使用します: {exc}")
                stt_input_path = str(src)
                vad_applied = False
                vad_kept_ranges = None

        # 4. VAD ファイルはマイク録音用の保存先（CEO_VAD_OUTPUT_DIR）にコピー保存
        if vad_path and os.path.exists(vad_path):
            try:
                saved_vad_path = _save_vad_output(vad_path, file_name)
                # 移動済みなので stt_input_path も更新
                stt_input_path = saved_vad_path
                vad_path = None  # cleanup 対象から外す
            except Exception as exc:
                logger.warning("CEO VAD 出力の保存に失敗（一時ファイルのまま継続）: %s", exc)

        # 5. STT
        wrapper = STTModelWrapper(selected_model)
        stt_result = wrapper.transcribe_detailed(stt_input_path)
        transcription = stt_result.text
        error_msg: Optional[str] = stt_result.error
        if error_msg:
            transcription = None

        if not transcription:
            result.status = "error"
            result.error = error_msg or "文字起こし結果が空でした"
            return result

        duration = original_duration or _safe_duration(str(src), stt_input_path)

        # 単語タイムスタンプ: VAD後基準(STTが返したまま)と、VAD保持区間から
        # 元音声基準へ復元した値の両方を保存する(desktopの作業録音と同方針)
        word_ts, word_ts_original = build_word_timestamp_columns(
            stt_result.words,
            vad_applied=vad_applied,
            vad_ranges=vad_kept_ranges,
        )

        # 6. DB 保存
        db = next(get_db())
        try:
            saved_path_for_db = saved_vad_path or file_name
            record = CeoTranscription(
                file_path=saved_path_for_db,
                local_file_path=saved_vad_path,
                source_file_path=source_file_path,
                source_file_size_bytes=source_file_size_bytes,
                source_file_modified_at=source_file_modified_at,
                source_file_hash=source_file_hash,
                source_app="web",
                input_method=input_method,
                title=title,
                speaker=speaker,
                recorded_at=recorded_at,
                model_id=selected_model,
                language_code=None,
                transcript=transcription,
                structured_json=None,
                duration_seconds=duration,
                tags=tags,
                # naive UTCで統一(日付判定はUTC解釈のため。models.utcnow_naive参照)
                created_at=utcnow_naive(),
                word_timestamps_json=word_ts,
                word_timestamps_original_json=word_ts_original,
            )
            db.add(record)
            db.flush()

            # 保存と同時に検索インデックスへ登録(失敗しても保存自体は成立させる)
            rag = get_rag_service()
            if rag.enabled:
                try:
                    rag.index_ceo_transcription(db, record.id, transcription)
                except Exception as exc:
                    logger.error("社長音声のRAG索引作成に失敗: %s", exc, exc_info=True)

            db.commit()
            db.refresh(record)
            result.record_id = record.id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        result.status = "ok"
        result.transcript = transcription
        result.duration_seconds = duration
        result.saved_path = saved_vad_path
        return result

    except Exception as exc:
        logger.exception("CEO 音声処理で例外: %s", exc)
        result.status = "error"
        result.error = str(exc)
        return result

    finally:
        # 一時ファイルのクリーンアップ（保存済み VAD は成功時だけ残す）
        cleanup_paths = [vad_path]
        if cleanup_source:
            cleanup_paths.append(str(src))
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if result.status != "ok" and saved_vad_path and os.path.exists(saved_vad_path):
            try:
                os.unlink(saved_vad_path)
            except OSError:
                pass
