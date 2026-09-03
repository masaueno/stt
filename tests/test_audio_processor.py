"""作業録音の共通処理(services/audio_processor.py)のテスト。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import models  # noqa: E402
from models import AudioTranscription, Base  # noqa: E402
from services import audio_processor  # noqa: E402
from stt_wrapper import STTResult  # noqa: E402


class _FakeStt:
    def __init__(self, text="こんにちは", error=None):
        self._text, self._error = text, error
        self.calls = []

    def transcribe_detailed(self, path):
        self.calls.append(path)
        return STTResult(text=self._text, words=[{"text": "こんにちは", "start": 0.0, "end": 0.5}], error=self._error)


class _FakeStructurer:
    def structure_text(self, text):
        return {"summary": text}

    def extract_tags(self, structured):
        return "会議"


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(models, "SessionLocal", session_factory)
    return session_factory


def _wav(tmp_path, name="sample.wav", seconds=1.0):
    path = tmp_path / name
    sf.write(path, np.zeros(int(16000 * seconds), dtype="float32"), 16000)
    return str(path)


def test_file_path_saves_record_and_cleans_temp(tmp_path, db_session):
    path = _wav(tmp_path)
    res = audio_processor.process_audio_path(
        file_name="sample.wav",
        input_path=path,
        source_kind="file",
        selected_model="Fake",
        stt_wrapper=_FakeStt(),
        text_structurer=_FakeStructurer(),
        use_vad=False,
        vad_aggressiveness=2,
    )
    assert res.status == "ok" and res.record_id is not None
    assert res.transcript == "こんにちは"
    assert res.tags == "会議"
    assert res.structured_json == {"summary": "こんにちは"}
    assert abs(res.duration_seconds - 1.0) < 0.01
    assert not os.path.exists(path)  # cleanup_input=True

    db = db_session()
    try:
        row = db.query(AudioTranscription).one()
        assert row.file_path == "sample.wav"
        assert row.tags == "会議"
        assert row.word_timestamps_json
    finally:
        db.close()


def test_mic_path_keeps_saved_file_and_default_tag(tmp_path, db_session):
    path = _wav(tmp_path, "mic_20260901_000000.wav")
    res = audio_processor.process_audio_path(
        file_name="mic_20260901_000000.wav",
        input_path=path,
        source_kind="mic",
        selected_model="Fake",
        stt_wrapper=_FakeStt(),
        text_structurer=None,
        use_vad=False,
        vad_aggressiveness=2,
        default_tags=audio_processor.DEFAULT_TAGS_MIC,
        duration=3.5,
        save_path=path,
        cleanup_input=False,
    )
    assert res.status == "ok"
    assert res.tags == "マイク録音"
    assert res.duration_seconds == 3.5
    assert os.path.exists(path)


def test_stt_error_does_not_save(tmp_path, db_session):
    path = _wav(tmp_path)
    res = audio_processor.process_audio_path(
        file_name="sample.wav",
        input_path=path,
        source_kind="file",
        selected_model="Fake",
        stt_wrapper=_FakeStt(text=None, error="invalid_api_key"),
        text_structurer=None,
        use_vad=False,
        vad_aggressiveness=2,
    )
    assert res.status == "error"
    assert res.error == "invalid_api_key"
    db = db_session()
    try:
        assert db.query(AudioTranscription).count() == 0
    finally:
        db.close()
