"""社長音声 / 業務記録の処理(tags・input_method の引数化)のテスト。"""

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
from models import Base, CeoTranscription  # noqa: E402
from services import ceo_processor  # noqa: E402
from stt_wrapper import STTResult  # noqa: E402


class _FakeWrapper:
    def __init__(self, model):
        self.model = model

    def transcribe_detailed(self, path):
        return STTResult(text="業務の内容", words=None, error=None)


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(models, "SessionLocal", session_factory)
    monkeypatch.setattr(ceo_processor, "STTModelWrapper", _FakeWrapper)
    return session_factory


def _wav(tmp_path, name):
    path = tmp_path / name
    sf.write(path, np.zeros(16000, dtype="float32"), 16000)
    return str(path)


def _run(tmp_path, name, speaker=None, **kwargs):
    return ceo_processor.process_ceo_uploaded_path(
        file_name=name,
        temp_file_path=_wav(tmp_path, name),
        title=None,
        speaker=speaker,
        recorded_at="2026-09-01T10:00:00",
        selected_model="Fake",
        use_vad=False,
        cleanup_source=True,
        **kwargs,
    )


def test_default_is_ceo_mic(tmp_path, db_session):
    res = _run(tmp_path, "ceo_mic_1.wav")
    assert res.status == "ok" and res.source_kind == "mic"
    db = db_session()
    try:
        row = db.query(CeoTranscription).one()
        assert row.tags == "社長音声"
        assert row.input_method == "mic"
        assert row.source_app == "web"
        assert row.speaker == "社長"
        assert row.source_file_path.startswith("mic:")
    finally:
        db.close()


def test_work_file_import(tmp_path, db_session):
    res = _run(tmp_path, "meeting.wav", tags="業務記録", input_method="file_import", speaker="山田")
    assert res.status == "ok" and res.source_kind == "file"
    assert res.title == "meeting"
    db = db_session()
    try:
        row = db.query(CeoTranscription).one()
        assert row.tags == "業務記録"
        assert row.input_method == "file_import"
        assert row.source_app == "web"
        assert row.speaker == "山田"
        assert row.source_file_path.startswith("upload:")
    finally:
        db.close()


def test_same_file_is_skipped_as_duplicate(tmp_path, db_session):
    first = _run(tmp_path, "dup.wav", tags="業務記録", input_method="file_import")
    second = _run(tmp_path, "dup.wav", tags="業務記録", input_method="file_import")
    assert first.status == "ok"
    assert second.status == "skipped_duplicate"
    assert second.matched_existing_id == first.record_id
