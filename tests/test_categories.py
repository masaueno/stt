"""カテゴリ定義(ui/categories.py)と用語統一のテスト。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.rag import prompt_builder  # noqa: E402
from services.rag.search_service import VALID_SOURCES, WORK_RECORD_TAG  # noqa: E402
from ui.categories import (  # noqa: E402
    AUDIO,
    CATEGORIES,
    CATEGORY_KEYS,
    CATEGORY_LABELS,
    CEO,
    WORK,
    ceo_category_key,
)


def test_keys_and_labels_match_spec():
    assert CATEGORY_KEYS == ("audio", "ceo", "work")
    assert CATEGORY_LABELS == {"audio": "作業録音", "ceo": "社長音声", "work": "業務記録"}
    assert set(CATEGORY_KEYS) == set(VALID_SOURCES)


def test_tables_and_tags():
    assert AUDIO.table == "audio_transcriptions" and AUDIO.tag is None
    assert CEO.table == WORK.table == "ceo_transcriptions"
    assert WORK.tag == WORK_RECORD_TAG
    assert CEO.default_speaker == "社長" and WORK.default_speaker == ""


def test_ceo_category_key_by_tags():
    assert ceo_category_key(None) == "ceo"
    assert ceo_category_key("") == "ceo"
    assert ceo_category_key("社長音声") == "ceo"
    assert ceo_category_key("会議") == "ceo"
    assert ceo_category_key(WORK_RECORD_TAG) == "work"


def test_prompt_labels_use_unified_terms():
    assert prompt_builder._SOURCE_LABELS == CATEGORY_LABELS
    for key in CATEGORIES:
        assert "現場" not in prompt_builder._CORPUS_DESCRIPTIONS[key]
    assert "現場" not in prompt_builder.build_system_prompt(corpus=("audio", "ceo", "work"))
