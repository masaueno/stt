"""録音カテゴリの定義(正本)。

UIラベル・タグ・保存先テーブルはここだけで定義し、全画面から参照する。
デスクトップ版(`stt-desktop/src/features/categories.ts`)と同じ内容。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# デスクトップ版・Web版とも ceo_transcriptions.tags に書き込む業務記録タグ。
WORK_RECORD_TAG = "業務記録"
CEO_TAG = "社長音声"


@dataclass(frozen=True)
class Category:
    key: str                 # audio / ceo / work
    label: str               # 作業録音 / 社長音声 / 業務記録
    table: str               # 保存先テーブル
    tag: Optional[str]       # ceo_transcriptions.tags に書く値(作業録音は自動タグのためNone)
    default_speaker: str     # 取り込み情報「話者」の既定値
    description: str         # 録音画面の1行説明
    file_prefix: str         # マイク録音ファイル名のprefix


AUDIO = Category(
    key="audio",
    label="作業録音",
    table="audio_transcriptions",
    tag=None,
    default_speaker="",
    description="録音の停止または「文字起こし開始」で、文字起こしと構造化を行い保存します。",
    file_prefix="mic",
)
CEO = Category(
    key="ceo",
    label="社長音声",
    table="ceo_transcriptions",
    tag=CEO_TAG,
    default_speaker="社長",
    description="録音の停止または「取り込み開始」で、無音カット後に文字起こしして保存します。",
    file_prefix="ceo_mic",
)
WORK = Category(
    key="work",
    label="業務記録",
    table="ceo_transcriptions",
    tag=WORK_RECORD_TAG,
    default_speaker="",
    description="録音の停止または「取り込み開始」で、無音カット後に文字起こしして保存します。",
    file_prefix="work_mic",
)

CATEGORIES: dict[str, Category] = {c.key: c for c in (AUDIO, CEO, WORK)}
CATEGORY_KEYS: tuple[str, ...] = tuple(CATEGORIES)
CATEGORY_LABELS: dict[str, str] = {k: c.label for k, c in CATEGORIES.items()}


def category_label(key: str) -> str:
    return CATEGORY_LABELS.get(key, key)


def ceo_category_key(tags: Optional[str]) -> str:
    """ceo_transcriptions の行を tags でカテゴリ判定する(業務記録以外は社長音声)。"""
    return WORK.key if (tags or "").strip() == WORK_RECORD_TAG else CEO.key
