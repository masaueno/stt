from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)

    raw = str(value).strip()
    if not raw:
        return None

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def timestamp_seconds(value: Any) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.timestamp()


def ceo_record_sort_key(record: Any) -> tuple[bool, float, int]:
    timestamp = timestamp_seconds(getattr(record, "recorded_at", None))
    if timestamp is None:
        timestamp = timestamp_seconds(getattr(record, "created_at", None))
    return (timestamp is None, -(timestamp or 0), -(getattr(record, "id", 0) or 0))


def _aware_as_jst_naive(value: Any) -> tuple[str, datetime | None]:
    if value is None:
        return "-", None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return "-", None
    else:
        raw = str(value)

    parsed = parse_timestamp(value)
    if parsed is None:
        return raw, None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return raw, None
    return raw, parsed.astimezone(JST).replace(tzinfo=None)


def format_recorded_at_for_web(value: Any) -> str:
    raw, converted = _aware_as_jst_naive(value)
    if converted is None:
        return raw
    return converted.strftime("%Y-%m-%dT%H:%M:%S")


def format_created_at_for_web(value: Any) -> str:
    raw, converted = _aware_as_jst_naive(value)
    if converted is not None:
        return converted.strftime("%Y-%m-%d %H:%M:%S")

    parsed = parse_timestamp(value)
    if parsed is None:
        return raw
    return parsed.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def to_jst_naive(value: Any, *, naive_tz: timezone = JST) -> datetime | None:
    """任意形式のタイムスタンプをJSTのnaive datetimeへ変換する。TZなしは naive_tz とみなす。"""
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=naive_tz)
    return parsed.astimezone(JST).replace(tzinfo=None)


def recorded_at_to_jst(value: Any) -> datetime | None:
    """recorded_at(TZなしはデスクトップ版のローカル時刻=JST)をJSTへ。"""
    return to_jst_naive(value, naive_tz=JST)


def created_at_to_jst(value: Any) -> datetime | None:
    """created_at(TZなしはUTC。models.utcnow_naive参照)をJSTへ。"""
    return to_jst_naive(value, naive_tz=timezone.utc)
