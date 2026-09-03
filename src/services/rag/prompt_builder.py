"""回答生成用プロンプトの組み立て(Responses API形式)。"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence, Union

from services.rag.context_builder import ContextDoc
from services.rag.date_utils import jst_today

# ui/categories.py のラベルと揃える(services層はuiに依存しないため重複定義)
_SOURCE_LABELS = {"audio": "作業録音", "ceo": "社長音声", "work": "業務記録"}


_CORPUS_DESCRIPTIONS = {
    "audio": "作業録音を文字起こしした「音声DB」",
    "ceo": "社長が録音した音声メモ・打ち合わせを文字起こしした「社長音声DB」",
    "work": "業務内容を録音した「業務記録」",
}

# corpusはソースキー1つ(str)または複数(検索ソースの組)を受け取る
Corpus = Union[str, Sequence[str]]


def _corpus_description(corpus: Corpus) -> str:
    """検索対象ソースの説明文。複数ソース検索時は「、」で並べる。"""
    keys = [corpus] if isinstance(corpus, str) else list(corpus)
    descs: List[str] = []
    for key in keys:
        desc = _CORPUS_DESCRIPTIONS.get(key)
        if desc and desc not in descs:
            descs.append(desc)
    return "、".join(descs) or _CORPUS_DESCRIPTIONS["audio"]


def build_system_prompt(today: Optional[date] = None, corpus: Corpus = "audio") -> str:
    today_str = (today or jst_today()).isoformat()
    corpus_desc = _corpus_description(corpus)
    return (
        "あなたは射出成形工場の社内アシスタントです。"
        f"{corpus_desc}から検索した内容(コンテキスト)に基づいて質問に答えます。\n"
        f"今日の日付: {today_str}\n"
        "ルール:\n"
        "- 事実は必ずコンテキストに基づき、該当する録音番号 [#n] を出典として示す\n"
        "- コンテキストは音声の自動文字起こしのため、誤変換や言い淀みが含まれうる。文脈から明らかな誤変換は補って解釈してよいが、推測した場合はその旨を付記する\n"
        "- 回答の形式は質問の指定に従う(表形式・報告書形式など)。指定がなければ簡潔な箇条書き\n"
        "- 日付は YYYY-MM-DD 形式で明示する\n"
        "- コンテキスト本文の [MM:SS〜MM:SS] は録音開始からの経過時間。発言時刻(録音内の経過時間)を引用する際は MM:SS 形式で明示する\n"
        "- コンテキストに根拠がない事項は推測せず、「記録には見つからない」と正直に述べる\n"
        "- 会話の文脈を維持し、直前のやり取りとの関連を保つ"
    )


def format_context_block(docs: List[ContextDoc]) -> str:
    blocks: List[str] = []
    for d in docs:
        meta_parts = [f"録音: {d.title}"]
        if d.recorded_date:
            meta_parts.append(f"録音日: {d.recorded_date}")
        if d.tags:
            meta_parts.append(f"タグ: {d.tags}")
        meta_parts.append(_SOURCE_LABELS.get(d.source, d.source))
        if not d.is_full_text:
            meta_parts.append("※関連部分の抜粋")
        if getattr(d, "time_basis", None) == "vad":
            # 元音声基準(word_timestamps_original_json)が無い録音は
            # 無音カット後の音声基準の時刻しか無いことを明示する
            meta_parts.append(
                "※録音内時刻は無音カット(VAD)後の音声基準のため、"
                "元の録音の再生位置とはズレている可能性がある"
            )
        header = f"[#{d.n}] " + " / ".join(meta_parts)
        blocks.append(f"{header}\n{d.text}")
    return "\n\n".join(blocks)


def build_chat_messages(
    query: str,
    docs: List[ContextDoc],
    chat_history: Optional[List[Dict]] = None,
    today: Optional[date] = None,
    corpus: Corpus = "audio",
    notes: Optional[List[str]] = None,
) -> List[Dict]:
    """回答生成用メッセージ列を組み立てる。

    notesには検索システム側の補足(期間拡大した・期間内の一部のみ参照している等)を
    渡す。モデルがコンテキストの範囲を誤解して「日付が矛盾する」等と混乱するのを防ぐ。
    """
    messages: List[Dict] = [{"role": "system", "content": build_system_prompt(today, corpus)}]

    if chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    notes_block = ""
    if notes:
        notes_block = "検索システムからの補足:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n"

    user_prompt = (
        "以下は音声DBから検索した録音の文字起こしです。これに基づいて質問に答えてください。\n\n"
        f"{notes_block}"
        f"{format_context_block(docs)}\n\n"
        f"質問:\n{query}"
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages
