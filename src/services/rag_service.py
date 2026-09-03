"""RAGオーケストレーション。Turso(libSQL)専用。

検索実行はservices/rag/search_service.py、索引の再同期はservices/rag/reconcile.py、
コンテキスト組み立てはservices/rag/context_builder.pyに分離している。
Phase 2(agentic search)ではSearchServiceの各メソッドをLLMのツールとして
公開する想定。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from sqlalchemy.orm import Session

from models import (
    AudioTranscriptionChunk,
    CeoTranscriptionChunk,
    EMBEDDING_DIM,
    RAG_FTS_TABLES,
    USE_VECTOR,
)
from sqlalchemy import text as sql_text

from services.rag import chunk_text
from services.rag.chunk_timing import assign_chunk_times, select_timing_words
from services.rag.context_builder import ContextDoc, build_context_docs
from services.rag.date_utils import DateRange, parse_date_from_query
from services.rag.prompt_builder import build_chat_messages
from services.rag.query_cleaner import (
    build_match_query,
    expand_synonyms,
    has_content_keywords,
    is_followup,
    strip_instructions,
    wants_aggregate,
)
from services.rag.reconcile import (
    JST,
    cleanup_orphan_chunks,
    fill_recorded_dates,
    find_unindexed,
    normalize_to_jst_date,
    set_index_meta,
    sync_fts,
)
from services.rag.search_service import SearchFilters, SearchService
from services.rag.tokenizer import fts_query_exact, to_fts_text

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
# 既定はtext-embedding-3-large(dimensions=1536で既存スキーマのまま)。
# prod実データ評価でベクトル検索のnDCG@6が0.33→0.55に改善したため引き上げた。
# モデル変更時はrag_index_metaのマーカー不一致で全録音が自動的に再索引対象になる。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
COMPLETION_MODEL = os.getenv("RAG_COMPLETION_MODEL", "gpt-5.6-luna")
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in {"1", "true", "yes", "on"}

# ベクトル側の重み。prod実データ評価ではキーワード(FTS)側が強く、
# 0.6ではハイブリッドがFTS単独に負けたため0.4に引き下げた。
HYBRID_DEFAULT_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.4"))
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "100"))

# コンテキストは録音単位で組み立てる。期間ブラウズ・集約系の質問では
# AGGREGATE_MAX_DOCSまで件数を広げ、1件あたりを薄く読む(網羅性優先)。
CONTEXT_MAX_DOCS = int(os.getenv("RAG_CONTEXT_MAX_DOCS", "6"))
AGGREGATE_MAX_DOCS = int(os.getenv("RAG_AGGREGATE_MAX_DOCS", "30"))
CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "40000"))
WHOLE_DOC_THRESHOLD = int(os.getenv("RAG_WHOLE_DOC_THRESHOLD", "4000"))
AGGREGATE_MIN_DOC_CHARS = 1200


@dataclass
class SearchPlan:
    query: str
    retrieval_text: str  # 埋め込み用テキスト(指示語除去+同義語付与済み)
    match_query: Optional[str]  # FTS5クエリ(内容語のみ+同義語展開)
    date_range: Optional[DateRange]
    mode: str  # "search" | "browse" | "followup"
    sources: Tuple[str, ...]
    aggregate: bool = False  # 「まとめて」等、期間内を広く読むべき質問か


class RAGService:
    """埋め込み生成・索引管理・検索・回答生成のオーケストレーター。"""

    def __init__(self) -> None:
        self._enabled = bool(USE_VECTOR) and ENABLE_RAG
        self._search = SearchService()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self._enabled:
                logger.warning("OPENAI_API_KEY が未設定のため RAG を無効化します")
            self._enabled = False
        self._client = OpenAI() if self._enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def search(self) -> SearchService:
        return self._search

    # ------------------------------------------------------------------
    # 索引作成
    # ------------------------------------------------------------------
    def index_transcription(self, db: Session, transcription_id: int, text: str) -> bool:
        """作業録音の文字起こしをチャンク化して索引に登録する。"""
        return self._index_generic(db, "audio", transcription_id, text)

    def index_ceo_transcription(self, db: Session, transcription_id: int, text: str) -> bool:
        """社長音声の文字起こしを索引に登録する。"""
        return self._index_generic(db, "ceo", transcription_id, text)

    def _index_generic(self, db: Session, source: str, transcription_id: int, text: str) -> bool:
        if not self.enabled:
            return False

        chunk_model = AudioTranscriptionChunk if source == "audio" else CeoTranscriptionChunk
        fts = RAG_FTS_TABLES[source]

        chunks = list(chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP))
        if not chunks:
            logger.debug("RAG: チャンクなしのためスキップ (%s id=%s)", source, transcription_id)
            return False

        embeddings = self._embed_texts(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.warning("RAG: 埋め込み生成に失敗したためスキップ (%s id=%s)", source, transcription_id)
            return False

        # recorded_dateの補完と単語タイムスタンプの取得
        # (ORMロードは不正JSON列で失敗しうるため生SQL)
        parent_table = "audio_transcriptions" if source == "audio" else "ceo_transcriptions"
        date_source_col = "recorded_at" if source == "ceo" else "created_at"
        row = db.execute(
            sql_text(
                f"SELECT recorded_date, {date_source_col}, created_at, "
                f"word_timestamps_json, word_timestamps_original_json "
                f"FROM {parent_table} WHERE id = :id"
            ),
            {"id": transcription_id},
        ).first()
        if row is not None and not row[0]:
            recorded_date = (
                normalize_to_jst_date(row[1])
                or normalize_to_jst_date(row[2])
                or datetime.now(JST).strftime("%Y-%m-%d")
            )
            db.execute(
                sql_text(f"UPDATE {parent_table} SET recorded_date = :d WHERE id = :id"),
                {"d": recorded_date, "id": transcription_id},
            )

        # チャンクへの録音内時刻割り当て(単語タイムスタンプがある録音のみ)。
        # デスクトップ版保存分(word_timestamps_jsonあり)も再同期経由でここを通る。
        time_basis: Optional[str] = None
        chunk_times: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * len(chunks)
        if row is not None:
            try:
                timing_words, time_basis = select_timing_words(row[3], row[4])
                if timing_words:
                    chunk_times = assign_chunk_times(
                        text, chunks, timing_words, chunk_overlap=DEFAULT_CHUNK_OVERLAP
                    )
            except Exception as exc:
                logger.warning(
                    "チャンク時刻の割り当てに失敗したため時刻なしで索引します (%s id=%s): %s",
                    source, transcription_id, exc,
                )
                time_basis = None
                chunk_times = [(None, None)] * len(chunks)

        # 既存チャンク+FTS行を削除してから再作成
        old_ids = [
            r[0]
            for r in db.query(chunk_model.id).filter_by(transcription_id=transcription_id).all()
        ]
        for cid in old_ids:
            db.execute(sql_text(f"DELETE FROM {fts} WHERE rowid = :id"), {"id": cid})
        db.query(chunk_model).filter_by(transcription_id=transcription_id).delete()

        new_chunks = []
        for idx, (piece, embedding) in enumerate(zip(chunks, embeddings)):
            start_sec, end_sec = chunk_times[idx] if idx < len(chunk_times) else (None, None)
            chunk = chunk_model(
                transcription_id=transcription_id,
                chunk_index=idx,
                chunk_text=piece,
                embedding=embedding,
                start_sec=start_sec,
                end_sec=end_sec,
                time_basis=time_basis if start_sec is not None else None,
            )
            db.add(chunk)
            new_chunks.append(chunk)
        db.flush()  # idを確定させFTSに登録
        for chunk in new_chunks:
            db.execute(
                sql_text(f"INSERT INTO {fts}(rowid, tokens) VALUES (:id, :tokens)"),
                {"id": chunk.id, "tokens": to_fts_text(chunk.chunk_text)},
            )
        return True

    # ------------------------------------------------------------------
    # 再同期(デスクトップ保存分・社長音声・FTS/日付の差分補完)
    # ------------------------------------------------------------------
    def pending_counts(self, db: Session) -> Dict[str, int]:
        missing = find_unindexed(db, embedding_model=EMBEDDING_MODEL)
        return {source: len(ids) for source, ids in missing.items()}

    def reconcile(
        self,
        db: Session,
        embed: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """索引の差分を補完する。embed=Falseなら埋め込み不要の処理のみ。"""
        report: Dict = {}
        report["dates_filled"] = fill_recorded_dates(db)
        report["orphan_chunks_removed"] = cleanup_orphan_chunks(db)
        report["fts"] = sync_fts(db)
        db.commit()

        missing = find_unindexed(db, embedding_model=EMBEDDING_MODEL)
        report["unindexed"] = {s: len(ids) for s, ids in missing.items()}
        report["indexed"] = {"audio": 0, "ceo": 0}
        report["errors"] = 0
        if not embed or not self.enabled:
            return report

        total = sum(len(ids) for ids in missing.values())
        done = 0
        for source, ids in missing.items():
            parent_table = "audio_transcriptions" if source == "audio" else "ceo_transcriptions"
            for tid in ids:
                row = db.execute(
                    sql_text(f"SELECT transcript FROM {parent_table} WHERE id = :id"),
                    {"id": tid},
                ).first()
                if row is None or not row[0]:
                    done += 1
                    continue
                try:
                    if self._index_generic(db, source, tid, row[0]):
                        db.commit()
                        report["indexed"][source] += 1
                    else:
                        db.rollback()
                        report["errors"] += 1
                except Exception as exc:
                    db.rollback()
                    report["errors"] += 1
                    logger.error("再同期での索引作成に失敗 (%s id=%s): %s", source, tid, exc)
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"{source} #{tid}")

        if report["errors"] == 0:
            # 全件が現行の埋め込みモデルで索引済みになったらマーカーを記録する。
            # (マーカー確認はモデル無指定=チャンク有無のみで行う)
            leftover = find_unindexed(db, embedding_model=None)
            if not any(leftover.values()):
                set_index_meta(db, "embedding_model", EMBEDDING_MODEL)
                db.commit()
        return report

    def corpus_stats(self, db: Session) -> Dict[str, Dict]:
        return self._search.corpus_stats(db)

    # ------------------------------------------------------------------
    # 検索計画
    # ------------------------------------------------------------------
    def plan_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        manual_date_range: Optional[Tuple[date, date]] = None,
        sources: Sequence[str] = ("audio",),
        today: Optional[date] = None,
        allow_followup: bool = True,
    ) -> SearchPlan:
        if manual_date_range:
            date_range = DateRange(
                manual_date_range[0], manual_date_range[1], "manual", ""
            )
        else:
            date_range = parse_date_from_query(query, today)

        aggregate = wants_aggregate(query)

        # 形式変更・メタ質問(「表形式で」「具体的に」「前の回答は〜」等)は
        # 新規検索せず前回の参照録音を再利用する(実質問ログの約4割が該当)
        if (
            allow_followup
            and chat_history
            and not manual_date_range
            and is_followup(query, has_history=True, has_date=date_range is not None)
        ):
            return SearchPlan(
                query=query,
                retrieval_text=query,
                match_query=None,
                date_range=None,
                mode="followup",
                sources=tuple(sources),
                aggregate=aggregate,
            )

        residual = query
        if date_range and date_range.matched_text:
            residual = residual.replace(date_range.matched_text, " ")

        mode = "search"
        if not has_content_keywords(residual) and (date_range or aggregate):
            # 「7/28の作業内容は?」「最近の記録をまとめて」のような、
            # 期間・要約だけが手がかりの質問は期間ブラウズで確実に拾う
            mode = "browse"

        # 短い追問(「ボイドは?」等)は直前の質問を検索文に含める
        base_text = query
        if chat_history and len(query) <= 12:
            prev_user = next(
                (m.get("content", "") for m in reversed(chat_history) if m.get("role") == "user"),
                "",
            )
            if prev_user:
                base_text = f"{prev_user}\n{query}"

        # 埋め込み・FTSとも指示語(「表形式で」等)を除いた内容語で検索する。
        # 表記ゆれ同義語(ヒケ→引け等)は両方に付与する。
        synonyms = expand_synonyms(base_text)
        clean = strip_instructions(base_text)
        if date_range and date_range.matched_text:
            clean = clean.replace(date_range.matched_text, " ")
        retrieval_text = clean.strip() or query
        if synonyms:
            retrieval_text = f"{retrieval_text} {' '.join(synonyms)}"

        return SearchPlan(
            query=query,
            retrieval_text=retrieval_text,
            match_query=build_match_query(clean, synonyms),
            date_range=date_range,
            mode=mode,
            sources=tuple(sources),
            aggregate=aggregate,
        )

    # ------------------------------------------------------------------
    # 回答生成(ストリーミング)
    # ------------------------------------------------------------------
    @staticmethod
    def _previous_hits(previous_docs: Optional[List[Dict]]) -> List[Dict]:
        """前回の参照録音(docs形式)をコンテキスト組み立て用のヒット形式に変換する。"""
        hits: List[Dict] = []
        for i, d in enumerate(previous_docs or []):
            tid = d.get("transcription_id")
            if not tid:
                continue
            hits.append(
                {
                    "source": d.get("source") or "audio",
                    "transcription_id": int(tid),
                    "title": d.get("title") or d.get("file_path") or "",
                    "recorded_date": d.get("recorded_date"),
                    "tags": d.get("tags"),
                    "score": 1.0 - i * 0.01,
                    "chunk_id": None,
                    "chunk_index": None,
                    "chunk_text": None,
                }
            )
        return hits

    def answer_stream(
        self,
        db: Session,
        query: str,
        *,
        sources: Sequence[str] = ("audio",),
        manual_date_range: Optional[Tuple[date, date]] = None,
        chat_history: Optional[List[Dict]] = None,
        previous_docs: Optional[List[Dict]] = None,
        alpha: Optional[float] = None,
        retrieval_k: Optional[int] = None,
        max_docs: Optional[int] = None,
        today: Optional[date] = None,
    ) -> Dict:
        """検索→コンテキスト組み立てまで実行し、生成はストリーミングで返す。

        previous_docsには直前の回答で参照した録音(docs形式)を渡す。追問と
        判定された場合に再検索せずこれを再利用する。
        戻り値: {"docs": [...], "meta": {...}, "stream_fn": callable}
        """
        if not self.enabled or not self._client:
            return {"docs": [], "meta": {}, "stream_fn": lambda: iter(())}

        alpha = HYBRID_DEFAULT_ALPHA if alpha is None else float(alpha)
        k = int(retrieval_k or RETRIEVAL_K)

        t0 = time.time()
        plan = self.plan_query(
            query,
            chat_history=chat_history,
            manual_date_range=manual_date_range,
            sources=sources,
            today=today,
        )
        prev_hits = self._previous_hits(previous_docs) if plan.mode == "followup" else []
        if plan.mode == "followup" and not prev_hits:
            # 再利用できる前回の参照録音が無ければ通常検索として計画し直す
            plan = self.plan_query(
                query,
                chat_history=chat_history,
                manual_date_range=manual_date_range,
                sources=sources,
                today=today,
                allow_followup=False,
            )

        notes: List[str] = []
        fallback = None
        widened_term = None
        coverage: Optional[Dict] = None

        # --- 追問: 前回の参照録音を再利用(検索・埋め込みなし) ---
        if plan.mode == "followup":
            n_docs = len(prev_hits)
            per_doc_cap = None
            if n_docs > CONTEXT_MAX_DOCS:
                per_doc_cap = max(AGGREGATE_MIN_DOC_CHARS, CONTEXT_MAX_CHARS // n_docs)
            t1 = time.time()
            docs = build_context_docs(
                db,
                prev_hits,
                max_docs=max(1, n_docs),
                max_chars=CONTEXT_MAX_CHARS,
                whole_doc_threshold=WHOLE_DOC_THRESHOLD,
                order="score",
                per_doc_cap=per_doc_cap,
            )
            notes.append(
                "この質問は直前のやり取りへの追加要望と判断し、"
                "前回の回答で参照した録音を引き続き参照している"
            )
            return self._build_result(
                plan, query, chat_history, docs, notes,
                fallback=None, widened_term=None, n_candidates=len(prev_hits),
                coverage=None, today=today, t0=t0, t1=t1,
            )

        # 期間ブラウズ・集約系は件数を広げ、1録音あたりを薄く読む(網羅性優先)
        if max_docs:
            n_docs = int(max_docs)
        elif plan.mode == "browse" or plan.aggregate:
            n_docs = AGGREGATE_MAX_DOCS
        else:
            n_docs = CONTEXT_MAX_DOCS

        filters = SearchFilters(
            date_from=plan.date_range.start if plan.date_range else None,
            date_to=plan.date_range.end if plan.date_range else None,
            sources=plan.sources,
        )

        if plan.mode == "browse":
            hits = self._search.browse_recent(db, filters, max_recordings=n_docs + 2)
            if not hits and plan.date_range and plan.date_range.kind == "recency":
                # 「最近」で30日以内にデータがなければ全期間の直近から
                fallback = "recency_widened"
                filters = SearchFilters(sources=plan.sources)
                hits = self._search.browse_recent(db, filters, max_recordings=n_docs + 2)
        else:
            qvecs = self._embed_texts([plan.retrieval_text])
            qvec = qvecs[0] if qvecs else None
            hits = self._search.hybrid_search(
                db, plan.retrieval_text, qvec, filters, k, alpha,
                match_query=plan.match_query,
            )
            if not hits and plan.date_range and plan.date_range.kind == "recency":
                fallback = "recency_widened"
                filters = SearchFilters(sources=plan.sources)
                hits = self._search.hybrid_search(
                    db, plan.retrieval_text, qvec, filters, k, alpha,
                    match_query=plan.match_query,
                )

            # 「ヒケ」等の引用符付き用語は必須語として扱う。期間内に本語も
            # 表記ゆれも完全一致しない一方で全期間には存在する場合のみ、
            # 期間を広げて確実に拾う
            if plan.date_range and filters.date_from is not None:
                for term in re.findall(r"[「『\"]([^」』\"]{1,20})[」』\"]", query):
                    probes = [term] + expand_synonyms(term)
                    queries = [q for q in (fts_query_exact(p) for p in probes) if q]
                    if not queries:
                        continue
                    in_range = any(
                        self._search.keyword_search(db, q, filters, 1) for q in queries
                    )
                    if in_range:
                        continue
                    unfiltered = SearchFilters(sources=plan.sources)
                    if any(
                        self._search.keyword_search(db, q, unfiltered, 1) for q in queries
                    ):
                        fallback = "quoted_term_widened"
                        widened_term = term
                        filters = unfiltered
                        hits = self._search.hybrid_search(
                            db, plan.retrieval_text, qvec, filters, k, alpha,
                            match_query=plan.match_query,
                        )
                        break

        t1 = time.time()

        if not hits:
            msg = "関連する録音が見つかりませんでした。"
            if plan.date_range:
                available = self._search.available_date_range(db, plan.sources)
                msg = (
                    f"指定された期間（{plan.date_range.start} 〜 {plan.date_range.end}）に"
                    "該当する録音が見つかりませんでした。"
                )
                if available:
                    msg += f"\n\nデータが存在する期間: {available[0]} 〜 {available[1]}"
            return {
                "docs": [],
                "meta": self._build_meta(plan, fallback, 0, [], 0, t0, t1, t1),
                "stream_fn": lambda m=msg: iter((m,)),
            }

        per_doc_cap = None
        if n_docs > CONTEXT_MAX_DOCS:
            per_doc_cap = max(AGGREGATE_MIN_DOC_CHARS, CONTEXT_MAX_CHARS // n_docs)
        order = "date" if plan.mode == "browse" else "score"
        docs = build_context_docs(
            db,
            hits,
            max_docs=n_docs,
            max_chars=CONTEXT_MAX_CHARS,
            whole_doc_threshold=WHOLE_DOC_THRESHOLD,
            order=order,
            per_doc_cap=per_doc_cap,
        )

        # 補足メモ(モデルがコンテキストの範囲を誤解しないように明示する)
        if fallback == "recency_widened" and plan.date_range:
            notes.append(
                f"「{plan.date_range.matched_text or '最近'}」の期間内に録音が無いため、"
                "全期間の直近の録音を参照している"
            )
        elif fallback == "quoted_term_widened" and widened_term:
            notes.append(
                f"「{widened_term}」が指定期間内に見つからないため、"
                "全期間から検索した(指定期間外の録音を含む)"
            )
        if plan.mode == "browse" and filters.date_from is not None:
            total_in_range = self._search.count_recordings(db, filters)
            if total_in_range > len(docs):
                coverage = {"in_range": total_in_range, "used": len(docs)}
                notes.append(
                    f"期間内の録音は全{total_in_range}件あり、新しい順に{len(docs)}件を"
                    "参照している(全件ではない)。件数を断定する場合はその旨を付記する"
                )

        return self._build_result(
            plan, query, chat_history, docs, notes,
            fallback=fallback, widened_term=widened_term, n_candidates=len(hits),
            coverage=coverage, today=today, t0=t0, t1=t1,
        )

    def _build_result(
        self,
        plan: SearchPlan,
        query: str,
        chat_history: Optional[List[Dict]],
        docs: List[ContextDoc],
        notes: List[str],
        *,
        fallback: Optional[str],
        widened_term: Optional[str],
        n_candidates: int,
        coverage: Optional[Dict],
        today: Optional[date],
        t0: float,
        t1: float,
    ) -> Dict:
        """コンテキスト確定後の共通処理(プロンプト構築・ストリーム関数・メタ)。"""
        messages = build_chat_messages(
            query, docs, chat_history, today, corpus=plan.sources, notes=notes or None
        )
        t2 = time.time()

        client = self._client

        def _stream_gen():
            try:
                with client.responses.stream(model=COMPLETION_MODEL, input=messages) as stream:
                    for event in stream:
                        et = getattr(event, "type", None)
                        if et == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str) and delta:
                                yield delta
                        elif et == "response.error":
                            err = getattr(event, "error", None)
                            logger.error("Responses stream error: %s", err)
                            yield f"\n\n⚠️ 回答生成でエラーが発生しました: {err}"
            except Exception as exc:
                logger.error("Responses stream failed: %s", exc, exc_info=True)
                yield (
                    f"\n\n⚠️ 回答の生成に失敗しました（モデル: {COMPLETION_MODEL}）。\n"
                    f"エラー: {exc}"
                )

        used_chars = sum(len(d.text) for d in docs)
        meta = self._build_meta(plan, fallback, n_candidates, docs, used_chars, t0, t1, t2)
        meta["widened_term"] = widened_term
        meta["aggregate"] = plan.aggregate
        meta["coverage"] = coverage
        return {"docs": [d.to_dict() for d in docs], "meta": meta, "stream_fn": _stream_gen}

    @staticmethod
    def _build_meta(
        plan: SearchPlan,
        fallback: Optional[str],
        n_candidates: int,
        docs: List[ContextDoc],
        used_chars: int,
        t0: float,
        t1: float,
        t2: float,
    ) -> Dict:
        return {
            "mode": plan.mode,
            "sources": list(plan.sources),
            "date_filter": (
                {
                    "start": plan.date_range.start.isoformat(),
                    "end": plan.date_range.end.isoformat(),
                    "kind": plan.date_range.kind,
                    "matched_text": plan.date_range.matched_text,
                }
                if plan.date_range
                else None
            ),
            "fallback": fallback,
            "candidates": n_candidates,
            "used_docs": len(docs),
            "used_context_chars": used_chars,
            "doc_summaries": [
                {
                    "n": d.n,
                    "title": d.title,
                    "recorded_date": d.recorded_date,
                    "score": d.score,
                    "source": d.source,
                }
                for d in docs
            ],
            "timings_ms": {
                "retrieval": int((t1 - t0) * 1000.0),
                "prompt_build": int((t2 - t1) * 1000.0),
            },
        }

    # ------------------------------------------------------------------
    # 埋め込み
    # ------------------------------------------------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            return []
        try:
            # dimensionsを明示することで、3-large(本来3072次元)でも既存の
            # スキーマ(EMBEDDING_DIM=1536)にそのまま格納できる
            response = self._client.embeddings.create(
                model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIM
            )
        except Exception as exc:  # pragma: no cover - APIエラー
            logger.error("OpenAI embeddings API 呼び出しで失敗: %s", exc)
            return []

        embeddings: List[List[float]] = []
        for item in response.data:
            embedding = getattr(item, "embedding", None)
            if embedding and len(embedding) == EMBEDDING_DIM:
                embeddings.append(list(embedding))
            else:
                logger.warning(
                    "RAG: 埋め込みベクトルの次元が想定と異なります (expected=%s, actual=%s)",
                    EMBEDDING_DIM,
                    len(embedding) if embedding else None,
                )
        return embeddings


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
