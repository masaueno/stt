"""AIチャット(QA)タブ。

作業録音・社長音声・業務記録の3ソースを1つのチャットで横断検索する。
検索対象はソース選択pillsで自由に絞り込める(既定は全ソース)。
検索エンジン(RAGService)は共通で、選択ソースが検索対象とプロンプトに反映される。
"""

import json
import logging
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
from sqlalchemy import func

from models import RAGChatLog, USE_VECTOR, get_db
from services.rag import highlight_date_in_query
from services.rag.date_utils import jst_today
from services.rag_service import get_rag_service
from ui.categories import CATEGORY_LABELS
from ui.components import render_category_pills, render_date_range

logger = logging.getLogger(__name__)

_SOURCE_LABELS = CATEGORY_LABELS

_HEADER = "💬 AIチャット"
_PLACEHOLDER = "質問を入力（例: 7/28の作業内容は？ / 〇〇の件はどういう話だった？）"

# 自動インデックスを黙って実行する上限(超える場合はボタンで明示実行)
_AUTO_INDEX_LIMIT = 20


# ----------------------------------------------------------------------
# セッション管理
# ----------------------------------------------------------------------
def _get_or_create_session_id() -> str:
    if "rag_session_id" not in st.session_state:
        st.session_state["rag_session_id"] = str(uuid.uuid4())
    return st.session_state["rag_session_id"]


def _load_session_history(session_id: str) -> List[Dict]:
    db = next(get_db())
    try:
        logs = (
            db.query(RAGChatLog)
            .filter(RAGChatLog.session_id == session_id)
            .order_by(RAGChatLog.created_at.asc())
            .all()
        )
        history = []
        for log in logs:
            history.append({"role": "user", "content": log.user_text})
            history.append(
                {
                    "role": "assistant",
                    "content": log.answer_text or "",
                    "contexts": log.contexts or [],
                }
            )
        return history
    finally:
        db.close()


def _fetch_session_summaries(limit: int = 15):
    """過去の会話一覧。旧2タブ時代の会話(chat_kind別)もすべて表示する。"""
    db = next(get_db())
    try:
        base = (
            db.query(
                RAGChatLog.session_id.label("session_id"),
                func.min(RAGChatLog.created_at).label("first_created"),
                func.max(RAGChatLog.created_at).label("last_updated"),
            )
            .filter(RAGChatLog.session_id.isnot(None))
            .group_by(RAGChatLog.session_id)
            .subquery()
        )

        return (
            db.query(
                base.c.session_id,
                base.c.last_updated,
                RAGChatLog.user_text.label("first_question"),
            )
            .join(
                RAGChatLog,
                (RAGChatLog.session_id == base.c.session_id)
                & (RAGChatLog.created_at == base.c.first_created),
            )
            .order_by(base.c.last_updated.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()


def _render_session_picker(session_id: str) -> None:
    with st.popover("🗂 過去の会話", use_container_width=True):
        sessions = _fetch_session_summaries()
        if not sessions:
            st.caption("保存された会話はありません。")
        for s in sessions:
            title = (s.first_question or "（無題）").strip()
            if len(title) > 26:
                title = title[:26] + "…"
            prefix = "▶ " if s.session_id == session_id else ""
            updated = str(s.last_updated)[:10]  # YYYY-MM-DD
            if st.button(
                f"{prefix}{updated}　{title}",
                key=f"rag_resume_{s.session_id}",
                use_container_width=True,
            ):
                st.session_state["rag_session_id"] = s.session_id
                st.session_state["rag_history"] = _load_session_history(s.session_id)
                st.rerun()


# ----------------------------------------------------------------------
# 表示部品
# ----------------------------------------------------------------------
def _render_context_docs(contexts: List[Dict]) -> None:
    """参照した録音のカード表示。旧形式(chunk単位ログ)にも対応。"""
    for ctx in contexts:
        if "text" in ctx and "n" in ctx:  # 新形式(録音単位)
            title = ctx.get("title") or "（不明なファイル）"
            date_str = ctx.get("recorded_date") or "日付不明"
            source = _SOURCE_LABELS.get(ctx.get("source"), "")
            st.markdown(f"**[#{ctx['n']}] {date_str} — {title}**")
            caps = []
            if source:
                caps.append(source)
            if ctx.get("tags"):
                caps.append(f"🏷️ {ctx['tags']}")
            caps.append("全文" if ctx.get("is_full_text") else "関連部分の抜粋")
            st.caption(" / ".join(caps))
            body = ctx.get("text") or ""
            if len(body) > 600:
                st.text(body[:600] + f"…\n（表示は先頭600文字。使用したのは{len(body)}文字）")
            else:
                st.text(body)
        else:  # 旧形式
            st.markdown(f"**{ctx.get('file_path', '（不明）')}** — {ctx.get('recorded_at', '')}")
            if ctx.get("score") is not None:
                st.caption(f"スコア {ctx.get('score', 0):.3f}")
            st.text((ctx.get("chunk_text") or "")[:400])
        st.divider()


def _render_search_badges(meta: Dict) -> None:
    """検索時に適用された条件をバッジ表示する。"""
    parts = []
    date_filter = meta.get("date_filter")
    if date_filter:
        start, end = date_filter.get("start", ""), date_filter.get("end", "")
        span = start if start == end else f"{start} 〜 {end}"
        if date_filter.get("kind") == "recency":
            parts.append(f"🕒 :blue[「{date_filter.get('matched_text', '最近')}」→ {span} を優先]")
        elif date_filter.get("kind") == "manual":
            parts.append(f"📅 :green[期間指定: {span}]")
        else:
            parts.append(f"📅 :green[期間で絞り込み: {span}]")
    coverage = meta.get("coverage")
    if coverage:
        parts.append(
            f"📚 :gray[期間内{coverage['in_range']}件中、新しい順に{coverage['used']}件を参照]"
        )
    elif meta.get("mode") == "browse":
        parts.append("📚 :gray[期間内の録音を新しい順に参照]")
    if meta.get("mode") == "followup":
        parts.append("🔁 :gray[前回の参照録音を再利用]")
    if meta.get("fallback") == "recency_widened":
        parts.append("⚠️ :orange[期間内にデータがないため全期間から検索]")
    elif meta.get("fallback") == "quoted_term_widened":
        term = meta.get("widened_term") or ""
        parts.append(f"⚠️ :orange[「{term}」が期間内に見つからないため全期間から検索]")
    if parts:
        st.markdown("  ".join(parts))


def _handle_rag_error(error: Exception, context: str = ""):
    error_msg = str(error)
    if "OPENAI_API_KEY" in error_msg or "api_key" in error_msg.lower():
        st.error("🔑 APIキーが設定されていないか無効です。環境変数を確認してください。")
    elif "rate_limit" in error_msg.lower() or "429" in error_msg:
        st.warning("⏳ APIのレート制限に達しました。しばらく待ってから再試行してください。")
    else:
        st.error(f"❌ エラーが発生しました{': ' + context if context else ''}")
        with st.expander("詳細を表示"):
            st.code(error_msg)
    logger.error(f"RAG error ({context}): {error}", exc_info=True)


# ----------------------------------------------------------------------
# インデックス状態
# ----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def _corpus_snapshot() -> Tuple[Dict, Dict]:
    """コーパス統計と未索引件数。再描画のたびの再照会を防ぐため30秒キャッシュする。"""
    rag = get_rag_service()
    db = next(get_db())
    try:
        return rag.corpus_stats(db), rag.pending_counts(db)
    finally:
        db.close()


def _ensure_index(rag, pending: Dict[str, int]) -> None:
    """索引の差分を検出し、軽い処理は自動で、重い処理は明示実行で補完する。

    デスクトップ版などWeb以外の経路で保存されたデータもここで検索対象に取り込む。
    _AUTO_INDEX_LIMIT件以下なら開いたときに自動実行するため、通常運用では
    ユーザー操作なしで新しいデータが検索できるようになる。
    """
    if "rag_light_reconciled" not in st.session_state:
        db = next(get_db())
        try:
            rag.reconcile(db, embed=False)
            st.session_state.rag_light_reconciled = True
        except Exception as exc:
            logger.warning("軽量再同期に失敗: %s", exc)
        finally:
            db.close()

    total = sum(pending.values())
    if total == 0:
        return

    def _run_indexing():
        db2 = next(get_db())
        try:
            with st.status(f"未登録の録音 {total}件をインデックス化しています…", expanded=True) as status:
                bar = st.progress(0.0)
                def cb(done, all_count, label):
                    bar.progress(min(1.0, done / max(1, all_count)), text=f"{done}/{all_count} 件 ({label})")
                report = rag.reconcile(db2, embed=True, progress_cb=cb)
                ok = sum(report.get("indexed", {}).values())
                errs = report.get("errors", 0)
                status.update(
                    label=f"インデックス化が完了しました（成功 {ok}件 / 失敗 {errs}件）",
                    state="complete" if errs == 0 else "error",
                    expanded=False,
                )
        finally:
            db2.close()
        _corpus_snapshot.clear()
        st.rerun()

    if total <= _AUTO_INDEX_LIMIT:
        _run_indexing()
    else:
        st.warning(
            f"⚠️ **{total}件の録音が検索インデックスに未登録です**"
            f"（作業録音 {pending.get('audio', 0)}件 / 社長音声・業務記録 {pending.get('ceo', 0)}件）。"
            "デスクトップ版で保存されたデータや、検索エンジンの更新"
            "（埋め込みモデル変更）による再作成分が該当します。"
            "インデックス化するまで、これらはキーワード・類似検索の対象になりません。"
        )
        if st.button(
            f"今すぐインデックス化する（約{max(1, total // 100)}〜{max(2, total // 50)}分）",
            type="primary",
            key="rag_reindex_btn",
        ):
            _run_indexing()


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
def run_rag_tab():
    """AIチャットタブ。作業録音・社長音声・業務記録を横断検索する。"""
    st.header(_HEADER)

    rag = get_rag_service()

    if not rag.enabled:
        if not USE_VECTOR:
            st.warning(
                "この機能には Turso(libSQL) データベースが必要です。"
                "`DATABASE_URL` に `sqlite+libsql://...` を設定してください（ローカルSQLiteでは無効）。"
            )
        else:
            st.warning("OPENAI_API_KEY が未設定のためQA検索を利用できません。")
        return

    session_id = _get_or_create_session_id()
    hist_key = "rag_history"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    # --- コーパス統計と索引状態 ---
    stats, pending = _corpus_snapshot()

    # 検索ソース選択(既定は全ソース。自由に絞り込める)
    sel_sources = render_category_pills("rag_sources", "検索ソース")
    if sel_sources:
        counts = " / ".join(
            f"{_SOURCE_LABELS[s]} **{stats.get(s, {}).get('count', 0)}件**"
            for s in sel_sources
        )
        latest = max(
            (stats.get(s, {}).get("latest") or "" for s in sel_sources), default=""
        )
        st.caption(f"検索対象: {counts}（最新の録音日 {latest or '—'}）")

    _ensure_index(rag, pending)

    # --- 検索条件 ---
    ctrl_cols = st.columns([1.8, 1.1, 1.1])
    manual_range: Optional[Tuple[date, date]] = None
    with ctrl_cols[0]:
        manual_range = render_date_range(
            "rag",
            unset_label="📅 期間: 自動判定",
            clear_label="解除（自動に戻す）",
            caption="通常は質問文から自動判定します（「7/28の作業」「先月の記録」等）。固定したい場合のみ指定してください。",
        )
    with ctrl_cols[1]:
        if st.button("✨ 新規会話", use_container_width=True, key="rag_new_session"):
            st.session_state["rag_session_id"] = str(uuid.uuid4())
            st.session_state[hist_key] = []
            st.rerun()
    with ctrl_cols[2]:
        _render_session_picker(session_id)

    # --- 会話履歴表示 ---
    for message in st.session_state[hist_key]:
        block = st.chat_message(message["role"])
        if message["role"] == "user":
            block.markdown(highlight_date_in_query(message["content"]))
        else:
            block.markdown(message["content"])
            if message.get("contexts"):
                with block.expander(f"参照した録音（{len(message['contexts'])}件）", expanded=False):
                    _render_context_docs(message["contexts"])

    if not sel_sources:
        st.info("検索ソースを1つ以上選択してください。")
        return

    query = st.chat_input(_PLACEHOLDER, key="rag_chat_input")
    if not query:
        return

    st.session_state[hist_key].append({"role": "user", "content": query})
    st.chat_message("user").markdown(highlight_date_in_query(query))

    chat_history_for_rag = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state[hist_key][:-1]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    # 直前の回答で参照した録音(「表形式で」等の追問時に再検索せず再利用する)
    previous_docs = next(
        (
            m.get("contexts")
            for m in reversed(st.session_state[hist_key][:-1])
            if m.get("role") == "assistant" and m.get("contexts")
        ),
        None,
    )

    with st.spinner("音声DBを検索中…"):
        db = next(get_db())
        try:
            result = rag.answer_stream(
                db,
                query,
                sources=sel_sources,
                manual_date_range=manual_range,
                chat_history=chat_history_for_rag,
                previous_docs=previous_docs,
                # サーバーはUTCのため、「今日」「昨日」等の判定と
                # プロンプトの今日日付にはJSTの今日を明示的に渡す
                today=jst_today(),
            )
        except Exception as e:
            _handle_rag_error(e, "検索実行")
            db.close()
            return
        finally:
            db.close()

    docs = result.get("docs", [])
    meta = result.get("meta") or {}
    stream_fn = result.get("stream_fn")

    _render_search_badges(meta)

    with st.chat_message("assistant"):
        try:
            full_text = st.write_stream(stream_fn()) if callable(stream_fn) else ""
        except Exception as e:
            _handle_rag_error(e, "回答生成")
            full_text = ""

        if docs:
            with st.expander(f"参照した録音（{len(docs)}件）", expanded=False):
                _render_context_docs(docs)

    # --- 履歴・ログ保存 ---
    st.session_state[hist_key].append(
        {"role": "assistant", "content": full_text, "contexts": docs}
    )

    def _json_default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)

    # ログにはコンテキスト全文ではなく先頭部分のみ保存(肥大化防止)
    log_docs = []
    for d in docs:
        dd = dict(d)
        if isinstance(dd.get("text"), str) and len(dd["text"]) > 1500:
            dd["text"] = dd["text"][:1500] + "…"
        log_docs.append(dd)
    contexts_json = json.loads(json.dumps(log_docs, default=_json_default))

    db2 = next(get_db())
    try:
        log = RAGChatLog(
            session_id=session_id,
            # タブ統合後もカラムの値域("audio"/"ceo")は変えず、デスクトップ版
            # (stt-desktop commands/rag.rs)と同じ規則で記録する:
            # 作業録音を含む検索=audio / 含まない=ceo。
            # 回答ごとの参照ソースはcontexts[].sourceに残る。
            chat_kind="audio" if "audio" in sel_sources else "ceo",
            user_text=query,
            answer_text=full_text,
            contexts=contexts_json,
            used_hybrid=True,
            alpha=None,  # UI設定を廃止(既定値RAG_HYBRID_ALPHAで動作)
            date_filter_applied=bool(meta.get("date_filter")),
        )
        db2.add(log)
        db2.commit()
    except Exception as e:
        db2.rollback()
        logger.error(f"チャット保存エラー: {e}")
        st.warning("チャットの保存に失敗しました。")
    finally:
        db2.close()
