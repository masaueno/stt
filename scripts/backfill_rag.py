"""RAG索引の再同期CLI。

デスクトップ版で保存された録音・社長音声など、チャンク/FTS/recorded_dateが
未整備のデータを検出してインデックス化する。何度実行しても安全(冪等)。

使い方:
    uv run python scripts/backfill_rag.py            # 差分をすべて索引化
    uv run python scripts/backfill_rag.py --dry-run  # 差分の件数確認のみ
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import get_db  # noqa: E402
from services.rag_service import get_rag_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG索引の再同期")
    parser.add_argument("--dry-run", action="store_true", help="差分の件数確認のみ(索引化しない)")
    args = parser.parse_args()

    rag = get_rag_service()
    if not rag.enabled:
        raise SystemExit("RAGが有効化されていません。DATABASE_URLやOPENAI_API_KEYを確認してください。")

    db = next(get_db())
    t0 = time.time()
    try:
        if args.dry_run:
            pending = rag.pending_counts(db)
            print(
                f"未インデックス: 作業録音 {pending.get('audio', 0)}件 / "
                f"社長音声・業務記録 {pending.get('ceo', 0)}件"
            )
            return

        def progress(done: int, total: int, label: str) -> None:
            print(f"  {done}/{total} ({time.time() - t0:.0f}s) {label}", flush=True)

        report = rag.reconcile(db, embed=True, progress_cb=progress)
        print("--- 再同期結果 ---")
        print(f"recorded_date補完: {report['dates_filled']}")
        print(f"孤児チャンク削除: {report['orphan_chunks_removed']}")
        print(f"FTS同期: {report['fts']}")
        print(f"索引化: {report['indexed']} (失敗 {report['errors']}件)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
