import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

from ui.sidebar import build_sidebar
from ui.categories import AUDIO, CEO, WORK
from ui.tabs.audio_tab import run_audio_tab
from ui.tabs.ceo_tab import run_ceo_tab
from ui.tabs.db_tab import run_db_tab
from ui.tabs.rag_tab import run_rag_tab
# .envファイルを読み込む
load_dotenv()

from models import AudioTranscription, get_db
from stt_wrapper import STTModelWrapper
from text_structurer import TextStructurer
from env_watcher import check_env_changes, display_env_status
from app_settings import AppSettings
from auth import check_password, logout

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ログディレクトリの作成
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# ファイルハンドラー
file_handler = logging.FileHandler(log_dir / "streamlit_app.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# フォーマッター
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ページ設定
st.set_page_config(
    page_title="音声文字起こしWebアプリ",
    page_icon="🎙️",
    layout="wide"
)

# 環境変数の変更をチェック
check_env_changes()

# アプリ設定の初期化
settings = AppSettings()

# Basic認証チェック
if not check_password():
    st.stop()

# タイトル
st.title("🎙️ 音声文字起こしWebアプリ")

# AppSettings はディスク（.app_settings.json）から毎回ロードされるため、
# session_state には常に最新の参照を入れ替えておく。
# 初回ガード（"settings" not in st.session_state）にすると、画面再描画後に
# 各タブから取り出すインスタンスが古いままになり、保存した設定が反映されない。
st.session_state.settings = settings

with st.sidebar:
    selected_model, use_structuring, debug_mode = build_sidebar(settings, log_dir, logger)

tab_record, tab_db, tab_chat = st.tabs([
    "🎙️ 録音",
    "🗄️ データベース",
    "💬 AIチャット",
])
with tab_record:
    sub_audio, sub_ceo, sub_work = st.tabs([AUDIO.label, CEO.label, WORK.label])
    with sub_audio:
        run_audio_tab(selected_model, use_structuring, logger)
    with sub_ceo:
        run_ceo_tab(CEO, selected_model, logger)
    with sub_work:
        run_ceo_tab(WORK, selected_model, logger)
with tab_db:
    run_db_tab()
with tab_chat:
    run_rag_tab()

# フッター
st.divider()
st.markdown("🎙️ 音声文字起こしWebアプリ v1.0")
