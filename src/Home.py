"""Incremental entrypoint: managed workspace plus existing legacy utilities."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from modules.ui_kit import GLOBAL_STYLE

load_dotenv()

st.set_page_config(page_title="Atlas · 智能知识工作区", page_icon=":material/auto_stories:", layout="wide")

from modules.auth_ui import restore_login
from modules.repository import Repository

restore_login(Repository())  # Restore a server-validated cookie before choosing login CSS.

pages = {
    "工作区": [
        st.Page("app_pages/dashboard.py", title="工作台", icon=":material/dashboard:", default=True),
        st.Page("app_pages/chat.py", title="AI 智能问答", icon=":material/forum:", url_path="chat"),
        st.Page("app_pages/knowledge.py", title="我的知识库", icon=":material/folder_open:", url_path="knowledge"),
        st.Page("app_pages/papers.py", title="论文智能分析", icon=":material/menu_book:", url_path="papers"),
        st.Page("app_pages/sessions.py", title="历史会话", icon=":material/history:", url_path="sessions"),
        st.Page("app_pages/evaluation.py", title="RAG 效果评测", icon=":material/query_stats:", url_path="evaluation"),
        st.Page("app_pages/settings.py", title="系统设置", icon=":material/settings:", url_path="settings"),
    ],
}
# Legacy document/table/video pages predate per-user permissions. Keep their code and
# indices intact, but do not expose them in the shared workspace by default.
if os.getenv("RAG_ENABLE_LEGACY", "false").lower() == "true":
    pages["旧版工具（仅本机单用户）"] = [
        st.Page("pages/1_📄文档问答.py", title="原有文档问答", icon=":material/article:", url_path="文档问答"),
        st.Page("pages/2_📊表格分析（可选）.py", title="表格分析", icon=":material/table_chart:", url_path="表格分析"),
        st.Page("pages/3_🎬视频摘要.py", title="视频摘要", icon=":material/video_library:", url_path="视频摘要"),
    ]
if st.session_state.get("rag_current_user"):
    assets = Path(__file__).with_name("assets")
    st.logo(str(assets / "atlas-logo.svg"), icon_image=str(assets / "atlas-mark.svg"), size="large")
navigation = st.navigation(pages, expanded=False)
st.html(GLOBAL_STYLE)
if not st.session_state.get("rag_current_user"):
    # Login is a dedicated product page; navigation appears after authentication.
    st.html("<style>[data-testid='stSidebar'],[data-testid='stExpandSidebarButton']{display:none!important} [data-testid='stHeader']{display:none!important} [data-testid='stMainBlockContainer']{padding-top:2rem!important}</style>")
navigation.run()