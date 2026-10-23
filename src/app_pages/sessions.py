"""Persistent session list; all queries are scoped to the signed-in account."""

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.repository import Repository

repo = Repository()
user = require_user(repo)
page_title("HISTORY / CONVERSATIONS", "历史会话", "保留每段研究和问答的来龙去脉，随时回到上次讨论。")
sessions = repo.list_sessions(user["id"])
if not sessions:
    st.info("目前没有历史会话。")
    st.page_link("app_pages/chat.py", label="开始新对话", icon=":material/arrow_forward:")
for session in sessions:
    with st.container(border=True, key=f"rag_session_card_{session['id']}"):
        a, b, c = st.columns([4, 1, 1], vertical_alignment="center")
        with a:
            st.markdown(f"**{session['title']}**")
            st.caption(f"{session['kb_name']} · {session['updated_at'][:16].replace('T',' ')}")
        with b:
            if st.button("继续", key=f"resume_{session['id']}"):
                st.session_state["rag_managed_kb"] = session["kb_id"]
                st.session_state["rag_managed_session"] = session["id"]
                st.switch_page("app_pages/chat.py")
        with c:
            if st.button("删除", key=f"delete_{session['id']}"):
                repo.delete_session(user["id"], session["id"])
                st.rerun()