"""Account-scoped workspace: all figures come from the SQLite repository."""

from pathlib import Path

import streamlit as st

from modules.auth_ui import require_user
from modules.repository import Repository
from modules.ui_kit import empty, hero, metric

repo = Repository()
user = require_user(repo)
metrics = repo.stats(user["id"])
hero("YOUR KNOWLEDGE, IN ONE PLACE", f"欢迎回来，{user['username']}",
     "从资料管理到可追溯的智能问答，把企业知识与论文研究放在清晰、有序的工作区。")
st.space("small")
for column, label, value, note in zip(
    st.columns(3, gap="medium"),
    ("知识库", "已上传文档", "历史会话"),
    (metrics["bases"], metrics["uploads"], metrics["conversations"]),
    ("企业与论文分库管理", f"其中 {metrics['documents']} 份处理成功", "重启服务后仍可继续"),
):
    with column:
        metric(label, value, note)
if metrics["uploads"] == 0:
    with st.container(border=True):
        st.markdown("#### 第一次试用？从一份无敏感内容的样例开始")
        st.caption("① 创建企业知识库  →  ② 上传下面的 TXT  →  ③ 等待“处理成功”  →  ④ 在问答页提问：谁是项目负责人？")
        sample = Path(__file__).resolve().parents[2] / "examples" / "rag-demo.txt"
        if sample.is_file():
            st.download_button("下载样例 TXT", sample.read_bytes(), file_name="rag-demo.txt",
                               mime="text/plain", icon=":material/download:")
st.space("small")
st.subheader("从这里开始", anchor=False)
for column, title, detail, path, label, icon in zip(
    st.columns(3, gap="medium"),
    ("整理你的知识", "开始一次对话", "深读学术论文"),
    ("创建独立知识库，上传资料并追踪处理。", "选定知识库，根据原文证据获得回答。", "总结方法与结果，按篇核对论文来源。"),
    ("app_pages/knowledge.py", "app_pages/chat.py", "app_pages/papers.py"),
    ("进入知识库", "进入智能问答", "进入论文分析"),
    (":material/folder_open:", ":material/forum:", ":material/menu_book:"),
):
    with column, st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(detail)
        st.page_link(path, label=label, icon=icon, width="stretch")
st.space("small")
left, right = st.columns(2, gap="large")
with left:
    st.subheader("最近知识库", anchor=False)
    bases = repo.list_bases(user["id"])[:5]
    if not bases:
        empty("还没有知识库", "先创建企业库或论文库，再上传第一份资料。")
    for item in bases:
        with st.container(border=True):
            a, b = st.columns([3, 1], vertical_alignment="center")
            a.markdown(f"**{item['name']}**")
            b.caption("论文" if item["kind"] == "paper" else "企业")
            st.caption(f"{item['document_count']} 份可检索文档 · {item['updated_at'][:10]}")
with right:
    st.subheader("最近会话", anchor=False)
    conversations = repo.list_sessions(user["id"])[:5]
    if not conversations:
        empty("还没有会话", "上传文档并完成处理后，即可发起可追溯的问答。")
    for item in conversations:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.caption(f"{item['kb_name']} · {item['updated_at'][:16].replace('T',' ')}")
