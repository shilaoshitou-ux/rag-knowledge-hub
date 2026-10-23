"""Account-scoped workspace; figures and recent activity come from SQLite."""

from pathlib import Path

import streamlit as st

from modules.auth_ui import require_user
from modules.repository import Repository
from modules.ui_kit import empty, feature_intro, hero, metric, section_heading

repo = Repository()
user = require_user(repo)
counts = repo.stats(user["id"])
hero("ATLAS / 工作台", f"欢迎回来，{user['username']}",
     "你的知识、论文和对话汇聚于此。找到需要的资料，然后从原文证据出发继续探索。")

stat_items = (
    ("知识库", counts["bases"], "企业与论文独立管理", "folder_open"),
    ("已上传文档", counts["uploads"], f"{counts['documents']} 份处理成功", "description"),
    ("历史会话", counts["conversations"], "讨论与引用可随时找回", "forum"),
)
for column, (label, value, note, icon) in zip(st.columns(3, gap="medium"), stat_items):
    with column:
        metric(label, value, note, icon)

section_heading("继续你的工作", "选择一个入口，立刻开始")
features = (
    ("整理知识", "创建独立空间、上传资料，并掌握每份文件的处理进度。",
     "folder_open", "app_pages/knowledge.py", "管理知识库"),
    ("基于证据提问", "检索可信片段，在同一会话中逐步追问并核对来源。",
     "forum", "app_pages/chat.py", "开始问答"),
    ("读懂论文", "定位作者方法、关键实验与局限，比较不同论文。",
     "menu_book", "app_pages/papers.py", "打开论文工作区"),
)
for index, (title, detail, icon, target, action) in enumerate(features):
    if index % 3 == 0:
        columns = st.columns(3, gap="medium")
    with columns[index % 3]:
        with st.container(border=True, key=f"rag_feature_{index}"):
            feature_intro(title, detail, icon)
            st.page_link(target, label=action, icon=":material/arrow_outward:", width="stretch")

if counts["uploads"] == 0:
    st.space("small")
    with st.container(border=True):
        st.markdown("**第一次使用？用一份合成样例体验流程**")
        st.caption("创建企业知识库 → 上传 TXT → 等待处理成功 → 提问“谁是项目负责人？”")
        sample = Path(__file__).resolve().parents[2] / "examples" / "rag-demo.txt"
        if sample.is_file():
            st.download_button("下载样例 TXT", sample.read_bytes(), file_name="rag-demo.txt",
                               mime="text/plain", icon=":material/download:")

st.space("small")
recent_bases, recent_sessions = st.columns(2, gap="large")
with recent_bases:
    section_heading("最近知识库", "RECENT COLLECTIONS")
    bases = repo.list_bases(user["id"])[:4]
    if not bases:
        empty("还没有知识库", "创建企业库或论文库，随后上传你的第一份资料。")
    for item in bases:
        with st.container(border=True, key=f"rag_recent_kb_{item['id']}"):
            name, tag = st.columns([4, 1], vertical_alignment="center")
            name.markdown(f"**{item['name']}**")
            tag.badge("论文" if item["kind"] == "paper" else "企业",
                      color="violet" if item["kind"] == "paper" else "blue")
            st.caption(f"{item['document_count']} 份可检索文档 · 更新于 {item['updated_at'][:10]}")
with recent_sessions:
    section_heading("最近会话", "RECENT ACTIVITY")
    sessions = repo.list_sessions(user["id"])[:4]
    if not sessions:
        empty("还没有会话", "上传资料并完成处理后，就能开启带引用的问答。")
    for item in sessions:
        with st.container(border=True, key=f"rag_recent_session_{item['id']}"):
            st.markdown(f"**{item['title']}**")
            st.caption(f"{item['kb_name']} · {item['updated_at'][:16].replace('T', ' ')}")