"""Account-scoped workspace; figures and recent activity come from SQLite."""

from pathlib import Path

import streamlit as st

from modules.auth_ui import require_user
from modules.repository import Repository
from modules.ui_kit import empty, hero, metric, section_heading

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

bases = repo.list_bases(user["id"])
ready_base = next((item for item in bases if item["document_count"] > 0), None)
if not bases:
    next_title, next_detail, next_label = (
        "先建一个知识库", "选择企业资料或学术论文库，为接下来上传的资料建立独立空间。", "创建知识库")
    next_target = None
elif ready_base is None:
    next_title, next_detail, next_label = (
        "上传你的第一份资料", "选中知识库后直接上传 PDF、Word 或文本；处理成功后即可基于原文提问。", "去上传资料")
    next_target = bases[0]["id"]
else:
    next_title, next_detail, next_label = (
        "从资料里找答案", f"{ready_base['name']} 已有可检索文档，开始提问并核对来源。", "开始问答")
    next_target = ready_base["id"]

section_heading("继续你的工作", "根据当前资料状态推荐下一步")
with st.container(border=True, key="rag_next_step"):
    intro, action = st.columns([4, 1], gap="medium", vertical_alignment="center")
    with intro:
        st.caption("建议的下一步 · 来自当前账号的真实数据")
        st.subheader(next_title, anchor=False)
        st.write(next_detail)
    with action:
        if st.button(next_label, key="rag_next_action", type="primary", width="stretch",
                     icon=":material/arrow_forward:"):
            if next_target:
                st.session_state["rag_managed_kb"] = next_target
            st.switch_page("app_pages/chat.py" if ready_base else "app_pages/knowledge.py")

with st.container(key="rag_onboarding_steps"):
    steps = (
        ("01", "建立知识库", bool(bases)),
        ("02", "上传并处理", bool(ready_base)),
        ("03", "开始会话", counts["conversations"] > 0),
    )
    for column, (number, label, done) in zip(st.columns(3, gap="small"), steps):
        with column:
            st.badge("已完成" if done else "待开始", color="green" if done else "gray")
            st.markdown(f"**{number} · {label}**")

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
    bases = bases[:4]
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