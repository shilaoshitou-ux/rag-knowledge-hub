"""Persistent, owner-scoped AI chat over managed knowledge bases."""

import streamlit as st

from modules.auth_ui import require_user
from modules.citations import original_page
from modules.managed_ui import render_model_settings
from modules.repository import Repository
from modules.workflow import stream_answer_turn
from modules.ui_kit import empty, page_title, section_heading

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("ASSISTANT / ASK WITH EVIDENCE", "AI 智能问答",
           "与你的知识库自然对话。回答基于检索证据，并保留可核对的原文出处。")
bases = repo.list_bases(user["id"])
if not bases:
    empty("还没有可选知识库", "先创建知识库并上传资料，处理完成后即可开始问答。")
    st.page_link("app_pages/knowledge.py", label="前往我的知识库", icon=":material/arrow_forward:")
    st.stop()
kb_ids = [item["id"] for item in bases]
previous = st.session_state.get("rag_managed_kb")
initial = kb_ids.index(previous) if previous in kb_ids else 0
selector, create = st.columns([4, 1], gap="medium", vertical_alignment="bottom")
with selector:
    kb_id = st.selectbox("当前知识库", kb_ids, index=initial,
                         format_func=lambda value: next(item["name"] for item in bases if item["id"] == value),
                         key="rag_managed_kb")
kb = repo.get_base(user["id"], kb_id)
with create:
    if st.button("新建会话", key="managed_new_session", icon=":material/add:", width="stretch", type="primary"):
        st.session_state["rag_managed_session"] = repo.create_session(user["id"], kb_id)
        st.rerun()
with st.sidebar.expander("检索与对话偏好", expanded=False, icon=":material/tune:"):
    mode_label = st.selectbox("检索策略", ["BM25 + 向量融合", "纯向量", "融合 + 模型重排"], key="rag_search_mode")
    mode = {"BM25 + 向量融合": "hybrid", "纯向量": "vector", "融合 + 模型重排": "rerank"}[mode_label]
    top_k = st.slider("引用片段数", 1, 8, 6, key="rag_top_k")
    history_turns = st.slider("近期对话轮数", 2, 20, 8, key="rag_history_turns")
    st.caption("模型重排会产生额外请求。")
with st.sidebar.expander("最近会话", expanded=True, icon=":material/history:"):
    sessions = repo.list_sessions(user["id"], kb_id)[:10]
    if not sessions:
        st.caption("当前库还没有会话。")
    for item in sessions:
        if st.button(item["title"][:32], key=f"session_{item['id']}", width="stretch"):
            st.session_state["rag_managed_session"] = item["id"]
            st.rerun()

sid = st.session_state.get("rag_managed_session")
session = repo.get_session(user["id"], sid) if sid else None
if (not session or session["kb_id"] != kb_id) and sessions:
    # A hard browser refresh clears Streamlit session_state, not persisted sessions.
    # Resume the newest conversation for this knowledge base by default.
    sid = sessions[0]["id"]
    session = repo.get_session(user["id"], sid)
    st.session_state["rag_managed_session"] = sid
if not session or session["kb_id"] != kb_id:
    empty("开启一段有据可循的对话", "新建会话开始提问，也可以从左侧继续之前的讨论。")
    if st.button("开始一段新对话", icon=":material/add:", type="primary"):
        st.session_state["rag_managed_session"] = repo.create_session(user["id"], kb_id)
        st.rerun()
    st.stop()

records = repo.list_messages(user["id"], sid)
section_heading("当前对话", f"{len(records)} 条消息 · {kb['name']}")
with st.container(key="rag_chat_feed"):
    if not records:
        empty("准备好了，问我任何关于当前资料的问题", "你可以提出具体问题，或让助手梳理重点、比较方法、追问细节。")
    for msg in records:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                with st.expander("复制回答纯文本", icon=":material/content_copy:"):
                    st.code(msg["content"], language=None)
            if msg["sources"]:
                with st.expander(f"查看引用来源 · {len(msg['sources'])} 条", icon=":material/source:"):
                    citation_key = f"rag_citation_{msg['id']}"
                    for i, source in enumerate(msg["sources"], 1):
                        if st.button(f"[{i}] {source['name']} · {source['location']}",
                                     key=f"cite_{msg['id']}_{i}"):
                            st.session_state[citation_key] = i - 1
                    selected = min(st.session_state.get(citation_key, 0), len(msg["sources"]) - 1)
                    source = msg["sources"][selected]
                    st.caption(f"第 {selected+1} 条证据 · {source['name']} · {source['location']}")
                    chunk = repo.get_chunk(user["id"], kb_id, source.get("chunk_id")) if source.get("chunk_id") else None
                    st.text(chunk["content"] if chunk else source.get("preview", "原文片段已更新或删除。"))
                    if chunk and source.get("document_id") and source.get("location", "").endswith(" 页"):
                        if st.checkbox("显示该页完整的 PDF 提取文本", key=f"full_page_{msg['id']}_{selected}"):
                            st.text(original_page(repo, user["id"], kb_id, source) or "原始 PDF 已不可用或该页无法提取文本。")

can_ask = bool(settings.api_key and settings.chat_model and settings.base_url == kb["base_url"]
               and any(doc["status"] == "success" for doc in repo.list_documents(user["id"], kb_id)))
if settings.base_url != kb["base_url"]:
    st.warning("当前服务地址与该知识库创建时不同，请切换到相同接口后再问答。")
elif not settings.api_key:
    st.warning("请展开左侧的“模型连接与密钥”，填写或加载此账号的密钥。")
elif not any(doc["status"] == "success" for doc in repo.list_documents(user["id"], kb_id)):
    st.info("知识库里尚无处理成功的文档。请到“我的知识库”上传并等待处理。")
if mode == "rerank":
    st.caption("当前开启模型重排：额外发送候选片段到同一模型接口，可能增加等待时间与费用。")
pending = st.session_state.get("rag_pending_question")
matching = bool(pending and pending[:3] == (user["id"], kb_id, sid))
resume = False
if matching:
    st.warning("上次生成被中断或失败，尚未写入会话。可以重新发送同一个问题。")
    resume = st.button("重新发送未完成的问题", icon=":material/replay:", disabled=not can_ask)
regenerate = False
if records and records[-1]["role"] == "assistant":
    regenerate = st.button("重新生成最后一条回答", icon=":material/refresh:", disabled=not can_ask)
question = st.chat_input("针对当前知识库提问…", disabled=not can_ask,
                         key="rag_managed_question", submit_mode="stop")
if question or regenerate or resume:
    asked = pending[3] if resume else (records[-2]["content"] if regenerate else question)
    replace = pending[4] if resume else regenerate
    st.session_state["rag_pending_question"] = (user["id"], kb_id, sid, asked, replace)
    try:
        with st.container(key="rag_stream_reply"):
            if not replace:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(asked)
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                st.caption("正在理解问题、检索证据并生成回答…")
                metrics = {}
                st.write_stream(stream_answer_turn(
                    repo, user["id"], kb, sid, asked,
                    settings.api_key, settings.chat_model, mode=mode, top_k=top_k,
                    history_turns=history_turns, regenerate=replace, metrics=metrics,
                ))
        st.session_state.pop("rag_pending_question", None)
        st.rerun()
    except Exception as exc:
        # Gateways may echo private documents or keys in errors.
        st.error(f"问答失败（{type(exc).__name__}）。请检查接口、模型和索引后重新发送。")
