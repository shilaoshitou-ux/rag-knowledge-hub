"""Persistent, owner-scoped AI chat over managed knowledge bases."""

import streamlit as st

from modules.auth_ui import require_user
from modules.citations import original_page
from modules.long_term_memory import add_memory, delete_memory
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
    top_k = st.slider("引用片段数", 1, 12, 6, key="rag_top_k")
    history_turns = st.slider("近期对话轮数", 2, 20, 8, key="rag_history_turns")
    st.caption("宽泛总结默认覆盖至少 10 条证据；选 1–5 条可精简。证据越多，耗时和模型费用可能越高。模型重排另需一次请求。")
memory_enabled = repo.memory_enabled(user["id"], kb_id)
with st.sidebar.expander("长期记忆（可选）", expanded=False, icon=":material/psychology:"):
    enabled = st.toggle("在此知识库跨会话使用记忆", value=memory_enabled,
                        key=f"rag_memory_on_{user['id']}_{kb_id}")
    if enabled != memory_enabled:
        repo.set_memory_enabled(user["id"], kb_id, enabled)
        memory_enabled = enabled
    st.caption("默认关闭。开启后只从你明确说过的话中筛选稳定偏好、目标与约束；会额外调用聊天/向量模型。"
               "只在本账号的当前知识库跨会话使用；用户原话可能发送到当前向量服务并以明文保存在本机。"
               "请勿输入密码、密钥等隐私；自动过滤无法保证识别所有敏感内容。关闭仅暂停读写，已有记录需手动删除。")
    with st.form(f"rag_manual_memory_{kb_id}", clear_on_submit=True):
        memory_quote = st.text_area("手动添加我的记忆", max_chars=180, key=f"rag_memory_input_{kb_id}",
                                    placeholder="例如：我的秋招目标是 Python 后端和 RAG 开发岗位。")
        category_label = st.selectbox("类型", ["长期目标", "个人偏好", "明确约束", "自述事实"],
                                      key=f"rag_memory_category_{kb_id}")
        add_clicked = st.form_submit_button("保存这条记忆", disabled=not memory_enabled,
                                            key=f"rag_memory_submit_{kb_id}")
    if add_clicked:
        category = {"长期目标": "goal", "个人偏好": "preference", "明确约束": "constraint", "自述事实": "fact"}[category_label]
        try:
            if settings.base_url != kb["base_url"] or (kb["embedding_provider"] == "company" and not settings.api_key):
                raise ValueError("请先配置与当前知识库一致的模型服务；在线向量化还需密钥。")
            created = add_memory(repo, user["id"], kb, None, memory_quote.strip(), category, 1.0, settings.api_key)
            st.success("记忆已保存。" if created else "这条记忆已存在。")
            st.rerun()
        except (ValueError, PermissionError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("记忆向量化失败；没有保存，请检查当前向量模型。")
    memories = repo.list_memories(user["id"], kb_id)
    if memories:
        by_id = {entry["id"]: entry for entry in memories}
        selected_id = st.selectbox("已保存的记忆", list(by_id),
                                   format_func=lambda mid: by_id[mid]["quote"][:32],
                                   key=f"rag_memory_selected_{kb_id}")
        st.caption(f"{by_id[selected_id]['category']} · {by_id[selected_id]['created_at']}")
        st.write(by_id[selected_id]["quote"])
        confirmed = st.checkbox("确认删除选中的记忆", key=f"rag_memory_confirm_{kb_id}_{selected_id}")
        if st.button("删除选中的记忆", disabled=not confirmed, key=f"rag_memory_delete_{kb_id}"):
            try:
                delete_memory(repo, user["id"], kb, selected_id, settings.api_key)
                st.rerun()
            except Exception:
                st.error("删除失败，记忆未从当前库移除，请稍后重试。")
    else:
        st.caption("当前知识库尚无长期记忆；可手动添加，或开启后在问答中明确表达长期目标。")

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
               and (any(doc["status"] == "success" for doc in repo.list_documents(user["id"], kb_id))
                    or (memory_enabled and any(item["indexed"] for item in repo.list_memories(user["id"], kb_id)))))
if settings.base_url != kb["base_url"]:
    st.warning("当前服务地址与该知识库创建时不同，请切换到相同接口后再问答。")
elif not settings.api_key:
    st.warning("请展开左侧的“模型连接与密钥”，填写或加载此账号的密钥。")
elif not any(doc["status"] == "success" for doc in repo.list_documents(user["id"], kb_id)) \
        and not (memory_enabled and any(item["indexed"] for item in repo.list_memories(user["id"], kb_id))):
    st.info("知识库里尚无处理成功的文档。请到“我的知识库”上传并等待处理。")
if st.session_state.pop("rag_memory_saved", 0):
    st.success("本轮从你的原话中新增了长期记忆；可在左侧查看或删除。")
if st.session_state.pop("rag_memory_recall_error", False):
    st.warning("本轮未能读取长期记忆；请检查向量模型连接。本次回答不会使用过往个人记忆。")
if st.session_state.pop("rag_memory_error", False):
    st.warning("本轮问答已保存，但长期记忆提取未成功。请检查模型/向量服务；不影响已有对话。")
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
        if metrics.get("memories_saved"):
            st.session_state["rag_memory_saved"] = metrics["memories_saved"]
        if metrics.get("memory_recall_error"):
            st.session_state["rag_memory_recall_error"] = True
        if metrics.get("memory_error"):
            st.session_state["rag_memory_error"] = True
        st.session_state.pop("rag_pending_question", None)
        st.rerun()
    except Exception as exc:
        # Gateways may echo private documents or keys in errors.
        st.error(f"问答失败（{type(exc).__name__}）。请检查接口、模型和索引后重新发送。")
