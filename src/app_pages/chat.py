"""Persistent, owner-scoped AI chat over managed knowledge bases."""

import streamlit as st

from modules.auth_ui import require_user
from modules.citations import original_page
from modules.long_term_memory import add_memory, delete_memory
from modules.managed_ui import render_model_settings
from modules.repository import Repository
from modules.style_profile import (
    STYLE_OPTIONS, clear_style_data, confirm_style_proposal, get_style_profiles,
    list_style_proposals, reject_style_proposal, remove_style_profile,
    set_style_enabled, style_enabled,
)
from modules.workflow import abandon_turn, checkpoint_stage, stream_answer_turn
from modules.ui_kit import empty, page_title, section_heading

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("对话 / 有据可查", "AI 智能问答",
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
has_documents = any(doc["status"] == "success" for doc in repo.list_documents(user["id"], kb_id))
memory_enabled = repo.memory_enabled(user["id"], kb_id)
can_start = has_documents or (memory_enabled and any(mem["indexed"] for mem in repo.list_memories(user["id"], kb_id)))
with create:
    if st.button("新建会话", key="managed_new_session", icon=":material/add:", width="stretch",
                 type="primary", disabled=not can_start):
        st.session_state["rag_managed_session"] = repo.create_session(user["id"], kb_id)
        st.rerun()
with st.sidebar.expander("检索与对话偏好", expanded=False, icon=":material/tune:"):
    mode_label = st.selectbox("检索策略", ["BM25 + 向量融合", "纯向量", "融合 + 模型重排"], key="rag_search_mode")
    mode = {"BM25 + 向量融合": "hybrid", "纯向量": "vector", "融合 + 模型重排": "rerank"}[mode_label]
    top_k = st.slider("引用片段数", 1, 12, 6, key="rag_top_k")
    history_turns = st.slider("近期对话轮数", 2, 20, 8, key="rag_history_turns")
    st.caption("宽泛总结默认覆盖至少 10 条证据；选 1–5 条可精简。证据越多，耗时和模型费用可能越高。模型重排另需一次请求。")
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

with st.sidebar.expander("回答风格画像（需确认）", expanded=False, icon=":material/tune:"):
    style_on = style_enabled(repo, user["id"], kb_id)
    style_toggle = st.toggle("观察明确的回答风格反馈", value=style_on,
                             key=f"rag_style_on_{user['id']}_{kb_id}")
    if style_toggle != style_on:
        set_style_enabled(repo, user["id"], kb_id, style_toggle)
        style_on = style_toggle
    st.caption("默认关闭。同一知识库近 30 天内，3 条不同且已完成的用户消息表达同一风格诉求后才提出候选；"
               "确认前不影响回答。只提取明确反馈，不分析文档，也不额外调用模型。关闭暂停观察与应用，记录仍保留。")
    if style_on:
        proposals = list_style_proposals(repo, user["id"], kb_id)
        if not proposals:
            st.caption("暂无待确认的风格候选。")
        for proposal in proposals:
            label = STYLE_OPTIONS[proposal["dimension"]][proposal["value"]][0]
            st.markdown(f"**候选：{label}**")
            st.caption("判断依据（来自你的已完成问答）：")
            for evidence in proposal["evidence"]:
                st.text(f"· {evidence['evidence']} · {evidence['created_at'][:10]}")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("确认应用", key=f"rag_style_accept_{proposal['id']}", width="stretch"):
                    try:
                        confirm_style_proposal(repo, user["id"], kb_id, proposal["id"])
                        st.rerun()
                    except (PermissionError, ValueError) as exc:
                        st.error(str(exc))
            with col_no:
                if st.button("不采用", key=f"rag_style_reject_{proposal['id']}", width="stretch"):
                    reject_style_proposal(repo, user["id"], kb_id, proposal["id"])
                    st.rerun()
    profiles = get_style_profiles(repo, user["id"], kb_id)
    st.markdown("**已确认的偏好**" if profiles else "尚未确认任何回答风格。")
    for profile in profiles:
        label = STYLE_OPTIONS[profile["dimension"]][profile["value"]][0]
        col_label, col_delete = st.columns([3, 1], vertical_alignment="center")
        col_label.write(label)
        if col_delete.button("移除", key=f"rag_style_remove_{kb_id}_{profile['dimension']}"):
            remove_style_profile(repo, user["id"], kb_id, profile["dimension"])
            st.rerun()
    if profiles or (style_on and list_style_proposals(repo, user["id"], kb_id)):
        confirm_clear = st.checkbox("确认清空此知识库的风格数据", key=f"rag_style_clear_confirm_{kb_id}")
        if st.button("清空风格数据", disabled=not confirm_clear, key=f"rag_style_clear_{kb_id}"):
            clear_style_data(repo, user["id"], kb_id)
            st.rerun()

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
    with st.container(border=True, key="rag_chat_onboarding"):
        if not can_start:
            st.subheader("先准备一份可检索资料", anchor=False)
            st.write("当前知识库还没有处理成功的文档。上传 PDF、Word 或文本，等待处理完成后再回来提问。")
            st.page_link("app_pages/knowledge.py", label="前往上传资料", icon=":material/upload_file:")
        else:
            st.subheader("资料已就绪，可以开始提问", anchor=False)
            st.write("点击右上角「新建会话」，或者从左侧继续已有的讨论。")
    st.stop()

records = repo.list_messages(user["id"], sid)
section_heading("当前对话" if records else "开始提问",
                f"{len(records)} 条消息 · {kb['name']}" if records else f"当前知识库 · {kb['name']}")
with st.container(key="rag_chat_feed"):
    if not records:
        if not can_start:
            empty("还没有可检索内容", "先完成文档上传与处理，提问入口才会开放。")
            st.page_link("app_pages/knowledge.py", label="去上传资料", icon=":material/upload_file:")
        else:
            empty("从你的资料里找答案", "输入问题，回答会附上可核对的文档来源。")
            st.caption("试试这样问：总结核心观点 · 作者提出了什么方法？ · 哪一段支持这个结论？")
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
                    if source.get("edited") or source.get("location", "").endswith(" · 知识库修订"):
                        st.info("这条证据是知识库中的人工修订，不代表上传文件原文已更改。")
                    chunk = repo.get_chunk(user["id"], kb_id, source.get("chunk_id")) if source.get("chunk_id") else None
                    st.text(chunk["content"] if chunk else source.get("preview", "原文片段已更新或删除。"))
                    if chunk and source.get("document_id") and source.get("location", "").endswith(" 页"):
                        if st.checkbox("显示该页完整的 PDF 提取文本", key=f"full_page_{msg['id']}_{selected}"):
                            st.text(original_page(repo, user["id"], kb_id, source) or "原始 PDF 已不可用或该页无法提取文本。")

can_ask = bool(settings.api_key and settings.chat_model and settings.base_url == kb["base_url"]
               and can_start)
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
if st.session_state.pop("rag_style_proposed", False):
    st.info("本轮产生新的回答风格候选，请在左侧「回答风格画像」中审阅并确认；候选尚未应用。")
if st.session_state.pop("rag_rerank_fallback", False):
    st.warning("重排模型本轮不可用，已回退到 Hybrid 检索并完成回答；可以检查模型连接后重试。")
if st.session_state.pop("rag_style_error", False):
    st.warning("回答已保存，但本轮风格观察失败；不影响问答内容。")
if mode == "rerank":
    st.caption("当前开启模型重排：额外发送候选片段到同一模型接口，可能增加等待时间与费用。")
pending = repo.pending_turn(user["id"], sid)
resume = False
if pending:
    st.warning("这轮问答被中断或失败，但任务已保存在本机。你可以从已完成的节点继续，或放弃后重新提问。")
    st.caption(f"待恢复问题：{pending['question'][:160]} · {checkpoint_stage(repo, user['id'], sid)}")
    resume = st.button("从检查点继续", icon=":material/replay:", disabled=not can_ask)
    if st.button("放弃本轮任务", icon=":material/delete:"):
        try:
            abandon_turn(repo, user["id"], sid, pending["id"])
            st.rerun()
        except (ValueError, PermissionError, OSError) as exc:
            st.error(str(exc))
regenerate = False
if records and records[-1]["role"] == "assistant":
    regenerate = st.button("重新生成最后一条回答", icon=":material/refresh:",
                           disabled=not can_ask or pending is not None)
question = st.chat_input("针对当前知识库提问…", disabled=not can_ask or pending is not None,
                         key="rag_managed_question", submit_mode="stop")
if question or regenerate or resume:
    try:
        if not resume:
            asked = records[-2]["content"] if regenerate else question
            pending = repo.begin_turn(user["id"], sid, asked, mode, top_k,
                                      history_turns, settings.chat_model, regenerate=regenerate)
        asked = pending["question"]
        replace = bool(pending["regenerate"])
        with st.container(key="rag_stream_reply"):
            if not replace:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(asked)
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                st.caption("正在理解问题、检索证据并生成回答…")
                metrics = {}
                st.write_stream(stream_answer_turn(
                    repo, user["id"], kb, sid, asked,
                    settings.api_key, pending["chat_model"], mode=pending["mode"],
                    top_k=pending["top_k"], history_turns=pending["history_turns"],
                    regenerate=replace, metrics=metrics, run_id=pending["id"],
                ))
        if metrics.get("rerank_fallback"):
            st.session_state["rag_rerank_fallback"] = True
        if metrics.get("style_proposals"):
            st.session_state["rag_style_proposed"] = True
        if metrics.get("style_error"):
            st.session_state["rag_style_error"] = True
        if metrics.get("memories_saved"):
            st.session_state["rag_memory_saved"] = metrics["memories_saved"]
        if metrics.get("memory_recall_error"):
            st.session_state["rag_memory_recall_error"] = True
        if metrics.get("memory_error"):
            st.session_state["rag_memory_error"] = True
        st.rerun()
    except Exception as exc:
        # Gateways may echo private documents or keys in errors.
        st.error(f"问答失败（{type(exc).__name__}）。请检查接口、模型和索引后重新发送。")
