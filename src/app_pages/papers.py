"""Paper workspace backed by managed RAG; no second index or canned answers."""

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.citations import original_page
from modules.managed_ui import render_model_settings
from modules.paper_analysis import run_comparison, run_paper_task, synthesize_comparison
from modules.reading_tasks import READING_TASKS
from modules.repository import Repository

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("RESEARCH / PAPER WORKSPACE", "论文智能分析", "让论文阅读更有结构：独立检索、逐篇分析，所有结论都能回到对应的原文。")
paper_bases = [kb for kb in repo.list_bases(user["id"]) if kb["kind"] == "paper"]
if not paper_bases:
    st.info("还没有论文知识库，请到“我的知识库”创建并上传论文。")
    st.stop()
kb_id = st.selectbox("论文知识库", [kb["id"] for kb in paper_bases],
                     format_func=lambda value: next(kb["name"] for kb in paper_bases if kb["id"] == value))
kb = repo.get_base(user["id"], kb_id)
docs = [doc for doc in repo.list_documents(user["id"], kb_id) if doc["status"] == "success"]
if not docs:
    st.info("此库还没有处理成功的论文。请先上传并等待处理完成。")
    st.stop()
if settings.base_url != kb["base_url"]:
    st.warning("请在侧栏切换到创建论文库时使用的模型接口。")
if not settings.api_key:
    st.warning("请先配置此账号在当前接口的问答密钥。")
ready = bool(settings.api_key and settings.chat_model and settings.base_url == kb["base_url"])
doc_by_id = {doc["id"]: doc for doc in docs}
st.caption(f"可用论文：{len(docs)} 篇 · 当前库向量模型：{kb['embedding_model']}")
mode = st.segmented_control("研读方式", ["单篇精读", "多篇对比"], default="单篇精读")
if mode == "单篇精读":
    selected = st.selectbox("选择论文", list(doc_by_id), format_func=lambda value: doc_by_id[value]["name"])
    task = st.selectbox("研读任务", READING_TASKS, format_func=lambda item: item.title)
    st.caption(task.description)
    custom = st.text_area("提问内容（可修改）", value=task.question, height=110, key=f"paper_question_{task.slug}")
    if st.button("开始分析", type="primary", icon=task.icon, disabled=not ready or not custom.strip()):
        with st.status("正在检索论文证据并分析…", expanded=False):
            try:
                result = run_paper_task(repo, user["id"], kb, selected,
                                        settings.api_key, settings.chat_model, custom, task.search_queries)
                st.session_state["paper_result"] = (user["id"], kb_id, [result])
            except Exception as exc:
                # Remote errors can echo credentials or document text.
                st.error(f"分析失败（{type(exc).__name__}），请检查服务、索引和日志。")
else:
    selected_ids = st.multiselect("选择 2–5 篇论文", list(doc_by_id),
                                  format_func=lambda value: doc_by_id[value]["name"], max_selections=5)
    st.caption("分别检索每篇论文的背景、方法、创新、实验与局限；逐篇显示结构化报告和独立来源，避免混淆论文证据。")
    if st.button("生成对比报告", type="primary", icon=":material/compare_arrows:",
                 disabled=not ready or len(selected_ids) < 2):
        with st.status("逐篇检索、分析与核对来源…", expanded=False):
            try:
                results = run_comparison(repo, user["id"], kb, selected_ids,
                                         settings.api_key, settings.chat_model)
                st.session_state["paper_result"] = (user["id"], kb_id, results)
                try:
                    comparison = synthesize_comparison(results, settings.api_key, settings.chat_model, kb["base_url"])
                    st.session_state["paper_synthesis"] = (user["id"], kb_id, comparison)
                except Exception as exc:
                    st.session_state.pop("paper_synthesis", None)
                    st.warning(f"逐篇分析完成，但综合对比未生成（{type(exc).__name__}）；可在下方分别核对报告。")
            except Exception as exc:
                st.error(f"对比未完成（{type(exc).__name__}）；已完成的单篇记录仍在历史会话中。")

saved = st.session_state.get("paper_result")
if saved and saved[0] == user["id"] and saved[1] == kb_id:
    reports = saved[2]
    st.subheader("逐篇分析结果" if len(reports) > 1 else "分析结果")
    if len(reports) > 1:
        st.info("不同论文分别检索和引用，按同一套五维问题进行并列比较；请核对下方原文证据。")
    synthesis = st.session_state.get("paper_synthesis")
    if len(reports) > 1 and synthesis and synthesis[:2] == (user["id"], kb_id):
        with st.container(border=True):
            st.markdown("### 多篇论文对比")
            st.markdown(synthesis[2])
            st.caption("[A:1] 表示论文 A 的第 1 条原文证据。该表由模型生成，请以下方逐篇证据复核。")
    for index, report in enumerate(reports):
        with st.container(border=True):
            label = f"{chr(65+index)} · " if len(reports) > 1 else ""
            st.markdown(f"### {label}{report['document']['name']} · v{report['document']['version']}")
            st.markdown(report["answer"])
            st.caption(f"检索耗时 {report['metrics'].get('retrieval_ms', 0)} ms · 会话 ID：{report['session_id']}")
            with st.expander(f"查看原文证据 · {len(report['sources'])} 条"):
                for number, source in enumerate(report["sources"], 1):
                    chunk = repo.get_chunk(user["id"], kb_id, source["chunk_id"]) if source.get("chunk_id") else None
                    st.markdown(f"**[{number}] {source['name']} · {source['location']}**")
                    st.text(chunk["content"] if chunk else "该版本已更新或删除；请在历史会话核对原始预览。")
                    if chunk and source.get("location", "").endswith(" 页"):
                        if st.checkbox("查看该页完整提取文本", key=f"paper_full_{report['session_id']}_{number}"):
                            st.text(original_page(repo, user["id"], kb_id, source) or "原始 PDF 已不可用或该页无法提取文本。")
