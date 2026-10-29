"""Real local evaluation of managed knowledge-base retrieval strategies."""

import json

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.evaluation import chunk_manifest, evaluate, parse_cases, summarize
from modules.managed_ui import render_model_settings
from modules.repository import Repository

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("QUALITY / EVALUATION", "RAG 效果评测", "基于你标注的问题集真实运行三种检索策略，量化召回、排序和耗时。")
bases = repo.list_bases(user["id"])
if not bases:
    st.info("请先创建知识库，上传并成功处理文档。")
    st.page_link("app_pages/knowledge.py", label="前往我的知识库", icon=":material/arrow_forward:")
    st.stop()
kb_id = st.selectbox("评测知识库", [kb["id"] for kb in bases],
                     format_func=lambda value: next(kb["name"] for kb in bases if kb["id"] == value))
kb = repo.get_base(user["id"], kb_id)
with st.container(border=True, key="rag_eval_controls"):
    st.markdown("每题标注当前库的相关文档完整文件名；单篇论文可选标注 `relevant_chunks`（有效片段 ID），否则文档级指标无法判断是否找到真正的证据。`expected_answer` 仅用于机械字符串包含率，语义裁判也需要人工复核。")
    st.code('[{"question":"关键要求是什么？","relevant_documents":["文件.pdf"],"relevant_chunks":["粘贴当前片段的完整 ID"],"expected_answer":"可选参考短语"}]', language="json")
    st.caption("文档级与证据级 Recall@K 都按相关项命中比例计算；MRR 是首个相关项的倒数排名。片段 ID 更新后需重新标注；评测发起真实请求并可能产生费用。")
    if len([doc for doc in repo.list_documents(user["id"], kb_id) if doc["status"] == "success"]) == 1:
        st.info("当前仅一份文档：文档级召回/MRR 很容易满分，建议标注相关片段再比较策略。")
    with st.expander("可选：生成片段清单，标注证据级指标"):
        st.caption("点击后导出当前有效片段的完整 ID、文件名、位置及短预览。依据原文人工挑选能回答问题的片段，将 ID 写入相关问题的 relevant_chunks。")
        if st.button("生成当前片段清单", key="rag_eval_make_manifest"):
            st.session_state["rag_eval_manifest"] = (user["id"], kb_id, chunk_manifest(repo, user["id"], kb_id))
        manifest = st.session_state.get("rag_eval_manifest")
        if manifest and manifest[:2] == (user["id"], kb_id):
            st.caption(f"当前有效片段：{len(manifest[2])} 条；只预览前 100 条，下载包含全部。")
            st.dataframe(manifest[2][:100], hide_index=True, alt="当前知识库片段 ID、文件和位置清单")
            st.download_button("下载当前片段清单 JSON", json.dumps(manifest[2], ensure_ascii=False, indent=2),
                               file_name="rag-active-chunks.json", mime="application/json")
    file = st.file_uploader("上传 JSON 问题集（1–20 条）", type=["json"])
    k = st.slider("检索 K", 1, 10, 4)
    semantic_judge = st.checkbox("额外调用模型评估语义正确性（每份参考答案 × 3 次，会增加费用）")
    ready = bool(settings.api_key and settings.chat_model and settings.base_url == kb["base_url"])
    if settings.base_url != kb["base_url"]:
        st.warning("请在侧栏切换到该库绑定的模型接口。")
    if st.button("运行三组真实评测", type="primary", disabled=not ready or file is None):
        try:
            cases = parse_cases(file.getvalue(), repo, user["id"], kb_id)
            with st.status(f"评测 {len(cases)} 个问题 × 3 种策略…", expanded=True):
                result = evaluate(repo, user["id"], kb, cases, settings.api_key, settings.chat_model, k, semantic_judge=semantic_judge)
            st.session_state["rag_eval_result"] = (user["id"], kb_id, result)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"评测中断（{type(exc).__name__}）。请检查模型服务、向量索引及日志。")
saved = st.session_state.get("rag_eval_result")
if saved and saved[0] == user["id"] and saved[1] == kb_id:
    st.subheader("实际运行结果")
    st.caption("证据级指标仅统计已标注 relevant_chunks 的题；检索耗时不含生成，总耗时可能包含生成和可选裁判。无标注或未实际运行时不显示成绩。")
    st.dataframe(summarize(saved[2]), hide_index=True, alt="三种检索策略的文档级、证据级评测指标与耗时")
    with st.expander("逐题明细与模型回答"):
        for row in saved[2]:
            st.markdown(f"**{row['strategy']} · {row['question']}**")
            evidence = (f" · 证据召回 {row['evidence_recall_at_k']} / 证据 MRR {row['evidence_reciprocal_rank']}"
                        if row.get("evidence_recall_at_k") is not None else "")
            st.caption(f"文档召回 {row['recall_at_k']} / MRR {row['reciprocal_rank']}"
                       f"{evidence} · 检索 {row.get('retrieval_ms', row['elapsed_ms'])} ms / 总 {row['elapsed_ms']} ms")
            if row["answer"]:
                st.markdown(row["answer"])
    st.download_button("下载结果 JSON", json.dumps(saved[2], ensure_ascii=False, indent=2),
                       file_name="rag-evaluation.json", mime="application/json")
