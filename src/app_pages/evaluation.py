"""Real local evaluation of managed knowledge-base retrieval strategies."""

import json

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.evaluation import evaluate, parse_cases, summarize
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
    st.markdown("问题集格式：每条填写真实问题及相关文档的**完整文件名**。可选 `expected_answer` 用于参考字符串包含率；这是机械指标，不代表语义正确率；可另外启用真实模型裁判，其结论仍需人工复核。")
    st.code('[{"question":"该文档的关键要求是什么？","relevant_documents":["你的文件名.pdf"],"expected_answer":"可选参考关键词"}]', language="json")
    st.caption("如某条有多个相关文档，Recall@K 为召回的相关文档占标注文档的比例；MRR 取首个相关文档的倒数排名。评测会对所选接口发起真实请求并产生费用。")
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
    st.dataframe(summarize(saved[2]), hide_index=True)
    with st.expander("逐题明细与模型回答"):
        for row in saved[2]:
            st.markdown(f"**{row['strategy']} · {row['question']}**")
            st.caption(f"召回 {row['recall_at_k']} · MRR {row['reciprocal_rank']} · {row['elapsed_ms']} ms")
            if row["answer"]:
                st.markdown(row["answer"])
    st.download_button("下载结果 JSON", json.dumps(saved[2], ensure_ascii=False, indent=2),
                       file_name="rag-evaluation.json", mime="application/json")
