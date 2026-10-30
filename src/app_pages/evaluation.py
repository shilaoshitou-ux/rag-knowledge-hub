"""Real local evaluation of managed knowledge-base retrieval strategies."""

import json

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.evaluation import check_question_documents, chunk_manifest, evaluate, parse_cases, summarize
from modules.managed_ui import render_model_settings
from modules.repository import Repository

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("QUALITY / EVALUATION", "RAG 效果评测", "上传一份问题集，比较三种检索方式能否找到正确资料。")
bases = repo.list_bases(user["id"])
if not bases:
    st.info("请先创建知识库，上传并成功处理文档。")
    st.page_link("app_pages/knowledge.py", label="前往我的知识库", icon=":material/arrow_forward:")
    st.stop()

with st.container(border=True, key="rag_eval_controls"):
    st.markdown("#### 第一步：选择知识库")
    kb_id = st.selectbox("评测哪个知识库？", [kb["id"] for kb in bases],
                         format_func=lambda value: next(kb["name"] for kb in bases if kb["id"] == value))
    kb = repo.get_base(user["id"], kb_id)
    docs = repo.list_documents(user["id"], kb_id)
    successful = [doc for doc in docs if doc["status"] == "success"]
    st.caption(f"当前有 {len(successful)} 份处理成功的文档；上传完成但仍在处理的文档暂时不能参加评测。")
    if len(successful) == 1:
        st.info("目前只有 1 份文档处理成功。如果题集标注了更多文件，请先确认它们都在这个库里并处理成功。")
    if not successful:
        st.page_link("app_pages/knowledge.py", label="查看文档处理状态", icon=":material/arrow_forward:")

    st.markdown("#### 第二步：上传问题集")
    file = st.file_uploader("选择 JSON 问题集（1–20 道题）", type=["json"])
    valid_set = False
    if file is not None:
        try:
            question_count, unavailable = check_question_documents(file.getvalue(), docs)
            if unavailable:
                st.error(f"这份题集有 {question_count} 道题，但以下 {len(unavailable)} 份相关文档尚不能评测：",
                         title="先核对文档")
                status_text = {"waiting": "等待处理", "processing": "处理中", "failed": "处理失败",
                               "deleted": "已删除", "未上传": "未上传到当前库（或文件名不一致）"}
                for name, status in unavailable:
                    st.write(f"- {name}：{status_text.get(status, status)}")
                st.page_link("app_pages/knowledge.py", label="去知识库查看或重试处理",
                             icon=":material/arrow_forward:")
            else:
                st.success(f"已核对 {question_count} 道题：标注的文档都在当前库且处理成功。")
                valid_set = True
        except ValueError as exc:
            st.error(str(exc), title="问题集无法读取")
    else:
        st.caption("题集中的相关文档名必须与该库处理成功的文件名完全一致。")

    st.markdown("#### 第三步：运行并查看结果")
    k = st.slider("每题最多检索几个片段（K）", 1, 10, 4)
    semantic_judge = False
    with st.expander("高级选项：片段标注、语义裁判与指标说明（第一次可跳过）"):
        st.caption("参考答案短语包含率不等于回答正确率；要写简历准确率，请人工核对答案和引用。")
        st.caption("Recall@K 是找到的相关项比例，MRR 衡量第一个相关项排得多靠前。片段级指标需人工标注 relevant_chunks。")
        st.code('[{"question":"关键要求是什么？","relevant_documents":["文件.pdf"],"expected_answer":"可选参考短语"}]', language="json")
        if st.button("生成当前片段清单", key="rag_eval_make_manifest"):
            st.session_state["rag_eval_manifest"] = (user["id"], kb_id, chunk_manifest(repo, user["id"], kb_id))
        manifest = st.session_state.get("rag_eval_manifest")
        if manifest and manifest[:2] == (user["id"], kb_id):
            st.caption(f"当前有效片段：{len(manifest[2])} 条；页面仅预览前 100 条，下载包含全部。")
            st.dataframe(manifest[2][:100], hide_index=True, alt="当前知识库片段 ID、文件和位置清单")
            st.download_button("下载片段清单 JSON", json.dumps(manifest[2], ensure_ascii=False, indent=2),
                               file_name="rag-active-chunks.json", mime="application/json")
        semantic_judge = st.checkbox("额外调用模型做语义裁判（增加费用，仍需人工复核）")
    ready = bool(settings.api_key and settings.chat_model and settings.base_url == kb["base_url"])
    if settings.base_url != kb["base_url"]:
        st.warning("侧栏模型接口与该知识库不一致，请先切换。")
    elif not settings.api_key or not settings.chat_model:
        st.warning("先在侧栏配置模型，才能运行评测。")
    if st.button("运行三种检索方式的真实评测", type="primary", disabled=not ready or not valid_set):
        try:
            cases = parse_cases(file.getvalue(), repo, user["id"], kb_id)
            with st.status(f"正在评测 {len(cases)} 道题 × 3 种检索方式…", expanded=True):
                result = evaluate(repo, user["id"], kb, cases, settings.api_key, settings.chat_model, k,
                                  semantic_judge=semantic_judge)
            st.session_state["rag_eval_result"] = (user["id"], kb_id, result, file.getvalue(), k)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"评测中断（{type(exc).__name__}）。请检查模型服务、向量索引及日志。")
saved = st.session_state.get("rag_eval_result")
if (saved and len(saved) == 5 and saved[0] == user["id"] and saved[1] == kb_id
        and file is not None and saved[3] == file.getvalue() and saved[4] == k):
    st.subheader("实际运行结果")
    st.info("先比较三种方式的 Recall@K、MRR 和检索耗时；参考答案包含率不是回答准确率。")
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
