"""Evidence-first research-paper reading workspace."""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from modules.chatbot import Chatbot
from modules.credential_ui import render_api_key
from modules.embedder import EmbeddingBatchError, UploadedDocument, build_index, index_id
from modules.model_config import (
    CHAT_MODELS, COMPANY_EMBEDDING_MODELS, CUSTOM_MODEL, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SERVICE, OPENAI_SERVICE, CUSTOM_SERVICE, SERVICE_OPTIONS, OPENAI_EMBEDDING_MODELS,
    LOCAL_EMBEDDING_MODELS, clean_model_id, safe_error, service_base_url,
)
from modules.reading_tasks import READING_TASKS
from modules.research_ui import render_research_hero
from modules.utils import show_chinese_upload_style

load_dotenv()
st.set_page_config(page_title="论文研读工作台 · 文档智能助手", page_icon="📖", layout="wide")
show_chinese_upload_style()
render_research_hero(
    "RESEARCH DESK / 证据驱动的阅读",
    "把一篇论文，读成清晰的研究脉络。",
    "从问题与贡献，到方法、实验和局限；每一个可核查的判断，都回到原文出处。",
)
st.space("small")

with st.sidebar:
    st.caption("WORKSPACE / 研读配置")
    st.subheader("模型连接")
    service = st.selectbox("模型服务", SERVICE_OPTIONS, key="rag_service", bind="query-params",
                           help="默认服务已配置；切换服务后需填写对应服务自己的密钥。")
    custom_url = (
        st.text_input("接口 Base URL", placeholder="https://example.com/v1",
                      key="rag_custom_url", bind="query-params")
        if service == CUSTOM_SERVICE else ""
    )
    try:
        base_url = service_base_url(service, custom_url)
        url_error = None
        st.caption(f"请求地址：{base_url}")
    except ValueError as exc:
        base_url = None
        url_error = str(exc)
        st.warning(url_error)

    key = render_api_key(service, base_url, prefix="rag")
    st.divider()
    st.subheader("问答与检索")
    model_options = [*CHAT_MODELS, CUSTOM_MODEL] if service == DEFAULT_SERVICE else [CUSTOM_MODEL]
    model_choice = st.selectbox("问答模型", model_options, key=f"rag_chat_model_{service}")
    custom_model = (
        st.text_input("问答模型 ID", placeholder="填写该服务提供的模型 ID", key=f"rag_custom_chat_model_{service}")
        if model_choice == CUSTOM_MODEL else ""
    )
    embedding_route = st.segmented_control("文档向量化方式", ["在线 API", "本地 Ollama"], default="在线 API")
    # Keep the historical identifier so existing default-service indexes remain reusable.
    embedding_provider = "company" if embedding_route == "在线 API" else "ollama"
    if embedding_provider == "company":
        if service == DEFAULT_SERVICE:
            choices = COMPANY_EMBEDDING_MODELS
            configured_embedding = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        elif service == OPENAI_SERVICE:
            choices = OPENAI_EMBEDDING_MODELS
            configured_embedding = OPENAI_EMBEDDING_MODELS[0]
        else:
            choices = ()
            configured_embedding = ""
        embedding_options = [*choices, "自定义（填写向量化模型 ID）"]
        if service == DEFAULT_SERVICE:
            st.caption("默认推荐 text-embedding-v4；其他模型是否可用以当前密钥与网关为准。")
    else:
        choices = LOCAL_EMBEDDING_MODELS
        embedding_options = [*choices, "自定义（填写本地模型名）"]
        configured_embedding = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODELS[0])
        st.caption("本地向量化需启动 Ollama 并拉取对应模型。")
    initial_embedding = configured_embedding if configured_embedding in choices else embedding_options[-1]
    embedding_choice = st.selectbox(
        "向量化模型", embedding_options,
        index=embedding_options.index(initial_embedding), key=f"rag_embed_{service}_{embedding_provider}",
    )
    if service == DEFAULT_SERVICE and embedding_provider == "company" and embedding_choice in {"text-embedding-3-small", "text-embedding-3-large"}:
        st.warning("当前模型在最近一次网关测试中返回 503/500；建议选 text-embedding-v4。")
    embedding_model = (
        st.text_input(
            "向量化模型 ID", value=configured_embedding if initial_embedding == embedding_options[-1] else "",
            key=f"rag_custom_embed_{service}_{embedding_provider}",
        ) if embedding_choice == embedding_options[-1] else embedding_choice
    )
    try:
        model = clean_model_id(custom_model if model_choice == CUSTOM_MODEL else model_choice)
        embedding_model = clean_model_id(embedding_model)
        model_error = None
    except ValueError as exc:
        model = None
        model_error = str(exc)
        st.warning(model_error)

    with st.expander("检索设置与数据说明", icon=":material/tune:"):
        top_k = st.slider("每次检索片段数", min_value=1, max_value=8, value=4)
        st.caption("预设研读任务至少检索 6 片并覆盖多个主题；自由提问使用这里的片段数。")
        st.caption("在线向量化会发送文档片段；即使用本地向量化，提问时检索片段仍会发送给所选问答服务。")
        st.caption("本应用不读取 CC Switch 配置；密钥仅用于当前选择的接口。")
    if st.button("清空对话", icon=":material/delete_outline:", width="stretch"):
        st.session_state["rag_messages"] = []

with st.container(border=True):
    document_area, guide_area = st.columns([1.75, 1], gap="large")
    with document_area:
        st.subheader("01 · 准备研究资料")
        st.caption("上传论文或辅助材料，建立可按页定位的检索索引。支持 PDF、TXT、CSV；单文件最多 100 MB。")
        uploads = st.file_uploader("选择论文与材料", type=["pdf", "txt", "csv"],
                                   accept_multiple_files=True, key="rag_files")
        files = [UploadedDocument(file.name, file.getvalue()) for file in uploads]
        build = st.button(
            "建立 / 加载资料索引", type="primary", icon=":material/library_books:", key="rag_build",
            disabled=not files or (embedding_provider == "company" and not key) or bool(model_error or url_error),
        )
    with guide_area:
        st.subheader("你的阅读路径")
        st.markdown("**01** 上传材料并建立索引  ")
        st.markdown("**02** 选一个研读任务，或自由提问  ")
        st.markdown("**03** 对照答案中的编号查看原文  ")
        st.caption("扫描版 PDF 需要先 OCR。任务只检索相关片段，不等同于人工通读全文。")

key_id = hashlib.sha256(key.encode("utf-8")).hexdigest() if key else None
chat_scope = hashlib.sha256(repr((base_url, key_id, model)).encode("utf-8")).hexdigest()
if st.session_state.get("rag_chat_scope") != chat_scope:
    st.session_state["rag_messages"] = []
    st.session_state["rag_chat_scope"] = chat_scope

if build:
    try:
        with st.spinner("正在解析资料并建立检索索引…"):
            indexed = build_index(files, key, embedding_model=embedding_model, base_url=base_url,
                                  embedding_provider=embedding_provider)
        if (st.session_state.get("rag_collection_id") != indexed.collection_id or
                st.session_state.get("rag_key_id") != key_id):
            st.session_state["rag_messages"] = []
        st.session_state["rag_collection_id"] = indexed.collection_id
        st.session_state["rag_key_id"] = key_id
        st.session_state["rag_index"] = indexed
        st.success(f"已载入 {indexed.file_count} 份资料、{indexed.chunk_count} 个文本片段" +
                   ("（复用已有索引）" if indexed.cached else "（已完成向量化）"))
    except EmbeddingBatchError as exc:
        st.error(f"建索引失败（向量模型：{embedding_model}）：{exc}")
        if embedding_provider == "company":
            st.info("默认服务请先选择 text-embedding-v4；如果小 TXT 也失败，请核查当前密钥对所选模型的网关路由。")
        else:
            st.info("请检查 Ollama 是否运行、模型是否已拉取、本机内存是否足够。")
    except Exception as exc:
        st.error(f"建索引失败：{safe_error(exc, key)}")

info = st.session_state.get("rag_index")
ready = bool(
    not url_error and not model_error and key and files and info
    and st.session_state.get("rag_collection_id") == index_id(files, embedding_model, base_url, embedding_provider)
    and (embedding_provider != "company" or st.session_state.get("rag_key_id") == key_id)
)
with st.container(border=True):
    if ready:
        st.markdown(f"**资料索引已就绪** · {info.file_count} 份资料 · {info.chunk_count} 个片段 · {model}")
        st.caption("建议先从下面的阅读任务开始。回答里的 [1]、[2] 可在“原文证据”中逐项核查。")
    elif url_error or model_error:
        st.info("请先在左侧完成模型与接口配置。")
    elif not files:
        st.info("从上传一篇论文开始。资料仅在你点击建立索引后才会送去在线向量服务。")
    elif embedding_provider == "company" and not key:
        st.info("请先在左侧填写当前服务的 API 密钥，再建立资料索引。")
    elif not info or st.session_state.get("rag_collection_id") != index_id(files, embedding_model, base_url, embedding_provider):
        st.info("资料已选好，点击“建立 / 加载资料索引”开始。")
    elif not key:
        st.info("本地索引已就绪；提问前还需填写问答模型的 API 密钥。")
    else:
        st.info("模型或密钥已变更，请重新加载该资料索引。")

st.subheader("02 · 选择阅读任务")
st.caption("智能体按任务拆分检索方向，再用相关原文作答；不能从未检索到的内容推断整篇论文。")
selected_task = None
for start in (0, 3):
    for column, task in zip(st.columns(3, gap="medium"), READING_TASKS[start:start + 3]):
        with column, st.container(border=True):
            if st.button(task.title, key=f"rag_task_{task.slug}", icon=task.icon,
                         width="stretch", disabled=not ready):
                selected_task = task
            st.caption(task.description)

st.subheader("03 · 对照原文讨论")
with st.expander("为什么不是普通聊天机器人？", icon=":material/lightbulb:"):
    st.write("读论文的难点不是把全文压缩成一段话，而是定位研究问题、辨别本文方法与相关工作、核对实验条件，并识别证据不足的推断。")
    st.write("这里用“任务引导 → 多主题检索 → 证据约束回答 → 来源核查”的流程辅助阅读。它不替代阅读全文或同行评议；需要完整覆盖时，请继续追问并查看原文。")

if ready:
    st.session_state.setdefault("rag_messages", [])
    for message in st.session_state["rag_messages"]:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(message["question"])
        with st.chat_message("assistant", avatar=":material/menu_book:"):
            st.write(message["answer"])
            if message["sources"]:
                with st.expander(f"原文证据 · {len(message['sources'])} 条", icon=":material/source:"):
                    for number, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**[{number}]**")
                        st.text(f"{source['name']} · {source['location']}")
                        st.text(source["preview"])
    question = st.chat_input("继续追问：方法如何验证？结论有哪些限制？", key="rag_question",
                             disabled=not ready, submit_mode="disable")
    if selected_task or question:
        query = selected_task.question if selected_task else question
        history = [] if selected_task else [
            (msg["question"], msg["answer"]) for msg in st.session_state["rag_messages"]
        ]
        try:
            with st.spinner("检索相关证据并整理回答…"):
                result = Chatbot(model, info.vectorstore, key, max(top_k, 6) if selected_task else top_k,
                                 base_url=base_url).conversational_chat(
                    query, history, search_queries=selected_task.search_queries if selected_task else (),
                )
            sources = [
                {"name": doc.metadata.get("source", "未知文件"),
                 "location": doc.metadata.get("location", "位置未知"),
                 "preview": doc.page_content[:600]}
                for doc in result.sources
            ]
            st.session_state["rag_messages"].append({
                "question": query, "answer": result.answer, "sources": sources,
            })
            st.rerun()
        except Exception as exc:
            st.error(f"研读任务失败：{safe_error(exc, key)}")
else:
    st.caption("资料索引就绪后，阅读任务和自由提问会在这里开放。")