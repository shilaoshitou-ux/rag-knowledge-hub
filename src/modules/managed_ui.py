"""Shared model controls for the managed (account-isolated) workspace."""

from dataclasses import dataclass

import streamlit as st

from .credential_ui import render_api_key
from .model_config import (
    CHAT_MODELS, COMPANY_EMBEDDING_MODELS, CUSTOM_MODEL, CUSTOM_SERVICE, DEFAULT_SERVICE,
    DEFAULT_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODELS, OPENAI_EMBEDDING_MODELS,
    OPENAI_SERVICE, SERVICE_OPTIONS, clean_model_id, service_base_url,
)


@dataclass(frozen=True)
class ModelSettings:
    base_url: str | None
    api_key: str
    chat_model: str | None
    embedding_model: str | None
    embedding_provider: str


def render_model_settings(user_id: str) -> ModelSettings:
    with st.sidebar.expander("模型连接与密钥", icon=":material/tune:", expanded=False) as panel:
        st.caption("问答与向量模型可分别配置；密钥只用于当前选定的接口。")
        service = st.selectbox("模型服务", SERVICE_OPTIONS, key="managed_service")
        custom_url = st.text_input("接口 Base URL", key="managed_url", placeholder="https://example.com/v1") if service == CUSTOM_SERVICE else ""
        try:
            base_url = service_base_url(service, custom_url)
        except ValueError as exc:
            st.warning(str(exc))
            base_url = None
        if base_url:
            st.caption(base_url)
        api_key = render_api_key(service, base_url, prefix=f"managed_{user_id[:8]}", scope_prefix=user_id, area=panel)
        st.caption("不同账号的密钥独立保存；旧版个人页面的密钥不会自动给其他账号使用。")
        model_options = [*CHAT_MODELS, CUSTOM_MODEL] if service == DEFAULT_SERVICE else [CUSTOM_MODEL]
        selected = st.selectbox("问答模型", model_options, key=f"managed_chat_{service}")
        typed = st.text_input("问答模型 ID", key=f"managed_chat_custom_{service}") if selected == CUSTOM_MODEL else ""
        route = st.segmented_control("向量化方式", ["在线 API", "本地 Ollama"], default="在线 API", key="managed_embed_route")
        provider = "company" if route == "在线 API" else "ollama"
        if provider == "ollama":
            options = [*LOCAL_EMBEDDING_MODELS, CUSTOM_MODEL]
            default = LOCAL_EMBEDDING_MODELS[0]
        elif service == DEFAULT_SERVICE:
            options = [*COMPANY_EMBEDDING_MODELS, CUSTOM_MODEL]
            default = DEFAULT_EMBEDDING_MODEL
        elif service == OPENAI_SERVICE:
            options = [*OPENAI_EMBEDDING_MODELS, CUSTOM_MODEL]
            default = OPENAI_EMBEDDING_MODELS[0]
        else:
            options = [CUSTOM_MODEL]
            default = CUSTOM_MODEL
        choice = st.selectbox("向量模型", options, index=options.index(default), key=f"managed_embedding_{service}_{provider}")
        custom_embedding = st.text_input("向量模型 ID", key=f"managed_embedding_custom_{service}_{provider}") if choice == CUSTOM_MODEL else ""
        try:
            chat_model = clean_model_id(typed if selected == CUSTOM_MODEL else selected)
            embedding_model = clean_model_id(custom_embedding if choice == CUSTOM_MODEL else choice)
        except ValueError as exc:
            st.warning(str(exc))
            chat_model = embedding_model = None
    st.sidebar.caption(f"当前模型 · {chat_model or '未配置'}")
    return ModelSettings(base_url, api_key, chat_model, embedding_model, provider)