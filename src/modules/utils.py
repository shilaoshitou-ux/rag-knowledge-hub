"""Other pages' shared uploader and model-service settings."""

import streamlit as st

from modules.credential_ui import render_api_key
from modules.model_config import (
    CHAT_MODELS, CUSTOM_MODEL, CUSTOM_SERVICE, DEFAULT_SERVICE, SERVICE_OPTIONS,
    clean_model_id, service_base_url,
)


class Utilities:
    @staticmethod
    def load_model_settings():
        service = st.sidebar.selectbox("模型服务", SERVICE_OPTIONS, key="aux_service", bind="query-params")
        custom_url = (
            st.sidebar.text_input("接口 Base URL", placeholder="https://example.com/v1",
                                  key="aux_custom_url", bind="query-params")
            if service == CUSTOM_SERVICE else ""
        )
        try:
            base_url = service_base_url(service, custom_url)
            st.sidebar.caption(f"请求地址：{base_url}")
        except ValueError as exc:
            st.sidebar.warning(str(exc))
            base_url = None
        key = render_api_key(service, base_url, prefix="aux")
        choices = [*CHAT_MODELS, CUSTOM_MODEL] if service == DEFAULT_SERVICE else [CUSTOM_MODEL]
        choice = st.sidebar.selectbox("问答模型", choices, key=f"aux_chat_model_{service}")
        custom = (
            st.sidebar.text_input("问答模型 ID", key=f"aux_custom_chat_model_{service}")
            if choice == CUSTOM_MODEL else ""
        )
        try:
            model = clean_model_id(custom if choice == CUSTOM_MODEL else choice)
        except ValueError as exc:
            st.sidebar.warning(str(exc))
            model = None
        return key, model, base_url

    @staticmethod
    def handle_upload(file_types):
        return st.sidebar.file_uploader("选择文件上传", type=file_types)


def show_chinese_upload_style(file_types="PDF、TXT、CSV"):
    css = """<style>
/* Streamlit 内置上传组件的固定英文文案，仅做显示层汉化，不影响上传交互。 */
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] { display: none; }
[data-testid="stFileUploaderDropzone"] button::after {
  content: "选择文件"; font-size: 14px; line-height: 1.4;
}
[data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "单个文件最大 100 MB · 支持 PDF、TXT、CSV"; font-size: 12px;
}
</style>"""
    st.html(css.replace("支持 PDF、TXT、CSV", f"支持 {file_types}"))
