"""Shared Streamlit API-key input with explicit encrypted save/delete actions."""

import hashlib
import os

import streamlit as st

from .credentials import CredentialStore, CredentialStoreError
from .model_config import DEFAULT_SERVICE


def _save_key(scope: str, widget_key: str, notice_key: str) -> None:
    try:
        CredentialStore().save(scope, st.session_state.get(widget_key, "").strip())
        st.session_state[widget_key] = ""  # Clear the browser field after encrypted save.
        st.session_state[notice_key] = "密钥已加密保存在运行本应用的 Windows 用户账户下。"
    except CredentialStoreError as exc:
        st.session_state[notice_key] = str(exc)


def _delete_key(scope: str, widget_key: str, notice_key: str) -> None:
    try:
        CredentialStore().delete(scope)
        st.session_state[widget_key] = ""  # Button callbacks run before this widget is rendered.
        st.session_state[notice_key] = "已删除本机保存的密钥。"
    except CredentialStoreError as exc:
        st.session_state[notice_key] = str(exc)


def render_api_key(service: str, base_url: str | None, *, prefix: str, scope_prefix: str = "", area=None) -> str:
    """A typed key stays in the session; only a button click persists it."""
    if area is None:
        area = st.sidebar
    scope = f"{scope_prefix}|{service}|{base_url or ''}" if scope_prefix else f"{service}|{base_url or ''}"
    scope_id = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
    widget_key = f"{prefix}_api_key_{scope_id}"
    notice_key = f"{prefix}_key_notice_{scope_id}"
    store = CredentialStore()
    try:
        saved_key = store.load(scope) if base_url else None
        has_saved = store.has(scope) if base_url else False
    except CredentialStoreError as exc:
        saved_key = None
        has_saved = False
        area.warning(str(exc))
    default_key = "" if scope_prefix else (os.getenv("RAG_DEFAULT_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
    # Keep decrypted saved credentials on the server; never send them back as a widget value.
    typed_key = area.text_input(
        "API 密钥", value="", key=widget_key, type="password",
        placeholder="已从本机加载，可留空" if saved_key else "填写当前服务的密钥",
        help="留空时使用本机已保存的密钥；输入新密钥只在当前会话使用，点击保存才会加密写入本机。",
    ).strip()
    key = typed_key or saved_key or (default_key if service == DEFAULT_SERVICE else "")
    if typed_key:
        area.caption("当前输入仅在此会话使用；点击保存后才会覆盖本机密钥。")
    elif saved_key:
        area.caption("已从本机加密存储加载密钥；输入框保持空白，不会把密钥回填到浏览器。")
    elif key:
        area.caption("已从本机环境配置加载密钥。")
    else:
        area.caption("尚未填写密钥。")
    save_button, delete_button = area.columns(2)
    save_button.button(
        "保存到本机", key=f"{prefix}_save_{scope_id}",
        disabled=not typed_key or not base_url or os.name != "nt",
        on_click=_save_key, args=(scope, widget_key, notice_key),
    )
    delete_button.button(
        "删除已保存", key=f"{prefix}_delete_{scope_id}", disabled=not has_saved,
        on_click=_delete_key, args=(scope, widget_key, notice_key),
    )
    area.caption("仅点击保存后才会在运行应用的电脑加密落盘；调用模型时才发送给所选服务。")
    if os.name != "nt":
        area.caption("当前系统不支持本机加密保存，请使用本机 .env。")
    if notice_key in st.session_state:
        area.info(st.session_state.pop(notice_key))
    return key
