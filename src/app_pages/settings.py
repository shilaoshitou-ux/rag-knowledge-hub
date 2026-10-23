"""Connection settings and transparent local-storage status."""

import streamlit as st

from modules.ui_kit import page_title
from modules.auth_ui import require_user
from modules.managed_ui import render_model_settings
from modules.repository import Repository

repo = Repository()
user = require_user(repo)
page_title("ACCOUNT / SETTINGS", "系统设置", f"当前账号：{user['username']}。在这里了解本机数据与模型连接方式。")
render_model_settings(user["id"])
with st.container(border=True, key="rag_settings_guide"):
    st.markdown("**模型连接**")
    st.write("请从左侧展开“模型连接与密钥”，选择服务地址、问答模型和向量模型。")
    st.caption("密钥保存后不会再次明文显示；刷新后可继续使用该账号已保存的密钥。")
with st.container(border=True):
    st.subheader("本机数据")
    st.code(str(repo.path), language=None)
    st.write("业务元数据与会话保存在 SQLite；向量索引保存在 Chroma。旧版索引不迁移、不删除。")
    st.warning("这是本机演示部署；目前不是可直接暴露到公网的生产身份认证服务。")
with st.expander("账号安全 · 修改密码", icon=":material/lock:"):
    with st.form("change_password", clear_on_submit=True):
        old_password = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码（至少 10 位）", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        change = st.form_submit_button("更新密码")
    if change:
        try:
            if new_password != confirm_password:
                raise ValueError("两次输入的新密码不一致。")
            repo.change_password(user["id"], old_password, new_password)
            st.success("密码已更新。下次登录请使用新密码。")
        except ValueError as exc:
            st.error(str(exc))
