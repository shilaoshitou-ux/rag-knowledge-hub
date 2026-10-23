"""Local-account login with optional, server-validated browser persistence."""

import json
import os

import streamlit as st

from .repository import LOGIN_COOKIE_NAME, Repository


def remember_days() -> int:
    try:
        return min(30, max(1, int(os.getenv("RAG_LOGIN_TTL_DAYS", "7"))))
    except ValueError:
        return 7


def restore_login(repo: Repository) -> None:
    """A signed-in cookie is a random server-side token, never a user ID or password."""
    token = st.context.cookies.get(LOGIN_COOKIE_NAME)
    current = st.session_state.get("rag_current_user")
    if token:
        user = repo.user_from_login_token(token)
        if user:
            if not current or st.session_state.get("rag_cookie_login"):
                st.session_state["rag_current_user"] = user["id"]
                st.session_state["rag_cookie_login"] = True
            return
    if st.session_state.get("rag_cookie_login"):
        st.session_state.pop("rag_current_user", None)
        st.session_state.pop("rag_cookie_login", None)
        st.session_state.pop("rag_managed_session", None)


def _exchange_html(ticket: str) -> str:
    # Only a freshly generated, single-use ticket (90s TTL) reaches the DOM.
    # The persistent bearer token is created by the server and set as HttpOnly.
    return f"""<div id="rag-login-progress" role="status">正在安全地保存这台浏览器的登录状态……</div>
    <script>
    fetch('/api/auth/exchange', {{
      method: 'POST', credentials: 'same-origin',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ticket: {json.dumps(ticket)}}})
    }}).then(response => {{
      if (!response.ok) throw new Error('exchange failed');
      window.location.reload();
    }}).catch(() => {{
      document.getElementById('rag-login-progress').textContent =
        '无法保存登录状态；请使用下方按钮继续本次登录，并检查服务是否从新版入口启动。';
    }});
    </script>"""


def require_user(repo: Repository) -> dict:
    restore_login(repo)
    current = st.session_state.get("rag_current_user")
    if current:
        user = repo.get_user(current)
        if user:
            with st.sidebar:
                st.caption(f"账号 · {user['username']}")
                if st.button("退出登录", key="rag_sign_out", icon=":material/logout:"):
                    repo.revoke_login_token(st.context.cookies.get(LOGIN_COOKIE_NAME))
                    st.session_state.pop("rag_current_user", None)
                    st.session_state.pop("rag_cookie_login", None)
                    st.session_state.pop("rag_managed_session", None)
                    st.rerun()
            return user
        st.session_state.pop("rag_current_user", None)
    st.space("small")
    visual, form_column = st.columns([1.12, 1], gap="large", vertical_alignment="center")
    with visual:
        st.html("""<section class='rag-auth-visual'>
          <span class='brand'>ATLAS / KNOWLEDGE STUDIO</span>
          <div><h1>让每一个问题，<br>都有据可循。</h1>
          <p>统一管理企业资料与学术论文。<br>从可信证据出发，让知识真正参与工作。</p></div>
          <span class='badge'>知识库 · 智能检索 · 论文研读</span></section>""")
    with form_column:
        st.caption("WELCOME / 欢迎回来")
        st.title("登录工作区", anchor=False)
        st.write("继续你的知识整理与探索。")
        st.space("small")
        with st.container(key="rag_login_form"):
            mode = st.segmented_control("账号操作", ["登录", "注册"], default="登录", key="rag_auth_mode")
            with st.form("rag_auth_form", clear_on_submit=True, border=False):
                username = st.text_input("用户名", key="rag_auth_name", placeholder="输入用户名（支持中文）")
                password = st.text_input("密码", type="password", key="rag_auth_password", placeholder="至少 10 位")
                remember = st.checkbox(f"在这台浏览器保持登录 {remember_days()} 天", value=True, key="rag_remember") if mode == "登录" else False
                submitted = st.form_submit_button("登录" if mode == "登录" else "创建账号", type="primary", width="stretch")
        st.caption("资料与会话保存在本机，按账号隔离。共用电脑可取消保持登录；不要将服务直接开放至公网。")
    if submitted:
        try:
            if mode == "注册":
                repo.register(username, password)
                st.success("账号创建成功，请登录。")
            else:
                user = repo.authenticate(username, password)
                if user:
                    # Revoke a previous cookie (possibly from another account in this browser).
                    repo.revoke_login_token(st.context.cookies.get(LOGIN_COOKIE_NAME))
                    st.session_state["rag_current_user"] = user["id"]
                    st.session_state.pop("rag_cookie_login", None)
                    if remember:
                        ticket = repo.create_login_ticket(user["id"], lifetime_seconds=remember_days() * 86400)
                        st.html(_exchange_html(ticket), unsafe_allow_javascript=True)
                        if st.button("只在本次页面登录（无法自动跳转时点击）"):
                            st.rerun()
                        st.stop()
                    st.rerun()
                st.error("用户名或密码不正确。")
        except ValueError as exc:
            st.error(str(exc))
    st.stop()