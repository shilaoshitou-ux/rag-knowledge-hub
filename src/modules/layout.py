"""其他页面共用的中文提示。"""

import streamlit as st


class Layout:
    def show_header(self, types_files):
        st.title(f"使用文档智能助手分析 {types_files}")

    def show_api_key_missing(self):
        st.info("请先在左侧输入 OpenAI API 密钥，然后再继续操作。")

    def prompt_form(self):
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area("你的问题", placeholder="输入想了解的内容…")
            submit_button = st.form_submit_button(label="发送")
        return bool(submit_button and user_input.strip()), user_input
