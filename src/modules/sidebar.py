"""其他页面共用的中文侧栏组件。"""

import streamlit as st

from modules.model_config import CHAT_MODELS


class Sidebar:
    MODEL_OPTIONS = CHAT_MODELS
    TEMPERATURE_DEFAULT_VALUE = 0.0

    @staticmethod
    def about():
        with st.sidebar.expander("关于本项目"):
            st.write("基于开源项目 Robby-chatbot 开发的中文文档智能助手。")
            st.markdown("[查看原项目](https://github.com/yvann-ba/Robby-chatbot)")

    @staticmethod
    def reset_chat_button():
        if st.button("清空聊天记录"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        st.session_state["model"] = st.selectbox("对话模型", self.MODEL_OPTIONS)

    def temperature_slider(self):
        st.session_state["temperature"] = st.slider("回答灵活度", 0.0, 1.0, self.TEMPERATURE_DEFAULT_VALUE, 0.1)

    def show_options(self):
        with st.sidebar.expander("问答设置"):
            self.reset_chat_button()
            self.model_selector()
            self.temperature_slider()
