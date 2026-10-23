"""可选的 CSV/Excel 表格问答页。"""

import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from modules.utils import Utilities
from modules.model_config import safe_error

try:
    from modules.robby_sheet.table_tool import PandasAgent
except ImportError:
    PandasAgent = None

st.set_page_config(page_title="文档智能助手 · 表格分析", page_icon="📊", layout="wide")
st.title("📊 表格分析（可选）")
st.write("上传 CSV 或 Excel 表格，用中文提问，例如“数据有多少行？”“销售额最高的是哪一项？”")

if PandasAgent is None:
    st.info("此页面需要可选依赖。请在终端运行：pip install -r requirements-sheet.txt，然后重新启动应用。")
    st.stop()

key, model, base_url = Utilities.load_model_settings()
if not key or not model or not base_url:
    st.info("请先在左侧完成模型服务、API 密钥及模型配置。")
    st.stop()

uploaded_file = Utilities.handle_upload(["csv", "xlsx"])
if uploaded_file is None:
    st.info("请在左侧上传一个 CSV 或 Excel 文件。")
    st.stop()

try:
    content = uploaded_file.getvalue()
    file_id = hashlib.sha256(repr((base_url, hashlib.sha256(key.encode()).hexdigest(), model, uploaded_file.name)).encode("utf-8") + content).hexdigest()
    if st.session_state.get("sheet_file_id") != file_id:
        st.session_state["sheet_history"] = []
        st.session_state["sheet_file_id"] = file_id
    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(BytesIO(content))
    else:
        df = pd.read_csv(BytesIO(content))
except Exception as exc:
    st.error(f"表格读取失败：{exc}")
    st.stop()

st.caption(f"已上传：{uploaded_file.name} · {len(df)} 行 × {len(df.columns)} 列")
with st.expander("预览表格", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

if st.button("清空聊天记录"):
    st.session_state["sheet_history"] = []
for question, answer in st.session_state.get("sheet_history", []):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

question = st.chat_input("用中文询问表格内容…")
if question:
    try:
        with st.spinner("正在分析表格…"):
            result, _ = PandasAgent(model, base_url, api_key=key).get_agent_response(df, "请用中文回答：" + question)
        st.session_state["sheet_history"].append((question, str(result)))
        st.rerun()
    except Exception as exc:
        st.error(f"表格分析失败：{safe_error(exc, key)}")

st.caption("此功能使用 PandasAI，可能运行生成的分析代码。仅在可信的本机环境中使用。")
