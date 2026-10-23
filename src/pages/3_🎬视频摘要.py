"""有可访问字幕的 YouTube 视频中文摘要。"""

from urllib.parse import parse_qs, urlparse

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from modules.utils import Utilities
from modules.model_config import safe_error

st.set_page_config(page_title="文档智能助手 · 视频摘要", page_icon="🎬", layout="wide")
st.title("🎬 视频摘要")
st.write("粘贴有可访问字幕的 YouTube 视频链接，生成中文内容摘要。")
key, model, base_url = Utilities.load_model_settings()


def get_youtube_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/shorts/", "/live/")):
            return parsed.path.split("/")[2]
    if hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0] or None
    return None


if not key or not model or not base_url:
    st.info("请先在左侧完成模型服务、API 密钥及模型配置。")
else:
    video_url = st.text_input("YouTube 视频链接", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("生成中文摘要", type="primary"):
        video_id = get_youtube_id(video_url)
        if not video_id:
            st.error("链接不正确。请填写 YouTube 视频播放页或短链接。")
        else:
            try:
                with st.spinner("正在获取字幕并生成摘要…"):
                    transcript = YouTubeTranscriptApi().fetch(video_id)
                    transcript_text = " ".join(item.text for item in transcript)
                    if not transcript_text.strip():
                        raise ValueError("该视频没有可提取的字幕。")
                    prompt = PromptTemplate.from_template(
                        "请只依据下面的视频字幕用中文写摘要：包括主题、核心要点和结论。"
                        "不要编造字幕没有提及的信息。\n\n视频字幕：\n{text}\n\n中文摘要："
                    )
                    response = (prompt | ChatOpenAI(model=model, api_key=key, base_url=base_url)).invoke(
                        {"text": transcript_text[:15000]}
                    )
                st.subheader("摘要结果")
                st.write(response.content)
                with st.expander("查看原始字幕（最多展示 15000 字符）"):
                    st.text(transcript_text[:15000])
            except Exception as exc:
                st.error(f"处理失败：{safe_error(exc, key)}。请确认视频公开且有可访问的字幕。")
