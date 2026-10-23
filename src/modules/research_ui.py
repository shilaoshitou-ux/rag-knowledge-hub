"""A small, scoped visual identity shared by the research workspace and home."""

from html import escape

import streamlit as st


_HERO_STYLE = """<style>
.research-hero {
  background: linear-gradient(120deg, #1B2758 0%, #344D9C 68%, #5268AE 100%);
  border: 1px solid #677CC1; border-radius: 18px; padding: 28px 34px;
  color: #f8f9f3; box-shadow: 0 12px 32px rgba(21, 47, 49, .13);
}
.research-hero .eyebrow { color: #C8D7FF; font: 600 11px/1.5 sans-serif; letter-spacing: .19em; }
.research-hero h1 { color: #fff; margin: 10px 0 9px; font: 650 clamp(26px, 2.5vw, 38px)/1.25 sans-serif; }
.research-hero p { color: #E1E9FF; margin: 0; line-height: 1.7; font-size: 14px; }
@media (max-width: 700px) { .research-hero { padding: 22px; } }
</style>"""


def render_research_hero(kicker: str, title: str, subtitle: str) -> None:
    """Render only escaped text; no document or model content enters HTML."""
    st.html(
        f"{_HERO_STYLE}<div class='research-hero' role='banner'>"
        f"<div class='eyebrow'>{escape(kicker)}</div>"
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>"
    )