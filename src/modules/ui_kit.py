"""Small visual primitives; all user-provided text is escaped before HTML rendering."""

from html import escape

import streamlit as st


GLOBAL_STYLE = """<style>
/* Minimal, scoped product polish; native Streamlit remains responsible for widgets. */
.rag-hero { position:relative; overflow:hidden; border:1px solid #263e7f;
  border-radius:24px; padding:34px 38px; margin:2px 0 18px;
  background:linear-gradient(115deg,#121f42 0%,#243f7e 60%,#3e60ab 100%);
  box-shadow:0 16px 38px rgba(24,45,95,.16); color:#f7f9ff; }
.rag-hero:after { content:''; position:absolute; width:290px; height:290px; border-radius:50%;
  right:-68px; top:-110px; border:56px solid rgba(255,255,255,.06); pointer-events:none; }
.rag-eyebrow { display:block; text-transform:uppercase; color:#a6bfff;
  font:700 11px/1.4 sans-serif; letter-spacing:.18em; margin-bottom:13px; }
.rag-hero h1 { color:#fff!important; font-size:clamp(26px,3vw,40px)!important;
  letter-spacing:-.035em; line-height:1.25; margin:0 0 10px; }
.rag-hero p { color:#d9e3fa; font-size:14px; line-height:1.7; margin:0; max-width:620px; }
.rag-page-title { margin:4px 0 22px; }
.rag-page-title small { display:block; color:#4868af; letter-spacing:.16em; font:700 11px sans-serif; margin-bottom:10px; }
.rag-page-title h1 { margin:0 0 9px; color:#19274b; font-size:clamp(28px,3vw,38px); letter-spacing:-.04em; }
.rag-page-title p { margin:0; color:#5b6881; font-size:14px; line-height:1.7; }
.rag-stat { min-height:128px; border:1px solid #e2e8f4; border-radius:19px;
  background:#fff; box-shadow:0 8px 26px rgba(37,57,102,.045);
  padding:20px 24px; display:flex; flex-direction:column; justify-content:space-between; }
.rag-stat .label { color:#66738c; font:600 13px sans-serif; }
.rag-stat strong { color:#1c2e5a; font:700 36px/1.1 sans-serif; letter-spacing:-.045em; }
.rag-stat .foot { color:#8592aa; font:500 11px sans-serif; }
.rag-quiet { border:1px dashed #c9d5e8; border-radius:17px; padding:26px 28px;
  color:#556787; background:#f8faff; font-size:14px; }
.rag-quiet strong { display:block; color:#223967; font-size:16px; margin-bottom:6px; }
.rag-auth-visual { min-height:450px; display:flex; flex-direction:column; justify-content:space-between;
  padding:40px; border-radius:26px; color:#fff; overflow:hidden; position:relative;
  background:radial-gradient(circle at 90% 12%,#5476c4 0%,transparent 40%),
             linear-gradient(145deg,#132348,#253e83);
  box-shadow:0 20px 44px rgba(27,47,101,.19); }
.rag-auth-visual:before{content:'';position:absolute;width:350px;height:350px;
  right:-126px;bottom:-175px;border:65px solid rgba(255,255,255,.07);border-radius:50%}
.rag-auth-visual .brand {font:700 12px sans-serif;letter-spacing:.16em;color:#bcd0ff}
.rag-auth-visual h1 {color:#fff!important;font-size:clamp(32px,3vw,48px)!important;line-height:1.19;
  letter-spacing:-.045em;margin:0 0 18px}
.rag-auth-visual p{color:#d7e3ff;font-size:15px;line-height:1.9;max-width:360px}
.rag-auth-visual .badge{align-self:flex-start;border:1px solid rgba(255,255,255,.25);
  padding:9px 14px;border-radius:30px;font-size:12px;color:#e2ebff;background:rgba(255,255,255,.08)}
@media(max-width:700px){.rag-auth-visual{min-height:230px;padding:26px} .rag-hero{padding:25px 23px} .rag-stat{min-height:108px} }
</style>"""


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.html(f"<section class='rag-hero'><span class='rag-eyebrow'>{escape(kicker)}</span>"
            f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></section>")


def page_title(kicker: str, title: str, subtitle: str) -> None:
    st.html(f"<header class='rag-page-title'><small>{escape(kicker)}</small>"
            f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>")


def metric(label: str, value: int, foot: str) -> None:
    st.html(f"<div class='rag-stat'><span class='label'>{escape(label)}</span>"
            f"<strong>{int(value)}</strong><span class='foot'>{escape(foot)}</span></div>")


def empty(title: str, detail: str) -> None:
    st.html(f"<div class='rag-quiet'><strong>{escape(title)}</strong>{escape(detail)}</div>")
