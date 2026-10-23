"""Shared visual language for the existing Streamlit workspace.

Only authored labels/content are HTML; all dynamic fields are escaped.
"""

from html import escape

import streamlit as st


GLOBAL_STYLE = """<style>
:root { --rag-ink:#172137; --rag-muted:#56647b; --rag-line:#e4e9f2;
  --rag-blue:#345dd5; --rag-blue-soft:#eaf0ff; --rag-paper:#ffffff; }
.rag-page-title {margin:8px 0 28px;max-width:850px}
.rag-page-title small,.rag-kicker {display:inline-flex;align-items:center;gap:8px;color:#355bcc;
  text-transform:uppercase;letter-spacing:.15em;font:800 11px/1.5 sans-serif}
.rag-page-title small:before,.rag-kicker:before {content:'';height:7px;width:7px;border-radius:50%;background:#5479e9}
.rag-page-title h1 {margin:10px 0 8px;color:var(--rag-ink);font-size:clamp(28px,2.8vw,39px);
  line-height:1.2;letter-spacing:-.052em;font-weight:750}
.rag-page-title p {margin:0;color:var(--rag-muted);font-size:14px;line-height:1.75}
.rag-hero {position:relative;overflow:hidden;isolation:isolate;border:1px solid #dce4f3;border-radius:23px;
  background:linear-gradient(105deg,#fff 0%,#fff 62%,#eef3ff 100%);
  padding:37px clamp(26px,3.8vw,52px);margin:6px 0 23px;
  box-shadow:0 13px 32px rgba(22,42,87,.055)}
.rag-hero:after {content:'';position:absolute;z-index:-1;width:370px;height:370px;border:1px solid #d5e1fc;
  transform:rotate(24deg);border-radius:80px;right:-80px;top:-185px;
  box-shadow:0 0 0 44px rgba(227,236,255,.35),0 0 0 110px rgba(234,241,255,.3)}
.rag-eyebrow {display:block;color:#345ac8;font:800 11px/1.4 sans-serif;letter-spacing:.17em;margin-bottom:17px}
.rag-hero h1 {color:#1b2b52!important;font-size:clamp(30px,3vw,43px)!important;
  letter-spacing:-.055em;line-height:1.16;margin:0 0 12px;max-width:710px}
.rag-hero p {color:#50617b;font-size:14px;line-height:1.9;margin:0;max-width:620px}
.rag-hero .rag-hero-line {height:2px;width:46px;background:#486ee1;border-radius:6px;margin-bottom:19px}
.rag-stat {position:relative;min-height:146px;border:1px solid var(--rag-line);border-radius:18px;
  background:var(--rag-paper);box-shadow:0 8px 22px rgba(21,45,90,.038);padding:21px 23px;
  display:flex;flex-direction:column;justify-content:space-between;transition:box-shadow .2s,transform .2s}
.rag-stat:hover {transform:translateY(-2px);box-shadow:0 14px 28px rgba(21,45,90,.075)}
.rag-stat .top {display:flex;align-items:center;justify-content:space-between;gap:14px}
.rag-stat .label {color:#4f5e74;font:650 13px/1.3 sans-serif}
.rag-stat .icon {display:grid;place-items:center;height:35px;width:35px;border-radius:11px;
  background:#eef2fd;color:#4166cc;font-size:19px}
.rag-stat .icon,.rag-feature-icon {background-repeat:no-repeat!important;background-position:center!important;background-size:21px 21px!important}
.rag-icon-folder_open {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%234166cc%22%20stroke-width%3D%221.9%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M3%207a2%202%200%200%201%202-2h5l2%202h7a2%202%200%200%201%202%202v2%22%2F%3E%3Cpath%20d%3D%22M4%2011h17l-2%208H3l1-8Z%22%2F%3E%3C%2Fsvg%3E")!important}
.rag-icon-description {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%234166cc%22%20stroke-width%3D%221.9%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M6%203h9l4%204v14H6a2%202%200%200%201-2-2V5a2%202%200%200%201%202-2Z%22%2F%3E%3Cpath%20d%3D%22M14%203v5h5M8%2012h8M8%2016h8%22%2F%3E%3C%2Fsvg%3E")!important}
.rag-icon-forum {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%234166cc%22%20stroke-width%3D%221.9%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M4%205h14a2%202%200%200%201%202%202v9a2%202%200%200%201-2%202H9l-5%203V7a2%202%200%200%201%202-2Z%22%2F%3E%3Cpath%20d%3D%22M8%2010h8M8%2014h6%22%2F%3E%3C%2Fsvg%3E")!important}
.rag-icon-menu_book {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%234166cc%22%20stroke-width%3D%221.9%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M12%206c-2-2-5-3-9-2v15c4-1%207%200%209%202%202-2%205-3%209-2V4c-4-1-7%200-9%202Z%22%2F%3E%3Cpath%20d%3D%22M12%206v15%22%2F%3E%3C%2Fsvg%3E")!important}
.rag-icon-bar_chart {background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%234166cc%22%20stroke-width%3D%221.9%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M4%2020V13h4v7M10%2020V8h4v12M16%2020V4h4v16M3%2020h18%22%2F%3E%3C%2Fsvg%3E")!important}
.rag-stat strong {color:#172b56;font:760 38px/1 sans-serif;letter-spacing:-.07em}
.rag-stat .foot {color:#69778d;font:500 11px/1.4 sans-serif}
.rag-section-head {display:flex;align-items:end;justify-content:space-between;gap:14px;margin:19px 0 14px}
.rag-section-head h2 {font-size:20px!important;color:#1e2a42!important;letter-spacing:-.035em;margin:0}
.rag-section-head span {font-size:12px;color:#718098}
.rag-feature-intro {padding:3px 0 7px}
.rag-feature-icon {display:inline-grid;place-items:center;width:39px;height:39px;border-radius:12px;
  color:#3d61cc;background:#edf2ff;font-size:22px;margin-bottom:13px}
.rag-feature-intro h3 {color:#1d2b46;font-size:17px!important;letter-spacing:-.03em;margin:0 0 7px}
.rag-feature-intro p {color:#68768a;font-size:12px;line-height:1.7;margin:0 0 10px;min-height:42px}
.rag-kb-head {display:flex;align-items:center;gap:9px;color:#425679;font-size:11px;font-weight:750;letter-spacing:.1em}
.rag-kb-dot {height:9px;width:9px;border-radius:50%;background:#5c82e7}
.rag-kb-dot.paper {background:#b190e8}
.rag-quiet {border:1px dashed #cbd7ea;border-radius:16px;padding:25px 27px;
  color:#536580;background:#f9fbff;font-size:13px;line-height:1.7}
.rag-quiet strong {display:block;color:#22365b;font-size:16px;margin-bottom:5px}
.rag-auth-visual {min-height:455px;display:flex;flex-direction:column;justify-content:space-between;
  padding:38px;border-radius:23px;color:#f6f8ff;overflow:hidden;position:relative;
  background:#172b51;box-shadow:0 19px 42px rgba(25,44,86,.14)}
.rag-auth-visual:before {content:'';position:absolute;inset:0;opacity:.5;
  background:repeating-linear-gradient(0deg,transparent 0 46px,rgba(180,201,251,.09) 47px 48px),
             repeating-linear-gradient(90deg,transparent 0 46px,rgba(180,201,251,.09) 47px 48px);
  mask-image:linear-gradient(135deg,transparent 32%,#000 100%);pointer-events:none}
.rag-auth-visual:after {content:'01 / RESEARCH';position:absolute;right:37px;bottom:37px;
  color:#a8c3f2;font:750 11px sans-serif;letter-spacing:.14em}
.rag-auth-visual .brand {font:800 12px sans-serif;letter-spacing:.2em;color:#b6caf2}
.rag-auth-visual h1 {color:#fff!important;font-size:clamp(35px,3.1vw,51px)!important;
  line-height:1.18;letter-spacing:-.052em;margin:0 0 19px;position:relative}
.rag-auth-visual p {color:#d0ddf6;font-size:14px;line-height:2;max-width:410px;position:relative}
.rag-auth-visual .badge {align-self:flex-start;border-left:2px solid #94b0f7;padding:0 0 0 12px;
  font-size:12px;letter-spacing:.03em;color:#e2ebff;background:transparent;position:relative}
/* Only style named containers; native Streamlit widgets retain their semantics. */
.st-key-rag_login_form {background:#fff;border:1px solid #e1e7f2;border-radius:17px;
  box-shadow:0 14px 30px rgba(21,45,90,.045);padding:22px 25px 23px}
.st-key-rag_login_form [data-testid="stForm"] {border:0!important;padding:0!important}
.st-key-rag_login_form [data-testid="stTextInput"] {margin-top:3px}
[class*="st-key-rag_recent_"] {border-color:#e5eaf2!important;box-shadow:0 3px 13px rgba(22,43,83,.025)}
[class*="st-key-rag_session_card_"] {border-color:#e3eaf4!important;box-shadow:0 5px 15px rgba(22,43,83,.032)}
.st-key-rag_paper_controls,.st-key-rag_eval_controls {background:#fff;border-color:#e2e8f2!important;
  box-shadow:0 7px 22px rgba(21,45,90,.035)}

[class*="st-key-rag_feature_"] {border:1px solid var(--rag-line);border-radius:18px;
  background:#fff;padding:20px 21px 15px;box-shadow:0 8px 22px rgba(21,45,90,.037);
  transition:transform .2s,box-shadow .2s}
[class*="st-key-rag_feature_"]:hover {transform:translateY(-3px);box-shadow:0 17px 28px rgba(21,45,90,.085)}
[class*="st-key-rag_kb_card_"] {border-color:#e0e8f4!important;box-shadow:0 8px 20px rgba(21,45,90,.035)}
[class*="st-key-rag_kb_card_"]:hover {border-color:#b9caed!important;box-shadow:0 14px 25px rgba(21,45,90,.07)}
.st-key-rag_chat_feed {max-width:990px;margin:0 auto}
.st-key-rag_chat_feed [data-testid="stChatMessage"],
.st-key-rag_stream_reply [data-testid="stChatMessage"] {border:1px solid #e5eaf4;border-radius:18px;
  background:#fff;box-shadow:0 5px 16px rgba(25,45,90,.032);margin-bottom:15px;padding:18px 19px}
.st-key-rag_chat_feed [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
.st-key-rag_stream_reply [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  background:#edf2ff;border-color:#dde7ff;max-width:85%;margin-left:auto}
.st-key-rag_chat_feed [data-testid="stChatMessage"] p,
.st-key-rag_stream_reply [data-testid="stChatMessage"] p {line-height:1.82}
.st-key-rag_chat_feed [data-testid="stChatMessage"] pre,
.st-key-rag_stream_reply [data-testid="stChatMessage"] pre {border-radius:10px}
@media(max-width:750px) {
  .rag-hero {padding:26px 23px}.rag-auth-visual {min-height:260px;padding:27px}
  .rag-auth-visual:after {display:none}.rag-stat {min-height:120px}
  .rag-feature-intro p {min-height:0}.st-key-rag_login_form {padding:18px}
}
</style>"""


# Use a fixed class-name allowlist for authored CSS icons.
_ICON_NAMES = frozenset(("folder_open", "description", "forum", "menu_book", "bar_chart"))


def _icon(name: str) -> str:
    return "rag-icon-" + (name if name in _ICON_NAMES else "bar_chart")


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.html(f"<section class='rag-hero'><span class='rag-eyebrow'>{escape(kicker)}</span>"
            f"<div class='rag-hero-line'></div><h1>{escape(title)}</h1>"
            f"<p>{escape(subtitle)}</p></section>")


def page_title(kicker: str, title: str, subtitle: str) -> None:
    st.html(f"<header class='rag-page-title'><small>{escape(kicker)}</small>"
            f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>")


def metric(label: str, value: int, foot: str, icon: str = "bar_chart") -> None:
    st.html(f"<div class='rag-stat'><div class='top'><span class='label'>{escape(label)}</span>"
            f"<span class='icon {_icon(icon)}'></span></div>"
            f"<strong>{int(value)}</strong><span class='foot'>{escape(foot)}</span></div>")


def section_heading(title: str, detail: str = "") -> None:
    st.html(f"<div class='rag-section-head'><h2>{escape(title)}</h2>"
            f"<span>{escape(detail)}</span></div>")


def feature_intro(title: str, detail: str, icon: str) -> None:
    st.html(f"<div class='rag-feature-intro'><span class='rag-feature-icon {_icon(icon)}'></span>"
            f"<h3>{escape(title)}</h3><p>{escape(detail)}</p></div>")


def empty(title: str, detail: str) -> None:
    st.html(f"<div class='rag-quiet'><strong>{escape(title)}</strong>{escape(detail)}</div>")