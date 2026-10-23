"""ASGI entry point: exchange short-lived login tickets for HttpOnly cookies."""

import os
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

from modules.repository import LOGIN_COOKIE_NAME, Repository


async def exchange_login(request):
    origin = request.headers.get("origin", "")
    public_origin = os.getenv("RAG_PUBLIC_ORIGIN", "").strip().rstrip("/")
    if public_origin:
        parsed = urlsplit(public_origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query
                or parsed.fragment or parsed.username or parsed.password):
            return JSONResponse({"ok": False}, status_code=503, headers={"Cache-Control": "no-store"})
    expected = public_origin or f"{request.url.scheme}://{request.headers.get('host', '')}"
    if origin != expected or request.headers.get("content-type", "").split(";")[0] != "application/json":
        return JSONResponse({"ok": False}, status_code=403, headers={"Cache-Control": "no-store"})
    body = await request.body()
    if len(body) > 1024:
        return JSONResponse({"ok": False}, status_code=400, headers={"Cache-Control": "no-store"})
    try:
        import json
        ticket = json.loads(body).get("ticket")
    except (ValueError, AttributeError):
        ticket = None
    consumed = Repository().consume_login_ticket(ticket)
    if not consumed:
        return JSONResponse({"ok": False}, status_code=401, headers={"Cache-Control": "no-store"})
    token, lifetime = consumed
    response = JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})
    response.set_cookie(LOGIN_COOKIE_NAME, token, max_age=lifetime, httponly=True,
                        samesite="strict", secure=bool(public_origin) or request.url.scheme == "https", path="/")
    return response


app = st.App(Path(__file__).with_name("Home.py"), routes=[Route("/api/auth/exchange", exchange_login, methods=["POST"])])