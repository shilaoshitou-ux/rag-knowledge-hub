"""Managed knowledge bases: card overview and focused document workspace."""

import streamlit as st

from modules.auth_ui import require_user
from modules.chunk_edits import MAX_EDIT_CHARS, REVISION_SUFFIX, delete_chunk, update_chunk
from modules.embedder import UploadedDocument
from modules.managed_index import delete_base, delete_document, interrupted_count, queue_upload, recover_interrupted, retry_document
from modules.managed_ui import render_model_settings
from modules.repository import Repository
from modules.vector_cleanup import audit_vectors, repair_missing, retry_cleanup
from modules.ui_kit import empty, page_title, section_heading

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("ATLAS / 知识空间", "我的知识库", "分门别类整理资料，查看处理进度，让每个答案都能返回原始证据。")
bases = repo.list_bases(user["id"])
with st.expander("创建新知识库", expanded=not bases, icon=":material/add_circle:"):
    st.caption("向量模型在创建时绑定到知识库，避免不同维度的索引混用。")
    with st.form("new_base", border=False):
        name = st.text_input("知识库名称", placeholder="例如：研发技术规范 / 论文精读")
        kind = st.selectbox("知识库类型", ["企业知识库", "学术论文库"])
        description = st.text_area("描述（可选）", height=75)
        left, right = st.columns(2)
        chunk_size = left.number_input("切分长度", min_value=200, max_value=4000, value=1000, step=100)
        overlap = right.number_input("重叠长度", min_value=0, max_value=1000, value=150, step=50)
        submitted = st.form_submit_button("创建知识库", type="primary")
    if submitted:
        try:
            if not settings.base_url or not settings.embedding_model:
                raise ValueError("请先在侧栏配置模型服务及向量模型。")
            created_id = repo.create_base(user["id"], name, "paper" if kind == "学术论文库" else "enterprise",
                                          description, settings.embedding_model, settings.embedding_provider,
                                          settings.base_url, int(chunk_size), int(overlap))
            st.session_state["rag_managed_kb"] = created_id
            st.session_state["rag_just_created_kb"] = created_id
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

@st.dialog("永久删除知识库")
def confirm_delete_base(base: dict) -> None:
    st.warning(f"即将永久删除「{base['name']}」的文档、向量索引和历史会话；其他知识库不受影响。此操作不能撤销。")
    st.caption(f"当前包含 {base['document_count']} 份可检索文档。请确认这是要删除的知识库。")
    if any(doc["status"] in {"waiting", "processing"} for doc in repo.list_documents(user["id"], base["id"])):
        st.info("如有文档正在处理，请等待完成后再删除。")
    if any(repo.pending_turn(user["id"], session["id"])
           for session in repo.list_sessions(user["id"], base["id"])):
        st.info("此库有未完成的问答，请先在聊天页继续或放弃该任务。")
    with st.form(f"confirm_delete_base_{base['id']}"):
        typed_name = st.text_input("输入知识库完整名称确认", placeholder=base["name"])
        confirmed = st.form_submit_button("确认永久删除", type="primary")
    if not confirmed:
        return
    if typed_name != base["name"]:
        st.error("名称不一致，未删除。")
        return
    try:
        current = repo.get_base(user["id"], base["id"])
        if not current or current["name"] != base["name"]:
            raise ValueError("知识库名称已变更，请关闭窗口后重新确认。")
        delete_base(repo, user["id"], current, settings.api_key)
    except (ValueError, PermissionError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"删除未完成（{type(exc).__name__}），知识库仍保留。请检查本地索引后重试。")
    else:
        st.session_state.pop("rag_managed_kb", None)
        st.rerun()


active_id = st.session_state.get("rag_managed_kb")
item = next((kb for kb in bases if kb["id"] == active_id), None)
if item:
    st.space("small")
    page_title("COLLECTION / 管理资料", item["name"], "在这里上传、更新或删除资料；正在处理的文档会显示实时状态。")
    with st.expander("索引健康检查（仅当前知识库）", icon=":material/health_and_safety:"):
        queued = repo.count_vector_cleanup(user["id"], item["id"])
        st.caption(f"待重试清理任务：{queued} 条。检测只比较当前知识库 Chroma 向量 ID 与 SQLite 有效片段 ID，不读取其他账号索引。")
        ready = item["embedding_provider"] != "company" or bool(settings.api_key and settings.base_url == item["base_url"])
        if not ready:
            st.info("请先配置该知识库所属服务的密钥，再检查或清理向量索引。")
        if st.button("扫描并登记多余向量", key=f"audit_vectors_{item['id']}", disabled=not ready):
            try:
                result = audit_vectors(repo, user["id"], item, settings.api_key)
                st.info(f"当前有效片段 {result['active_chunks']}、向量 {result['vectors']}；已登记多余向量 {result['stale_queued']}，疑似缺失有效向量 {result['missing_active']}。可手动点击下方修复缺失向量；不会凭空生成文档内容。")
            except Exception as exc:
                st.error(f"索引对账失败（{type(exc).__name__}）。请检查知识库状态及模型连接。")
        st.caption("修复缺失片段会重新调用当前库绑定的 Embedding 模型（可能产生费用）；每次最多处理 20 条。")
        if st.button("修复缺失的有效向量", key=f"repair_vectors_{item['id']}", disabled=not ready):
            try:
                result = repair_missing(repo, user["id"], item, settings.api_key)
                st.success(f"发现缺失 {result['missing']}，已重新向量化 {result['repaired']}；剩余约 {result['remaining_estimate']}。")
            except Exception as exc:
                st.error(f"修复中断（{type(exc).__name__}）；已修复的片段保留，可再次点击继续。")
        if st.button("重试清理已登记的向量", key=f"retry_vectors_{item['id']}",
                     disabled=not ready or not queued):
            try:
                result = retry_cleanup(repo, user["id"], item, settings.api_key)
                st.success(f"成功清理 {result['deleted']}，失败待重试 {result['failed']}，仍有效未删除 {result['active_skipped']}。")
            except Exception as exc:
                st.error(f"清理失败（{type(exc).__name__}），任务仍保留，可稍后重试。")
    same_endpoint = settings.base_url == item["base_url"]
    active_docs = repo.list_documents(user["id"], item["id"])
    ready = sum(doc["status"] == "success" for doc in active_docs)
    in_progress = sum(doc["status"] in {"waiting", "processing"} for doc in active_docs)
    with st.container(border=True, key=f"rag_kb_flow_{item['id']}"):
        st.caption("从资料到答案 · 当前知识库")
        steps = (("01", "选择资料", bool(active_docs)),
                 ("02", "处理完成", ready > 0),
                 ("03", "开始问答", ready > 0))
        for column, (number, label, done) in zip(st.columns(3, gap="small"), steps):
            with column:
                st.badge(("可开始" if number == "03" else "已就绪") if done else "待完成",
                         color="blue" if number == "03" and done else ("green" if done else "gray"))
                st.markdown(f"**{number} · {label}**")
        if in_progress:
            st.info(f"{in_progress} 份文档正在等待或处理中。下方状态会自动更新，无需重复提交。")
        elif ready:
            st.caption(f"已有 {ready} 份可检索文档。你可以继续上传，也可以开始问答。")
        else:
            st.caption("先上传一份 PDF、Word 或文本文件，处理成功后即可提问。")
    if ready:
        st.page_link("app_pages/chat.py", label="用此知识库开始问答", icon=":material/arrow_forward:", width="content")
    documents_tab, settings_tab, danger_tab = st.tabs(["资料与处理状态", "知识库设置", "删除知识库"])
    with documents_tab:
        with st.container(border=True):
            st.markdown("#### 上传资料")
            st.caption("支持 PDF、DOCX、TXT、Markdown、CSV；单文件最大 100 MB。扫描版 PDF 需先 OCR。")
            if st.session_state.pop("rag_just_created_kb", None) == item["id"]:
                st.success("知识库已创建。现在点击下方“选择文件”，再点击“上传并处理”。")
            # A form sends the selected file and submit event together. With a separate
            # button, some browser reruns discarded the uploader value before submission.
            with st.form(f"kb_upload_form_{item['id']}", clear_on_submit=False, border=False):
                uploads = st.file_uploader("选择文件", type=["pdf", "docx", "txt", "md", "markdown", "csv"],
                                           accept_multiple_files=True, key=f"kb_upload_{item['id']}")
                st.caption("等上传框显示文件名和大小后，再点“上传并处理”；仅选中文件不会调用模型。")
                submitted_upload = st.form_submit_button("上传并处理", type="primary", key=f"enqueue_{item['id']}")
            if not same_endpoint:
                st.warning("当前模型服务与创建此库时不同；请在左侧切换到同一接口。")
            if item["embedding_provider"] == "company" and not settings.api_key:
                st.warning("在线向量化尚未检测到当前服务的 API 密钥，请展开左侧“模型连接与密钥”填写或核对已保存的密钥。")
            if submitted_upload:
                if not uploads:
                    st.error("服务端没有收到文件。请重新选择 PDF，等待上传框显示文件名后再提交。")
                elif not same_endpoint:
                    st.error("当前模型服务与创建此库时不同，请先切换到相同接口。")
                else:
                    st.caption(f"服务端已收到 {len(uploads)} 个文件，提交结果如下：")
                    for upload in uploads:
                        try:
                            doc, created = queue_upload(repo, user["id"], item,
                                                        UploadedDocument(upload.name, upload.getvalue()), settings.api_key)
                            if created:
                                st.success(f"{doc['name']} 已进入处理队列；请查看下方“已上传文档”的实时状态。")
                            else:
                                st.info(f"{doc['name']} 内容未变，已跳过重复任务。")
                        except (ValueError, PermissionError) as exc:
                            st.error(f"{upload.name}：{exc}")
                # Keep feedback visible. The document status fragment refreshes every 3 seconds.
        @st.fragment(run_every="3s")
        def document_status_panel():
            st.subheader("已上传文档", anchor=False)
            orphaned = interrupted_count(repo, user["id"], item["id"])
            if orphaned:
                if st.button("恢复中断的处理任务", key=f"resume_{item['id']}"):
                    try:
                        count = recover_interrupted(repo, user["id"], item, settings.api_key)
                        st.info(f"已重新提交 {count} 个任务。")
                    except (ValueError, PermissionError) as exc:
                        st.error(str(exc))
                st.caption("服务重启后若任务长期没有变化，可手动恢复；正在运行的任务不会重复提交。")
            docs = repo.list_documents(user["id"], item["id"])
            status_label = {"waiting": "等待处理", "processing": "处理中", "success": "处理成功",
                            "failed": "处理失败", "deleted": "已删除"}
            if not docs:
                empty("还没有文档", "从上方上传文件，处理成功后即可在当前知识库中问答。")
            if any(doc["status"] == "success" for doc in docs):
                st.page_link("app_pages/chat.py", label="处理完成，去问答并核对引用",
                             icon=":material/arrow_forward:")
            for doc in docs:
                with st.container(border=True):
                    title, action = st.columns([4, 1], vertical_alignment="center")
                    title.markdown(f"**{doc['name']}** · v{doc['version']}")
                    action.badge(status_label.get(doc["status"], doc["status"]),
                                 color={"waiting": "gray", "processing": "blue",
                                        "success": "green", "failed": "red"}.get(doc["status"], "gray"))
                    st.caption(f"{doc['chunk_count']} 个片段 · {doc['bytes']/1024:.1f} KB · {doc['updated_at'][:16].replace('T',' ')}")
                    if doc["error"]:
                        st.warning(doc["error"])
                    if doc["status"] == "failed" and st.button("重试处理", key=f"retry_{doc['id']}"):
                        try:
                            retry_document(repo, user["id"], item["id"], doc["id"], settings.api_key)
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                    if doc["status"] not in {"waiting", "processing"} and st.button("删除文档", key=f"remove_{doc['id']}"):
                        try:
                            delete_document(repo, user["id"], item, doc["id"], settings.api_key)
                            st.rerun()
                        except (ValueError, PermissionError) as exc:
                            st.error(str(exc))
        document_status_panel()
        with st.container(border=True, key=f"rag_chunk_editor_{item['id']}"):
            st.markdown("#### 编辑知识片段")
            st.caption("选中片段可直接修改或删除，无需重新上传整份文件；只重新向量化修改的片段。"
                       "原始 PDF/Word/TXT 文件不会改动，修改后的来源会标记为「知识库修订」。"
                       "若相邻片段有重叠文字，请一并检查。")
            notice = st.session_state.pop("rag_chunk_notice", None)
            if notice and notice[:2] == (user["id"], item["id"]):
                st.warning(notice[2]) if notice[3] else st.success(notice[2])
            editable = [doc for doc in repo.list_documents(user["id"], item["id"])
                        if doc["status"] == "success" and doc["chunk_count"] > 0]
            if not editable:
                st.info("目前没有可编辑的片段；请先等待文档处理成功。")
            else:
                by_doc = {doc["id"]: doc for doc in editable}
                selected_doc_id = st.selectbox("选择文档", list(by_doc),
                    format_func=lambda did: f"{by_doc[did]['name']} · {by_doc[did]['chunk_count']} 个片段",
                    key=f"rag_chunk_doc_{item['id']}")
                chunks = [row for row in repo.list_chunks(user["id"], item["id"])
                          if row["document_id"] == selected_doc_id]
                if chunks:
                    by_chunk = {row["id"]: row for row in chunks}
                    selected_chunk_id = st.selectbox("选择片段", list(by_chunk),
                        format_func=lambda cid: f"{by_chunk[cid]['location']} · {by_chunk[cid]['content'][:65]}",
                        key=f"rag_chunk_choice_{selected_doc_id}")
                    selected_chunk = by_chunk[selected_chunk_id]
                    st.caption(f"片段 ID：{selected_chunk_id[:16]}… · 来源：{selected_chunk['location']}")
                    if selected_chunk["location"].endswith(REVISION_SUFFIX):
                        st.info("这是知识库修订内容，不代表上传文件中的原文已改变。")
                    with st.form(f"rag_chunk_edit_form_{selected_chunk_id}"):
                        revised = st.text_area("修改此片段内容", value=selected_chunk["content"],
                            height=220, key=f"rag_chunk_text_{selected_chunk_id}")
                        st.caption(f"最多 {MAX_EDIT_CHARS} 字符。更新后片段 ID 会改变，其他片段无需重新上传或向量化。")
                        save_chunk = st.form_submit_button("保存片段修订", type="primary")
                    cannot_edit = not same_endpoint or (item["embedding_provider"] == "company" and not settings.api_key)
                    if save_chunk:
                        if cannot_edit:
                            st.error("请先选择创建此知识库时的模型接口，并配置在线向量化密钥。")
                        else:
                            try:
                                _, pending_cleanup = update_chunk(repo, user["id"], item["id"], selected_doc_id,
                                                                  selected_chunk_id, revised, settings.api_key)
                                st.session_state["rag_chunk_notice"] = (
                                    user["id"], item["id"], "片段已更新；请核对相邻重叠片段。" +
                                    ("旧向量清理失败，但已从检索中排除，请检查服务日志。" if pending_cleanup else ""),
                                    pending_cleanup)
                                st.rerun()
                            except (ValueError, PermissionError, RuntimeError) as exc:
                                st.error(str(exc))
                            except Exception as exc:
                                st.error(f"片段更新失败（{type(exc).__name__}），旧片段未改变。")
                    st.caption("删除只针对当前知识库片段，不修改原始上传文件或已生成的历史回答。"
                               "重叠内容可能仍在相邻片段中。")
                    confirm_chunk = st.checkbox("确认删除选中的片段", key=f"rag_chunk_confirm_{selected_chunk_id}")
                    if st.button("删除选中的片段", disabled=not confirm_chunk or len(chunks) <= 1,
                                 key=f"rag_chunk_delete_{selected_chunk_id}"):
                        if cannot_edit:
                            st.error("请先选择创建此知识库时的模型接口，并配置在线向量化密钥。")
                        else:
                            try:
                                pending_cleanup = delete_chunk(repo, user["id"], item["id"], selected_doc_id,
                                                               selected_chunk_id, settings.api_key)
                                st.session_state["rag_chunk_notice"] = (
                                    user["id"], item["id"], "片段已从检索中移除；请检查相邻重叠片段。" +
                                    ("向量清理失败，请检查服务日志。" if pending_cleanup else ""), pending_cleanup)
                                st.rerun()
                            except (ValueError, PermissionError) as exc:
                                st.error(str(exc))
                            except Exception as exc:
                                st.error(f"片段删除失败（{type(exc).__name__}），请检查索引服务。")
                    if len(chunks) == 1:
                        st.caption("这是文档最后一个片段；如需删除，请使用上方的「删除文档」。")

    with settings_tab:
        st.caption(f"向量模型：{item['embedding_model']} · 切分长度：{item['chunk_size']} · 重叠：{item['chunk_overlap']}")
        with st.form(f"edit_kb_{item['id']}"):
            edit_name = st.text_input("修改名称", value=item["name"], key=f"name_{item['id']}")
            edit_description = st.text_area("修改描述", value=item["description"], key=f"desc_{item['id']}")
            save = st.form_submit_button("保存信息")
        if save:
            try:
                repo.update_base(user["id"], item["id"], edit_name, edit_description)
                st.rerun()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
    with danger_tab:
        st.warning("删除后，本库文档、向量与历史会话将永久移除；其他知识库不受影响。")
        if st.button("删除此知识库", key=f"delete_kb_tab_{item['id']}"):
            confirm_delete_base(item)

section_heading("浏览知识库", "选择另一组资料或继续管理当前知识库")
st.space("small")
filter_kind = st.segmented_control("按类型查看", ["全部", "企业知识库", "学术论文库"], default="全部")
shown = [kb for kb in bases if filter_kind == "全部" or
         kb["kind"] == ("paper" if filter_kind == "学术论文库" else "enterprise")]
st.caption(f"已找到 {len(shown)} 个知识库 · 企业资料与论文互不混用")
if not shown:
    empty("还没有符合条件的知识库", "点开上方“创建新知识库”，开始上传和整理资料。")
for index, item in enumerate(shown):
    if index % 2 == 0:
        row = st.columns(2, gap="medium")
    with row[index % 2]:
        with st.container(border=True, key=f"rag_kb_card_{item['id']}"):
            st.html("<div class='rag-kb-head'><i class='rag-kb-dot paper'></i>论文研究</div>"
                    if item["kind"] == "paper"
                    else "<div class='rag-kb-head'><i class='rag-kb-dot'></i>企业知识</div>")
            st.subheader(item["name"], anchor=False)
            st.write(item["description"] or "为这组资料建立可检索、可追溯的知识空间。")
            st.caption(f"{'论文' if item['kind']=='paper' else '企业资料'} · {item['document_count']} 份可检索文档 · 更新于 {item['updated_at'][:10]}")
            a, b, c = st.columns(3)
            if a.button("管理资料", key=f"manage_{item['id']}", icon=":material/folder_open:", width="stretch"):
                st.session_state["rag_managed_kb"] = item["id"]
                st.rerun()
            if b.button("开始问答", key=f"ask_{item['id']}", icon=":material/arrow_forward:", width="stretch"):
                st.session_state["rag_managed_kb"] = item["id"]
                st.switch_page("app_pages/chat.py")
            if c.button("删除", key=f"delete_{item['id']}", icon=":material/delete:", width="stretch"):
                confirm_delete_base(item)
