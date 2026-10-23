"""Managed knowledge bases: card overview and focused document workspace."""

import streamlit as st

from modules.auth_ui import require_user
from modules.embedder import UploadedDocument
from modules.managed_index import delete_base, delete_document, interrupted_count, queue_upload, recover_interrupted, retry_document
from modules.managed_ui import render_model_settings
from modules.repository import Repository
from modules.ui_kit import empty, page_title

repo = Repository()
user = require_user(repo)
settings = render_model_settings(user["id"])
page_title("LIBRARY / KNOWLEDGE BASES", "我的知识库", "把企业资料和学术论文放进独立空间；每次问答都能回到原始文档。")
bases = repo.list_bases(user["id"])
with st.expander("创建新知识库", expanded=not bases, icon=":material/add_circle:"):
    st.caption("向量模型在创建时绑定到知识库，避免不同维度的索引混用。")
    with st.form("new_base"):
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

st.space("small")
filter_kind = st.segmented_control("按类型查看", ["全部", "企业知识库", "学术论文库"], default="全部")
shown = [kb for kb in bases if filter_kind == "全部" or
         kb["kind"] == ("paper" if filter_kind == "学术论文库" else "enterprise")]
if not shown:
    empty("还没有符合条件的知识库", "点开上方“创建新知识库”，开始上传和整理资料。")
for index, item in enumerate(shown):
    if index % 2 == 0:
        row = st.columns(2, gap="medium")
    with row[index % 2]:
        with st.container(border=True):
            st.caption("RESEARCH COLLECTION" if item["kind"] == "paper" else "TEAM KNOWLEDGE")
            st.subheader(item["name"], anchor=False)
            st.write(item["description"] or "为这组资料建立可检索、可追溯的知识空间。")
            st.caption(f"{'论文' if item['kind']=='paper' else '企业资料'} · {item['document_count']} 份可检索文档 · 更新于 {item['updated_at'][:10]}")
            a, b = st.columns(2)
            if a.button("管理资料", key=f"manage_{item['id']}", icon=":material/folder_open:", width="stretch"):
                st.session_state["rag_managed_kb"] = item["id"]
                st.rerun()
            if b.button("开始问答", key=f"ask_{item['id']}", icon=":material/arrow_forward:", width="stretch"):
                st.session_state["rag_managed_kb"] = item["id"]
                st.switch_page("app_pages/chat.py")

active_id = st.session_state.get("rag_managed_kb")
item = next((kb for kb in bases if kb["id"] == active_id), None)
if item:
    st.divider()
    page_title("COLLECTION / MANAGE", item["name"], "在这里上传、更新或删除资料；正在处理的文档会显示实时状态。")
    same_endpoint = settings.base_url == item["base_url"]
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
            for doc in docs:
                with st.container(border=True):
                    title, action = st.columns([4, 1], vertical_alignment="center")
                    title.markdown(f"**{doc['name']}** · v{doc['version']}")
                    action.caption(status_label.get(doc["status"], doc["status"]))
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
        st.warning("删除后，本库文档、向量与历史会话将永久移除；其他知识库和旧版索引不受影响。")
        with st.form(f"delete_kb_{item['id']}"):
            confirm_name = st.text_input("输入知识库完整名称确认")
            confirmed = st.form_submit_button("永久删除此知识库")
        if confirmed:
            if confirm_name != item["name"]:
                st.error("名称不一致，未删除。")
            elif not same_endpoint or (item["embedding_provider"] == "company" and not settings.api_key):
                st.error("请先在侧栏选择该库创建时的接口并填写对应密钥。")
            else:
                try:
                    delete_base(repo, user["id"], item, settings.api_key)
                    st.session_state.pop("rag_managed_kb", None)
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
