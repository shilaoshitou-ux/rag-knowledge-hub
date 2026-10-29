# 评测样例：仅供走通流程，不是实测成绩

三个 Markdown 文件和六道题为**完全虚构的演示数据**；没有任何模型分数、私人资料或真实业务结论。单凭文档名不能衡量证据找得准不准。

1. 在系统中创建测试知识库，把 `docs/` 下面三个文件上传并等到索引成功，记录库 ID。
2. 在仓库根目录运行 `python scripts/run_rag_eval.py --username <本地账号> --kb-id <库ID> --manifest`，按提示输入本地账号密码；片段清单仅保存在被 Git 忽略的 `data/evaluation/`。
3. 对照原文，为 `sample-questions.json` 每题补 `"relevant_chunks":["当前完整片段ID"]`。更新文件会改变 ID，请重新标注；不要猜测 ID。
4. 在当前进程设置 `RAG_EVAL_API_KEY`，运行 `python scripts/run_rag_eval.py --username <本地账号> --kb-id <库ID> --cases examples/evaluation/sample-questions.json --model <实际可用模型ID> --k 4`。
5. 在 `data/evaluation/` 查看题集 SHA-256、模型、K、逐题明细 JSON。结果可能包含模型回答，**不要上传公共仓库**。

未标注片段时只有文档级指标；本样例很小，不能推导真实生产效果。Rerank 回退到 Hybrid 会明确计数。无答案题的弃答率目前尚未实现，不要宣称已评测。
