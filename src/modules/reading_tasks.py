"""Focused, evidence-first reading tasks for research papers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadingTask:
    slug: str
    title: str
    description: str
    question: str
    search_queries: tuple[str, ...]
    icon: str


READING_TASKS = (
    ReadingTask(
        "overview", "论文速览", "用三分钟定位问题、方法与结论。",
        "请基于检索到的原文，为我制作论文速览：研究问题、作者提出的方法、主要实验发现各是什么？逐条标明证据；没有检索到的部分明确说无法确认。",
        ("论文摘要 abstract 研究背景 research problem motivation", "论文主要方法 proposed method contribution novelty", "论文结论 conclusion experimental findings limitations"),
        ":material/overview:",
    ),
    ReadingTask(
        "problem", "问题与贡献", "分清作者的研究动机和真正的新意。",
        "作者试图解决什么具体研究问题？已有方法有什么不足？本文声称的贡献分别是什么？请区分作者提出的方法和相关工作，附来源。",
        ("研究动机 motivation prior work limitations", "本文贡献 our contributions proposed approach novelty", "related work 与本文方法区别 difference from prior work"),
        ":material/psychology:",
    ),
    ReadingTask(
        "method", "方法拆解", "沿着输入、关键模块、输出梳理流程。",
        "请拆解这篇论文实际提出的方法：输入与输出是什么，关键模块与步骤如何衔接，核心公式或算法在哪里？不要把相关工作误当成本文方法；没有证据的环节请标明。",
        ("本文提出 proposed method architecture pipeline", "algorithm module equation input output", "ablation study method components"),
        ":material/account_tree:",
    ),
    ReadingTask(
        "experiment", "实验与指标", "搞清楚数据集、基线和评价设计。",
        "论文如何验证提出的方法？列出检索证据覆盖到的数据集、基线、评价指标和实验设置；缺失信息不要推测。",
        ("experimental setup dataset baselines", "evaluation metrics experimental conditions hyperparameters", "ablation comparison experiments"),
        ":material/science:",
    ),
    ReadingTask(
        "results", "结果与结论", "把主张和具体数字对应起来。",
        "作者报告了哪些核心结果？如有数字，请说明对应指标、对照对象和实验条件。哪些是原文报告的事实，哪些只是作者的解释？引用相应来源。",
        ("results table metrics improvement comparison", "analysis results discussion", "conclusions effectiveness conditions"),
        ":material/analytics:",
    ),
    ReadingTask(
        "limitations", "局限与追问", "检查论文没有回答的关键问题。",
        "论文明确写了哪些局限、假设或未来工作？请将作者明确承认的局限与基于现有证据提出的待验证问题分开；不要将你的推测当作作者结论。",
        ("limitations assumptions constraints", "future work directions", "conclusions scope threats to validity"),
        ":material/verified:",
    ),
)