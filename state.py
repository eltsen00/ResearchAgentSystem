"""
ResearchState — LangGraph 全局状态定义
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class ResearchState(TypedDict):
    """研究系统的全局状态"""
    # 用户输入
    user_query: str  # 用户的研究主题/问题

    # 消息历史（短期记忆 — 所有 Agent 的对话/思维/工具调用轨迹）
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # 研究计划（Orchestrator 产出）
    research_plan: List[Dict[str, Any]]  # [{angle, score, keywords}, ...]

    # 搜索结果（Searcher 产出）
    search_results: List[Dict[str, Any]]  # [{query, results, timestamp}, ...]

    # 提取的知识（Extractor 产出）
    extracted_facts: List[Dict[str, Any]]  # [{fact, source, category, timestamp}, ...]

    # 评审结果（Critic 产出）
    critique: Dict[str, Any]  # {strengths, weaknesses, gaps, suggestions, score}

    # 最终报告（Writer 产出）
    final_report: str

    # ---- 控制标志 ----
    # 当前阶段: "orchestrating" | "searching" | "extracting" | "reflecting" | "writing" | "done"
    current_phase: str

    # 是否需要搜索
    needs_search: bool

    # 是否需要重新搜索（Critic 发现空白）
    needs_revision: bool

    # 反思轮次计数（防止无限循环）
    reflection_round: int

    # 研究是否完成
    research_complete: bool

    # 当前正在处理的研究角度索引
    current_angle_index: int

    # 定向补充搜索：Critic 发现的具体空白对应的搜索查询
    gap_queries: List[str]

    # 上一轮评审分数（用于检测是否停滞）
    previous_score: Optional[float]

    # 错误信息
    error: Optional[str]
