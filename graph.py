"""
LangGraph 图构建 — 定义多智能体研究系统的完整工作流
支持动态条件路由：Orchestrator → Searcher ⇄ Extractor → Critic ⇄ Writer
"""
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.searcher import searcher_node
from agents.extractor import extractor_node
from agents.critic import critic_node
from agents.writer import writer_node
from memory.long_term import init_long_term_memory
from config import VERBOSE, CLEAR_MEMORY_ON_START


# ============ 条件路由函数 ============

def route_after_orchestrator(state: ResearchState) -> Literal["searcher", END]:
    """Orchestrator 之后的 routing：有研究计划 → searcher，否则 → END"""
    error = state.get("error")
    if error:
        if VERBOSE:
            print(f"  ❌ Orchestrator 错误: {error}")
        return END

    plan = state.get("research_plan", [])
    needs_search = state.get("needs_search", False)

    if needs_search and plan:
        return "searcher"
    else:
        if VERBOSE:
            print("  ⚠️ 无需搜索，直接结束")
        return END


def route_after_searcher(state: ResearchState) -> Literal["searcher", "extractor"]:
    """Searcher 之后的 routing：needs_search 标志由 searcher_node 正确计算，直接信任"""
    needs_search = state.get("needs_search", False)
    if needs_search:
        return "searcher"
    else:
        return "extractor"


def route_after_extractor(state: ResearchState) -> Literal["critic", "writer", END]:
    """Extractor 之后的 routing：有提取结果 → critic，少量结果 → writer（跳过评审），无任何结果 → END"""
    facts = state.get("extracted_facts", [])
    search_results = state.get("search_results", [])
    if facts:
        return "critic"
    elif search_results:
        # 有搜索结果但没提取到事实，跳到 writer（至少写个摘要报告）
        return "writer"
    else:
        if VERBOSE:
            print("  ⚠️ 无任何搜索或提取结果，结束流程")
        return END


def route_after_critic(state: ResearchState) -> Literal["searcher", "writer", END]:
    """Critic 之后的 routing：需要修订 → searcher（重新搜索），通过 → writer，否则 → END"""
    needs_revision = state.get("needs_revision", False)
    needs_search = state.get("needs_search", False)
    reflection_round = state.get("reflection_round", 0)

    if needs_revision and needs_search:
        from config import MAX_REFLECTION_ROUNDS
        if reflection_round < MAX_REFLECTION_ROUNDS:
            if VERBOSE:
                print(f"  🔄 反思轮次 {reflection_round}: 重新搜索")
            return "searcher"
        else:
            if VERBOSE:
                print(f"  ⚠️ 已达到最大反思轮次 ({MAX_REFLECTION_ROUNDS})")
            return "writer"
    else:
        return "writer"


# ============ 图构建 ============

def build_research_graph() -> StateGraph:
    """
    构建多智能体研究系统的 LangGraph 图。

    图结构:
        orchestrator → searcher (循环搜索各角度) → extractor → critic
                      ↑___________________________________|
                      (如评审不通过则重新搜索)
                      ↓
                    writer → END

    使用动态条件边（conditional_edges）而非固定顺序，
    每个 Agent 会根据状态自主决定下一步去向。
    """
    # 初始化长期记忆（全局单例，跨会话保留数据）
    init_long_term_memory(clear_first=CLEAR_MEMORY_ON_START)

    if VERBOSE:
        from memory.long_term import get_long_term_memory
        long_term = get_long_term_memory()
        stats = long_term.get_stats()
        clear_msg = "（已清空旧数据）" if CLEAR_MEMORY_ON_START else f"（跨会话保留，含 {stats['total_documents']} 条历史记录）"
        print(f"💾 长期记忆已初始化: {stats['collection_name']} {clear_msg}")

    # 创建 StateGraph
    workflow = StateGraph(ResearchState)

    # 添加所有节点
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    # 设置入口点
    workflow.set_entry_point("orchestrator")

    # 添加条件边（动态路由）
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "searcher": "searcher",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "searcher",
        route_after_searcher,
        {
            "searcher": "searcher",    # 继续搜索下一个角度
            "extractor": "extractor"    # 所有角度搜完，进入提取
        }
    )

    workflow.add_conditional_edges(
        "extractor",
        route_after_extractor,
        {
            "critic": "critic",
            "writer": "writer",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "searcher": "searcher",     # 评审不通过，重新搜索
            "writer": "writer",         # 评审通过，撰写报告
            END: END
        }
    )

    # Writer 之后结束
    workflow.add_edge("writer", END)

    # 编译图（带检查点支持状态持久化）
    memory_saver = MemorySaver()
    graph = workflow.compile(checkpointer=memory_saver)

    if VERBOSE:
        print("✅ 研究图构建完成")
        print(f"   节点: orchestrator → searcher ⇄ extractor → critic ⇄ writer")
        print(f"   路由: 动态条件边，支持重新搜索循环")

    return graph
