"""
Searcher Agent（检索员）— 使用 ReAct（Reasoning + Acting）进行搜索
"""
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from config import (
    MODEL_NAME, LLM_BASE_URL, LLM_API_KEY,
    MODEL_TEMPERATURE, MAX_REACT_ITERATIONS, VERBOSE
)
from state import ResearchState
from memory.short_term import ShortTermMemory
from tools.web_search import web_search
from tools.rag_retrieve import rag_retrieve


class SearcherAgent:
    """
    检索员 Agent：
    - 使用真正的 ReAct（Reasoning + Acting）循环
    - LLM 自主决定：何时搜索、搜索什么、是否需要 RAG 检索、何时停止
    - 不是硬编码的工具调用，而是 LLM 通过 bind_tools 自主选择
    """

    def __init__(self):
        self.name = "Searcher"
        self.role = "信息检索员"
        self.short_term = ShortTermMemory(window_size=15, name=self.name)

        self.model = ChatOpenAI(
            model=MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=MODEL_TEMPERATURE,
        )

        # 绑定工具
        self.tools = [web_search, rag_retrieve]
        self.model_with_tools = self.model.bind_tools(self.tools)

    def react_search(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """
        ReAct 循环：Thought → Action → Observation → Thought → ... → Final Answer

        Args:
            query: 搜索查询/研究问题
            context: 上下文信息（研究计划等）
        Returns:
            搜索轨迹列表 [{type, content, ...}, ...]
        """
        trace = []
        system_prompt = f"""你是一位专业的信息检索员。你的任务是根据研究问题进行搜索，收集相关信息。

## 研究背景
{context}

## 搜索任务
{query}

## 工作流程
1. **思考(Thought)**: 分析需要什么信息
2. **行动(Action)**: 使用工具搜索（web_search 搜索互联网，rag_retrieve 检索历史资料）
3. **观察(Observation)**: 分析搜索结果
4. 重复步骤1-3，直到收集到足够信息
5. **最终回答**: 总结所有收集到的信息

## 注意事项
- 先尝试 rag_retrieve 检查历史资料
- 使用 web_search 搜索最新信息（最多2-3次搜索即可）
- 收集到足够信息后，输出总结（不要无限搜索）
- 用中文总结找到的关键信息
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请开始搜索：{query}")
        ]

        if VERBOSE:
            print(f"\n  🔍 [ReAct] 开始搜索: {query}")
            print(f"  {'─'*50}")

        final_answer = None

        for iteration in range(MAX_REACT_ITERATIONS):
            try:
                response = self.model_with_tools.invoke(messages)
            except Exception as e:
                if VERBOSE:
                    print(f"  ⚠️ [ReAct] API调用失败 ({e})，使用工具直接搜索")
                result = web_search.invoke({"query": query})
                trace.append({
                    "iteration": iteration + 1,
                    "thought": "API不可用，直接搜索",
                    "action": {"tool": "web_search", "args": {"query": query}},
                    "observation": str(result)
                })
                trace.append({
                    "iteration": "final",
                    "thought": "搜索完成（API回退模式）",
                    "action": {"tool": "final_answer", "args": {}},
                    "observation": str(result)
                })
                return trace
            
            # API 调用成功，记录模型输出
            messages.append(response)

            # 检查是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})

                    if VERBOSE:
                        print(f"  💭 [ReAct 迭代 {iteration+1}] Thought: 需要调用工具...")
                        print(f"  🔧 Action: {tool_name}({tool_args})")

                    try:
                        if tool_name == "web_search":
                            result = web_search.invoke(tool_args)
                        elif tool_name == "rag_retrieve":
                            result = rag_retrieve.invoke(tool_args)
                        else:
                            result = f"未知工具: {tool_name}"

                        if VERBOSE:
                            result_preview = str(result)[:200]
                            print(f"  👁️  Observation: {result_preview}...")

                    except Exception as e:
                        result = f"工具调用出错: {str(e)}"
                        if VERBOSE:
                            print(f"  ❌ Error: {e}")

                    trace.append({
                        "iteration": iteration + 1,
                        "thought": f"调用 {tool_name}",
                        "action": {"tool": tool_name, "args": tool_args},
                        "observation": str(result)
                    })

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                continue
            else:
                # 模型输出最终答案（无工具调用）
                final_answer = response.content
                if VERBOSE:
                    print(f"  ✅ [ReAct 完成] 搜索结束 ({iteration+1} 轮)")
                    print(f"  最终回答: {final_answer[:300]}...")
                break

        # 如果达到最大迭代次数仍未获得最终答案，将最后一次工具结果作为final_answer（最后一个observation）
        if final_answer is None and trace:
            last_obs = trace[-1].get("observation", "无搜索结果")
            final_answer = f"[ReAct达到最大迭代次数] 最后搜索结果为: {last_obs[:500]}"
            if VERBOSE:
                print(f"  ⚠️ ReAct 达到最大迭代次数，使用最后搜索结果作为答案")

        trace.append({
            "iteration": "final",
            "thought": "搜索完成",
            "action": {"tool": "final_answer", "args": {}},
            "observation": final_answer or "搜索未返回有效结果"
        })

        return trace

    def search_angle(self, plan_item: Dict[str, Any], background: str = "") -> Dict[str, Any]:
        """
        搜索一个研究角度
        """
        keywords = " ".join(plan_item.get("keywords", []))
        question = plan_item.get("question", "")
        angle = plan_item.get("angle", "")

        search_query = f"{angle}: {question}"

        trace = self.react_search(search_query, background)

        # 提取最终的搜索结果（最后一个 trace 的 observation）
        final_result = trace[-1]["observation"] if trace else "无搜索结果"

        result = {
            "angle": angle,
            "question": question,
            "keywords": keywords,
            "search_query": search_query,
            "react_trace": trace,
            "summary": final_result,
            "timestamp": datetime.now().isoformat()
        }

        # 记录到短期记忆
        self.short_term.add({
            "role": self.name,
            "content": f"完成搜索: {angle}",
            "result_summary": final_result[:500]
        })

        return result


# ============ LangGraph 节点函数 ============

def searcher_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph 节点：Searcher
    使用 ReAct 循环搜索。支持两种模式：
    1. 常规模式：按研究计划依次搜索各个角度
    2. 定向补充模式：根据 Critic 识别的空白(gaps)精确搜索
    """
    agent = SearcherAgent()
    research_plan = state.get("research_plan", [])
    gap_queries = list(state.get("gap_queries", []))
    idx = state.get("current_angle_index", 0)
    all_search_results = list(state.get("search_results", []))

    # ---- 定向补充搜索模式 ----
    if gap_queries:
        gap_query = gap_queries[0]  # 取第一个 gap
        remaining_gaps = gap_queries[1:]

        if VERBOSE:
            print(f"\n  🎯 [定向补充] 针对评审空白搜索: {gap_query}")

        plan_item = {
            "angle": f"补充搜索: {gap_query[:50]}",
            "question": gap_query,
            "keywords": gap_query.split()
        }
        background = f"研究主题: {state.get('user_query', '')}\n这是针对评审发现的空白进行的定向补充搜索。"

        result = agent.search_angle(plan_item, background)
        all_search_results.append(result)

        return {
            "search_results": all_search_results,
            "gap_queries": remaining_gaps,
            "needs_search": len(remaining_gaps) > 0,  # 还有 gap 待搜索
            "current_phase": "extracting" if len(remaining_gaps) == 0 else "searching",
            "messages": [{
                "role": agent.name,
                "content": f"定向补充搜索完成: {gap_query[:80]}",
                "index": -1  # gap 搜索
            }]
        }

    # ---- 常规模式：按计划搜索 ----
    if idx >= len(research_plan):
        return {
            "current_phase": "extracting",
            "needs_search": False,
            "messages": [{"role": "Searcher", "content": "所有研究角度已搜索完毕"}]
        }

    plan_item = research_plan[idx]
    background = f"研究主题: {state.get('user_query', '')}\n研究角度: {plan_item.get('angle', '')}"

    if VERBOSE:
        print(f"\n  📚 搜索角度 [{idx+1}/{len(research_plan)}]: {plan_item.get('angle', '')}")

    result = agent.search_angle(plan_item, background)
    plan_item["status"] = "searched"
    research_plan[idx] = plan_item
    all_search_results.append(result)

    return {
        "search_results": all_search_results,
        "research_plan": research_plan,
        "current_angle_index": idx + 1 if idx + 1 < len(research_plan) else idx,
        "needs_search": idx + 1 < len(research_plan),
        "current_phase": "extracting" if idx + 1 >= len(research_plan) else "searching",
        "messages": [{
            "role": agent.name,
            "content": f"完成搜索角度: {plan_item.get('angle', '')}",
            "index": idx
        }]
    }
