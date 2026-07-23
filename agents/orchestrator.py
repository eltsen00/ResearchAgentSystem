"""
Orchestrator Agent（协调者）— 使用 ToT（Tree of Thoughts）进行任务规划
"""
import json
import os
from typing import Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import (
    MODEL_NAME, LLM_BASE_URL, LLM_API_KEY,
    MODEL_TEMPERATURE, TOT_NUM_BRANCHES, TOT_TOP_K, VERBOSE
)
from state import ResearchState
from memory.short_term import ShortTermMemory


class OrchestratorAgent:
    """
    协调者 Agent：
    - 使用 ToT 将研究主题分解为多个研究角度（树分支）
    - 对每个分支进行评分
    - 选择最优分支进行深入探索
    - 生成结构化的研究计划
    """

    def __init__(self):
        self.name = "Orchestrator"
        self.role = "研究协调者"
        self.short_term = ShortTermMemory(window_size=10, name=self.name)

        self.model = ChatOpenAI(
            model=MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=MODEL_TEMPERATURE,
        )

    def tot_explore(self, topic: str, num_branches: int = TOT_NUM_BRANCHES) -> list:
        """
        ToT 阶段 1：探索 — 生成多个研究角度（树分支）
        """
        prompt = f"""你是一位资深研究顾问。用户想研究以下主题：

"{topic}"

请从 {num_branches} 个不同的角度/维度来分解这个研究主题，每个角度成为一个独立的研究分支。
每个分支应包含：
1. 研究角度名称（简短）
2. 核心研究问题
3. 3-5个搜索关键词

请按以下JSON格式输出（只输出JSON，不要其他内容）：
[
  {{
    "angle": "研究角度名称",
    "question": "核心研究问题",
    "keywords": ["关键词1", "关键词2", "关键词3"]
  }},
  ...
]
"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 尝试提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            branches = json.loads(content) # 解释为 Python 字典
            if VERBOSE:
                print(f"\n  [ToT 探索] 生成了 {len(branches)} 个研究角度：")
                for b in branches:
                    print(f"    - {b['angle']}: {b['question']}")
            return branches[:num_branches]
        except Exception as e:
            print(f"  [警告] ToT探索失败 ({e})，终止程序")
            os._exit(1)
            

    def tot_evaluate(self, topic: str, branches: list) -> list:
        """
        ToT 阶段 2：评估 — 对每个分支打分
        """
        if len(branches) <= 1:
            return branches

        # 构建评估提示
        branches_text = "\n".join([
            f"{i+1}. {b['angle']}: {b['question']}"
            for i, b in enumerate(branches)
        ])

        prompt = f"""评估以下关于"{topic}"的研究角度。对每个角度在以下三个维度上打分（1-10分）：

维度说明：
- 可行性(Feasibility)：该角度是否易于研究和查找资料
- 重要性(Importance)：该角度对理解主题的重要性
- 深度(Depth)：该角度能否深入挖掘出有价值的内容

研究角度：
{branches_text}

请按以下JSON格式输出（只输出JSON）：
[
  {{"angle": "角度名", "feasibility": 8, "importance": 7, "depth": 6, "total": 21}},
  ...
]
"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            scores = json.loads(content) # 解释为 Python 字典

            # 将分数合并到分支
            for branch in branches:
                for s in scores:
                    if s["angle"] in branch["angle"] or branch["angle"] in s["angle"]:
                        branch["score"] = s.get("total", sum([
                            s.get("feasibility", 5),
                            s.get("importance", 5),
                            s.get("depth", 5)
                        ]))
                        branch["score_detail"] = s
                        break
                if "score" not in branch:
                    branch["score"] = 5

            if VERBOSE:
                print(f"\n  [ToT 评估] 各分支评分：")
                for b in sorted(branches, key=lambda x: x.get("score", 0), reverse=True):
                    print(f"    - {b['angle']}: {b.get('score', 'N/A')}分")

            return branches
        except Exception as e:
            print(f"  [警告] ToT评估失败 ({e})，使用原始顺序")
            for b in branches:
                b["score"] = 5
            return branches

    def tot_select(self, branches: list, top_k: int = TOT_TOP_K) -> list:
        """
        ToT 阶段 3：选择 — 选出最优分支
        """
        sorted_branches = sorted(
            branches,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        selected = sorted_branches[:top_k]

        if VERBOSE:
            print(f"\n  [ToT 选择] 从 {len(branches)} 个角度中选出最优 {len(selected)} 个：")
            for b in selected:
                print(f"    ✅ {b['angle']} (得分: {b.get('score', 'N/A')})")

        return selected

    def orchestrate(self, topic: str) -> Dict[str, Any]:
        """
        完整的 ToT 流程：探索 → 评估 → 选择 → 生成研究计划
        """
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"  🌳 [Orchestrator] 开始 Tree of Thoughts 规划")
            print(f"  研究主题: {topic}")
            print(f"{'='*60}")

        # 阶段 1: 探索
        branches = self.tot_explore(topic)

        # 阶段 2: 评估
        branches = self.tot_evaluate(topic, branches)

        # 阶段 3: 选择
        selected = self.tot_select(branches)

        # 生成研究计划
        research_plan = []
        for i, branch in enumerate(selected):
            research_plan.append({
                "priority": i + 1,
                "angle": branch["angle"],
                "question": branch["question"],
                "keywords": branch.get("keywords", []),
                "score": branch.get("score", 0),
                "status": "pending"
            })

        plan_msg = {
            "role": self.name,
            "content": f"研究计划已生成：{len(research_plan)} 个研究方向",
            "plan": research_plan,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term.add(plan_msg)

        if VERBOSE:
            print(f"\n  📋 最终研究计划：")
            for item in research_plan:
                print(f"    {item['priority']}. [{item['angle']}] {item['question']}")

        return {
            "research_plan": research_plan,
            "plan_message": plan_msg
        }

    # def _fallback_branches(self, topic: str) -> list:
    #     """当 LLM 调用失败时的回退分支"""
    #     return [
    #         {
    #             "angle": f"{topic}的基础概念与理论框架",
    #             "question": f"{topic}的核心概念是什么？有哪些主要理论？",
    #             "keywords": [topic, "基础概念", "理论框架", "定义"]
    #         },
    #         {
    #             "angle": f"{topic}的发展现状与最新进展",
    #             "question": f"{topic}当前发展状况如何？有哪些最新突破？",
    #             "keywords": [topic, "发展现状", "最新进展", "2024", "趋势"]
    #         },
    #         {
    #             "angle": f"{topic}的应用实践与案例分析",
    #             "question": f"{topic}在实际中如何应用？有哪些典型案例？",
    #             "keywords": [topic, "应用", "案例", "实践", "落地"]
    #         },
    #         {
    #             "angle": f"{topic}的挑战与未来展望",
    #             "question": f"{topic}面临哪些挑战？未来发展方向是什么？",
    #             "keywords": [topic, "挑战", "问题", "未来", "展望"]
    #         },
    #     ]


# ============ LangGraph 节点函数 ============

def orchestrator_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph 节点：Orchestrator
    使用 ToT 进行任务规划，返回研究计划
    """
    agent = OrchestratorAgent()
    topic = state.get("user_query", "")

    if not topic:
        return {
            "error": "未提供研究主题",
            "research_complete": True,
            "messages": [{"role": "Orchestrator", "content": "错误：未收到研究主题"}]
        }

    result = agent.orchestrate(topic)

    return {
        "research_plan": result["research_plan"],
        "needs_search": True,
        "needs_revision": False,
        "current_phase": "searching",
        "current_angle_index": 0,
        "reflection_round": 0,
        "messages": [result["plan_message"]]
    }
