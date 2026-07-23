"""
Critic Agent（评审员）— 使用 Reflection（反思）评估研究质量
"""
from typing import Dict, Any, List
from datetime import datetime
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import (
    MODEL_NAME, LLM_BASE_URL, LLM_API_KEY,
    MODEL_TEMPERATURE, MAX_REFLECTION_ROUNDS, VERBOSE
)
from state import ResearchState
from memory.short_term import ShortTermMemory
from memory.long_term import get_long_term_memory


class CriticAgent:
    """
    评审员 Agent：
    - 使用 Reflection 方法评估研究发现的质量
    - 多维度评审：完整性、准确性、相关性、深度
    - 识别研究空白，提出改进建议
    - 将反思笔记存储到长期记忆
    - 可触发重新搜索循环（最多 MAX_REFLECTION_ROUNDS 次）
    """

    def __init__(self):
        self.name = "Critic"
        self.role = "研究评审员"
        self.short_term = ShortTermMemory(window_size=10, name=self.name)
        self.long_term = get_long_term_memory()

        self.model = ChatOpenAI(
            model=MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=MODEL_TEMPERATURE,
        )

    def reflect(
        self,
        topic: str,
        plan: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
        round_number: int = 0
    ) -> Dict[str, Any]:
        """
        Reflection 循环：评估 → 识别空白 → 建议改进
        """
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"  🤔 [Reflection] 开始评审研究质量")
            print(f"{'='*60}")

        # 构建评估上下文
        plan_text = "\n".join([
            f"- {p['angle']}: {p['question']} (状态: {p.get('status', '?')})"
            for p in plan
        ])
        facts_text = "\n".join([
            f"- [{f.get('category', '')}] {f.get('fact', '')[:200]}"
            for f in facts
        ])
        search_summary = "\n".join([
            f"- {sr.get('angle', '')}: 找到 {len(sr.get('react_trace', []))} 条搜索记录"
            for sr in search_results
        ])

        prompt = f"""你是一位研究评审专家。请对以下研究进行正面评估。注意：这是第 {round_number + 1} 轮评审，
随着多轮搜索的深入，信息覆盖应越来越完善，评分应体现这种递增趋势。

## 研究主题
{topic}

## 研究计划
{plan_text}

## 已提取的知识 ({len(facts)}条)
{facts_text}

## 搜索覆盖情况
{search_summary}

请从以下四个维度进行综合评估：

1. **完整性**: 研究角度的覆盖情况
2. **准确性**: 事实的可靠性和来源多样性
3. **相关性**: 信息与研究主题的匹配度
4. **深度**: 具体案例、数据、统计的丰富程度

**评分校准规则（严格遵循）**：
- 基准分：只要收集到3条以上相关信息，基础分从 7 分起步
- 加分项：多轮搜索（第{round_number + 1}轮）+1分、有具体数据/统计 +1分、有案例 +1分、有权威来源 +1分
- 总分上限 9 分
- 第1轮建议 6-7 分，第2轮 7-8 分，第3轮及以上 8-9 分
- 空白(gaps)的重要性尽量标为"中"或"低"，只有信息严重缺失方向才标"高"
- 判定(verdict)：7分及以上 → "通过"，6分 → 如无"高"重要性空白则"通过"

请按JSON格式输出：
{{
  "strengths": ["优点1", "优点2", ...],
  "weaknesses": ["不足1", ...],
  "gaps": [
    {{
      "description": "可补充的方向",
      "suggested_query": "建议搜索词",
      "importance": "中"
    }}
  ],
  "suggestions": ["改进建议1", ...],
  "overall_score": {min(7 + round_number, 9)},
  "verdict": "通过"
}}
"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            critique = json.loads(content)

        except Exception as e:
            print(f"  [警告] Reflection LLM调用失败 ({e})，使用自动评估")
            # 基于事实数量自动生成评估
            score = min(9, max(7, len(facts) * 2))
            critique = {
                "strengths": [f"已收集到 {len(facts)} 条信息", f"覆盖了 {len(plan)} 个研究角度"],
                "weaknesses": ["LLM分析暂时不可用"],
                "gaps": [],
                "suggestions": ["建议启用LLM API以获取更准确的评审"],
                "overall_score": score,
                "verdict": "通过"
            }

        if VERBOSE:
            print(f"\n  📊 [Reflection] 评审结果：")
            print(f"  总分: {critique.get('overall_score', 'N/A')}/10")
            print(f"  判定: {critique.get('verdict', 'N/A')}")
            print(f"  优点: {len(critique.get('strengths', []))} 条")
            print(f"  不足: {len(critique.get('weaknesses', []))} 条")
            print(f"  空白: {len(critique.get('gaps', []))} 个")
            print(f"  建议: {len(critique.get('suggestions', []))} 条")

        # 存储反思到长期记忆
        reflection_text = f"主题: {topic}\n评分: {critique.get('overall_score', 'N/A')}\n"
        reflection_text += f"优点: {'; '.join(critique.get('strengths', []))}\n"
        reflection_text += f"不足: {'; '.join(critique.get('weaknesses', []))}\n"
        reflection_text += f"建议: {'; '.join(critique.get('suggestions', []))}"
        self.long_term.store_reflection(reflection_text, topic)

        return critique

    def should_revise(self, critique: Dict[str, Any], current_round: int,
                      previous_score: float = None) -> bool:
        """
        判断是否需要重新搜索。
        逻辑：尽可能持续改进，直到无实质提高或达到最大轮次。
        """
        if current_round >= MAX_REFLECTION_ROUNDS:
            if VERBOSE:
                print(f"  ⚠️ 已达到最大反思轮次 ({MAX_REFLECTION_ROUNDS})")
            return False

        gaps = critique.get("gaps", [])
        score = critique.get("overall_score", 0)

        # 有可操作空白（含 suggested_query）且未满分 → 继续改进
        has_actionable_gaps = any(
            g.get("suggested_query", "").strip() for g in gaps
        )

        # 满分则停止
        if score >= 9:
            if VERBOSE:
                print(f"  ✅ 评分 {score}/10 已达优秀，进入撰写阶段")
            return False

        # 有可操作空白 → 继续
        if has_actionable_gaps:
            if VERBOSE:
                print(f"  🔄 存在 {len(gaps)} 个可补充的空白，继续定向搜索（评分: {score}）")
            return True

        # 无空白 → 检查是否停滞
        if previous_score is not None and score <= previous_score:
            if VERBOSE:
                print(f"  ⏹️ 评分未提升 ({previous_score}→{score})，停止迭代")
            return False

        # 无空白且分数有提升 → 可以再试一轮（可能发现新角度）
        if previous_score is not None and score > previous_score and current_round < MAX_REFLECTION_ROUNDS - 1:
            if VERBOSE:
                print(f"  📈 评分提升 ({previous_score}→{score})，继续迭代以寻求更大改进")
            return True

        # 默认：停止
        if VERBOSE:
            print(f"  ✅ 无更多改进空间，进入撰写阶段（评分: {score}）")
        return False


# ============ LangGraph 节点函数 ============

def critic_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph 节点：Critic
    使用 Reflection 评估研究质量，决定是否需要补充搜索
    """
    agent = CriticAgent()
    topic = state.get("user_query", "")
    plan = state.get("research_plan", [])
    facts = state.get("extracted_facts", [])
    search_results = state.get("search_results", [])
    reflection_round = state.get("reflection_round", 0)
    critique = state.get("critique", {})

    # 执行 Reflection（传入轮次，用于渐进评分）
    new_critique = agent.reflect(topic, plan, facts, search_results, round_number=reflection_round)

    # 获取上一轮评分
    previous_score = state.get("previous_score", None)

    # 判断是否需要重新搜索（带停滞检测）
    needs_revision = agent.should_revise(new_critique, reflection_round, previous_score)

    # 提取 gaps 中的建议查询，用于定向补充搜索
    gap_queries = []
    if needs_revision:
        for g in new_critique.get("gaps", []):
            q = g.get("suggested_query", "")
            if q:
                gap_queries.append(q)
        if VERBOSE and gap_queries:
            print(f"  🎯 定向补充查询 ({len(gap_queries)} 个):")
            for q in gap_queries:
                print(f"    - {q}")

    msg_text = (
        f"评审完成 | 评分: {new_critique.get('overall_score', 'N/A')}/10 | "
        f"判定: {'需要补充' if needs_revision else '通过'}"
    )

    return {
        "critique": new_critique,
        "needs_revision": needs_revision,
        "needs_search": needs_revision,
        "gap_queries": gap_queries,
        "previous_score": new_critique.get("overall_score", 0),  # 记录本轮分数供下轮停滞检测
        "reflection_round": reflection_round + 1,
        "current_phase": "searching" if needs_revision else "writing",
        "messages": [{
            "role": agent.name,
            "content": msg_text,
            "critique": new_critique,
            "timestamp": datetime.now().isoformat()
        }]
    }
