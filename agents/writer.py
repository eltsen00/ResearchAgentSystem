"""
Writer Agent（撰写员）— 综合所有研究发现，生成最终研究报告
"""
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import (
    MODEL_NAME, LLM_BASE_URL, LLM_API_KEY,
    MODEL_TEMPERATURE, VERBOSE
)
from state import ResearchState
from memory.short_term import ShortTermMemory
from memory.long_term import get_long_term_memory


class WriterAgent:
    """
    撰写员 Agent：
    - 综合所有研究发现：研究计划、搜索结果、提取的事实、评审意见
    - 生成结构化 Markdown 研究报告
    - 自动包含：标题、摘要、关键发现、详细分析、参考文献、局限性
    - 从长期记忆中检索历史相关研究作为参考
    """

    def __init__(self):
        self.name = "Writer"
        self.role = "报告撰写员"
        self.short_term = ShortTermMemory(window_size=15, name=self.name)
        self.long_term = get_long_term_memory()

        self.model = ChatOpenAI(
            model=MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=MODEL_TEMPERATURE,
        )

    def write_report(
        self,
        topic: str,
        plan: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
        critique: Dict[str, Any]
    ) -> str:
        """
        撰写最终研究报告
        """
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"  ✍️  [Writer] 开始撰写研究报告")
            print(f"{'='*60}")

        # 准备材料
        plan_text = "\n".join([
            f"### {p.get('angle', '')}\n研究问题: {p.get('question', '')}\n关键词: {', '.join(p.get('keywords', []))}"
            for p in plan
        ])

        facts_text = "\n".join([
            f"- [{f.get('category', '')}] {f.get('fact', '')} (来源: {f.get('source', '')}, 可信度: {f.get('confidence', '')})"
            for f in facts
        ])

        search_text = "\n".join([
            f"### {sr.get('angle', '')}\n{sr.get('summary', '')[:500]}"
            for sr in search_results
        ])

        # ---- 从搜索结果中提取真实来源（标题 + URL）----
        verified_sources = []
        seen_urls = set()
        for sr in search_results:
            for step in sr.get("react_trace", []):
                obs = step.get("observation", "")
                lines = obs.split("\n")
                current_title = None
                for line in lines:
                    # 提取标题行: **1. 标题文字**
                    if line.startswith("**") and ".**" in line[:6]:
                        parts = line.split("**", 2)
                        if len(parts) >= 3:
                            current_title = parts[2].strip()
                    # 提取 URL
                    if line.startswith("来源:") and "http" in line:
                        url = line.replace("来源:", "").strip()
                        if url not in seen_urls and "example.com" not in url:
                            seen_urls.add(url)
                            title = current_title or "无标题"
                            verified_sources.append(f"{title} — {url}")
                            current_title = None
        sources_text = "\n".join(verified_sources[:30]) if verified_sources else "(无已验证来源)"
        # ----

        critique_text = f"""评审总分: {critique.get('overall_score', 'N/A')}/10
优点: {'; '.join(critique.get('strengths', []))}
不足: {'; '.join(critique.get('weaknesses', []))}
改进建议: {'; '.join(critique.get('suggestions', []))}"""

        prompt = f"""你是一位资深研究报告撰写专家。请根据以下研究资料，撰写一份专业的研究报告。

## 研究主题
{topic}

## 研究计划
{plan_text}

## 搜索与研究发现
{search_text}

## 提取的关键知识
{facts_text}

## 质量评审意见
{critique_text}

## ⚠️ 已验证可用的真实来源（标题 — URL，只有这些真实存在）
{sources_text}

## 🚨 参考文献严格规则（必须遵守！）
- **只能从上面真实来源列表中逐条选取，每条必须包含标题和URL**
- **参考文献格式：`[序号] 标题 — URL`（从上面列表中复制）**
- **禁止编造任何不存在的论文、书籍、报告、网址**
- **正文用 [1][2] 标注引用，对应参考文献序号**
- **如果真实来源不足，宁可少列，绝不造假**

请直接输出Markdown格式的报告，无需额外说明。
"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            report = response.content.strip()

            # 去除可能的代码块标记
            if report.startswith("```markdown"):
                report = report[len("```markdown"):].strip()
            elif report.startswith("```"):
                report = report[3:].strip()
            if report.endswith("```"):
                report = report[:-3].strip()

            if VERBOSE:
                print(f"  ✅ 报告撰写完成 ({len(report)} 字符)")

            # 存储报告到长期记忆
            self.long_term.store_report(report, topic)

            if VERBOSE:
                stats = self.long_term.get_stats()
                print(f"  💾 报告已存入长期记忆（总计 {stats['total_documents']} 条记录）")

            return report

        except Exception as e:
            print(f"  [警告] 报告撰写失败 ({e})，生成简化报告")
            return self._generate_simple_report(topic, plan, facts)

    def _generate_simple_report(self, topic: str, plan: list, facts: list) -> str:
        """简化报告生成（当LLM生成失败时使用）"""
        sections = [
            f"# 研究报告: {topic}\n",
            f"## 摘要\n本研究围绕'{topic}'展开系统研究，通过多智能体协作方式收集和分析相关资料。\n",
            f"## 引言\n{topic}是一个值得深入研究的重要领域。本研究旨在全面了解该主题的各个方面。\n",
            f"## 研究方法\n本研究采用基于LangGraph的多智能体协作系统，包括研究规划、信息检索、知识提取和质量评审四个阶段。\n",
            f"## 研究角度\n共规划了 {len(plan)} 个研究角度：\n"
        ]

        for p in plan:
            sections.append(f"- **{p.get('angle', '')}**: {p.get('question', '')}\n")

        sections.append(f"\n## 关键发现\n共提取 {len(facts)} 条关键信息：\n")
        for f in facts[:20]:
            sections.append(f"- [{f.get('category', '')}] {f.get('fact', '')}\n")

        sections.append(f"\n## 结论\n基于以上发现，{topic}涉及多方面因素，需要综合考虑不同角度的观点和证据。\n")
        sections.append(f"\n## 研究局限性\n本研究信息收集可能不全面，搜索引擎覆盖范围有限，建议进一步深入调研。\n")
        sections.append(f"\n> 本报告由智能研究助手系统自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        return "".join(sections)


# ============ LangGraph 节点函数 ============

def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph 节点：Writer
    综合所有信息，生成最终研究报告
    """
    agent = WriterAgent()
    topic = state.get("user_query", "")
    plan = state.get("research_plan", [])
    facts = state.get("extracted_facts", [])
    search_results = state.get("search_results", [])
    critique = state.get("critique", {})

    if VERBOSE:
        print(f"\n  最终研究状态：")
        print(f"  - 研究角度: {len(plan)} 个")
        print(f"  - 搜索结果: {len(search_results)} 组")
        print(f"  - 提取知识: {len(facts)} 条")
        print(f"  - 评审分数: {critique.get('overall_score', 'N/A')}")

    report = agent.write_report(topic, plan, facts, search_results, critique)

    return {
        "final_report": report,
        "research_complete": True,
        "current_phase": "done",
        "needs_search": False,
        "needs_revision": False,
        "messages": [{
            "role": agent.name,
            "content": f"研究报告撰写完成 | 长度: {len(report)} 字符",
            "timestamp": datetime.now().isoformat()
        }]
    }
