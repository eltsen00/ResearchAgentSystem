"""
Extractor Agent（提取员）— 从搜索结果中提取结构化知识，存储到长期记忆
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


class ExtractorAgent:
    """
    提取员 Agent：
    - 从原始搜索结果中提取结构化事实、数据、观点
    - 使用 RAG 检索长期记忆中相关内容作为补充
    - 将提取的知识存储到长期记忆（ChromaDB）
    """

    def __init__(self):
        self.name = "Extractor"
        self.role = "知识提取员"
        self.short_term = ShortTermMemory(window_size=10, name=self.name)
        self.long_term = get_long_term_memory()

        self.model = ChatOpenAI(
            model=MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=MODEL_TEMPERATURE,
        )

    def extract(self, search_results: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """
        从所有搜索结果中提取结构化知识
        """
        if not search_results:
            return []

        # 准备所有搜索结果的文本
        all_content = []
        for sr in search_results:
            angle = sr.get("angle", "未知角度")
            summary = sr.get("summary", "")
            all_content.append(f"## {angle}\n{summary}")

        combined_search_results = "\n\n".join(all_content)

        # RAG 检索历史相关记忆
        if VERBOSE:
            print(f"\n  📖 [Extractor] 检索长期记忆中的相关知识...")
        relevant_past = self.long_term.retrieve(topic, top_k=3)
        past_knowledge = ""
        if relevant_past:
            past_knowledge = "\n\n## 历史相关研究\n" + "\n".join([
                f"- {r['content'][:300]}" for r in relevant_past
            ])
            if VERBOSE:
                print(f"  找到 {len(relevant_past)} 条历史相关记录")

        # LLM 提取
        prompt = f"""你是一位知识提取专家。请从以下研究资料中提取关键知识和事实。

## 研究主题
{topic}

## 研究资料
{combined_search_results}
{past_knowledge}

请提取所有重要的：
1. **关键事实**（具体数据、统计、研究结果）
2. **主要观点**（不同角度的论述）
3. **引用来源**（资料中提到的研究、报告等）
4. **概念定义**（重要的术语和定义）

请按JSON格式输出：
{{
  "facts": [
    {{"fact": "事实内容", "category": "数据/观点/定义/方法", "source": "来源", "confidence": "高/中/低"}},
    ...
  ]
}}
"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            import json
            extracted = json.loads(content)
            facts = extracted.get("facts", [])

            if VERBOSE:
                print(f"  ✅ [Extractor] 提取了 {len(facts)} 条知识")
                for f in facts[:5]:
                    print(f"    - [{f.get('category', '')}] {f.get('fact', '')[:80]}...")

            # 存储到长期记忆
            for fact in facts:
                self.long_term.store_fact(
                    fact=fact.get("fact", ""),
                    source=fact.get("source", ""),
                    category=fact.get("category", "")
                )

            if VERBOSE:
                stats = self.long_term.get_stats()
                print(f"  💾 [长期记忆] 已存储，总计 {stats['total_documents']} 条记录")

            return facts

        except Exception as e:
            print(f"  [警告] 知识提取失败 ({e})，使用关键词匹配提取")
            return self._keyword_extract(combined_search_results)

    def _keyword_extract(self, text: str) -> List[Dict[str, Any]]:
        """关键词匹配提取（当LLM提取失败时使用）— 兼容中英文"""
        facts = []
        lines = text.split("\n")
        # 中英文关键词
        keywords = ["研究", "发现", "数据", "显示", "表明", "提出", "认为",
                    "study", "found", "data", "show", "report", "according",
                    "research", "result", "analysis"]
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                # 去除纯格式行（标题、来源等）
                if line.startswith("#") or line.startswith("来源") or line.startswith("http"):
                    continue
                has_keyword = False
                line_lower = line.lower()
                for kw in keywords:
                    if kw.lower() in line_lower:
                        has_keyword = True
                        break
                if has_keyword:
                    facts.append({
                        "fact": line[:500],
                        "category": "观点",
                        "source": "搜索资料",
                        "confidence": "中"
                    })
        # 如果没有匹配到关键词，取前10个足够长的行
        if not facts:
            for line in lines:
                line = line.strip()
                if len(line) > 40 and not line.startswith("#") and not line.startswith("http"):
                    facts.append({
                        "fact": line[:500],
                        "category": "信息",
                        "source": "搜索资料",
                        "confidence": "低"
                    })
                    if len(facts) >= 10:
                        break
        return facts


# ============ LangGraph 节点函数 ============

def extractor_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph 节点：Extractor
    从搜索结果中提取知识，存入长期记忆
    """
    agent = ExtractorAgent()
    search_results = state.get("search_results", [])
    topic = state.get("user_query", "")

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"  📖 [Extractor] 开始从 {len(search_results)} 组搜索结果中提取知识")
        print(f"{'='*60}")

    facts = agent.extract(search_results, topic)

    if not facts:
        return {
            "extracted_facts": [],
            "current_phase": "reflecting",
            "needs_search": False,
            "messages": [{"role": "Extractor", "content": "未提取到有效知识"}]
        }

    msg = {
        "role": agent.name,
        "content": f"提取了 {len(facts)} 条知识",
        "facts_count": len(facts),
        "timestamp": datetime.now().isoformat()
    }

    return {
        "extracted_facts": facts,
        "current_phase": "reflecting",
        "needs_search": False,
        "messages": [msg]
    }
