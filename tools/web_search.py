"""
网络搜索工具 — 使用 DuckDuckGo 免费 API（无需 API Key）
"""
from langchain_core.tools import tool
from config import WEB_SEARCH_MAX_RESULTS

@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索信息。当需要查找实时信息、最新数据、
    研究资料或任何你不确定的事实信息时使用此工具。

    Args:
        query: 搜索查询字符串（中英文均可）
    Returns:
        格式化的搜索结果，包含标题、摘要和来源URL
    """
    try:
        # 导入 DuckDuckGo 搜索库，使用 DuckDuckGo 免费 API 进行搜索
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))

        if not results:
            return f"搜索 '{query}' 未找到相关结果。请尝试调整搜索词。"

        formatted = [f"## 搜索结果: '{query}'\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            body = r.get("body", "无摘要")
            href = r.get("href", "")
            # 截断过长的摘要
            if len(body) > 300:
                body = body[:300] + "..."
            formatted.append(f"**{i}. {title}**\n> {body}\n来源: {href}\n")

        return "\n".join(formatted)

    except ImportError:
        return f"搜索不可用: DuckDuckGo 库未安装"
    except Exception as e:
        return f"搜索失败 ({type(e).__name__}): 请尝试更换搜索词或缩短查询"
