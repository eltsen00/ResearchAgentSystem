"""
RAG 检索工具 — 从长期记忆（ChromaDB）中检索相关文档
"""
from langchain_core.tools import tool
from config import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD
from memory.long_term import get_long_term_memory


@tool
def rag_retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
    """
    从长期记忆库中检索与查询相关的历史研究资料。
    当需要查找之前研究过的内容、历史发现、反思笔记时使用此工具。
    此工具搜索的是已存储的知识库，而非互联网。

    Args:
        query: 检索查询
        top_k: 返回结果数量（默认5）
    Returns:
        格式化的检索结果
    """
    long_term_memory = get_long_term_memory()

    try:
        results = long_term_memory.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=RETRIEVAL_SCORE_THRESHOLD
        )

        if not results:
            return f"在长期记忆中未找到与 '{query}' 相关的资料。这可能是新课题，需要通过网络搜索获取信息。"

        formatted = [f"## 长期记忆检索结果: '{query}'\n"]
        formatted.append(f"找到 {len(results)} 条相关记录：\n")

        for i, r in enumerate(results, 1):
            content = r["content"]
            if len(content) > 400:
                content = content[:400] + "..."
            meta = r.get("metadata", {})
            doc_type = meta.get("type", "未知")
            timestamp = meta.get("timestamp", "")[:10]
            score = r.get("score", 0)
            formatted.append(
                f"### 结果 {i} (类型: {doc_type}, 日期: {timestamp}, 相关度: {score:.2f})\n"
                f"> {content}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"检索时出错: {str(e)}"
