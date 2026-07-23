"""
长期记忆模块 — ChromaDB 向量数据库存储
存储历史研究报告、关键发现、反思笔记，支持相似度检索
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD
)


class LongTermMemory:
    """
    长期记忆（向量存储）：
    - 使用 ChromaDB 持久化存储文档片段
    - 支持基于语义相似度的检索
    - 存储：研究报告、关键发现、反思笔记、搜索查询
    - 使用 sentence-transformers 作为嵌入模型（离线、免费）
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME

        # 初始化 ChromaDB 客户端（持久化模式）
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 使用 sentence-transformers 嵌入函数（把文本变成向量）
        try:
            from chromadb.utils import embedding_functions
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception:
            self.embedding_fn = None

        # 获取或创建集合（类似 SQL 中的 TABLE）
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "研究助手长期记忆"}
        )

    def store(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """
        存储一个文档片段到长期记忆。
        Args:
            content: 文档内容
            metadata: 元数据（类型、来源、标签等）
            doc_id: 可选文档ID，不提供则自动生成
        Returns:
            文档ID
        """
        if metadata is None:
            metadata = {}

        # 确保元数据至少有 type 和 timestamp
        metadata.setdefault("type", "general")
        metadata.setdefault("timestamp", datetime.now().isoformat())
        metadata.setdefault("char_count", len(content))

        # 自动生成唯一 ID（精确到微秒）
        doc_id = doc_id or f"mem_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        self.collection.add(
            documents=[content], # 原始文本
            metadatas=[metadata], # 标签
            ids=[doc_id] # 唯一标识
        )

        return doc_id

    def retrieve(self, query: str, top_k: int = None, score_threshold: float = None, filter_type: str = None) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档片段。
        Args:
            query: 查询文本
            top_k: 返回数量
            score_threshold: 相似度阈值（低于此值的结果被过滤）
            filter_type: 按类型过滤（如 "fact", "reflection", "report"）
        Returns:
            [{id, content, metadata, score}, ...]
        """
        top_k = top_k or RETRIEVAL_TOP_K
        score_threshold = score_threshold or RETRIEVAL_SCORE_THRESHOLD

        results = self.collection.query(
            query_texts=[query], # 查询文本，自动转为向量，支持批量查询
            n_results=top_k,
            where={"type": filter_type} if filter_type else None,  # 可选类型过滤
            include=["documents", "metadatas", "distances"] # 返回结果中包含的字段
        )

        # 把 distance 转换为 similarity（距离越小 → 相似度越高）
        formatted = []
        if results["documents"] and results["documents"][0]: # 批量查询返回的是二维列表，取第一个查询结果
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1.0 / (1.0 + distance)

                # 过滤低相似度结果
                if similarity >= score_threshold:
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": meta,
                        "score": round(similarity, 4)
                    })

        return formatted

    def store_fact(self, fact: str, source: str = "", category: str = "") -> str:
        """存储一条提取的事实"""
        return self.store(
            content=fact,
            metadata={
                "type": "fact",
                "source": source,
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
        )

    def store_reflection(self, reflection: str, topic: str = "") -> str:
        """存储一条反思笔记"""
        return self.store(
            content=reflection,
            metadata={
                "type": "reflection",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }
        )

    def store_report(self, report: str, topic: str = "") -> str:
        """存储一份研究报告"""
        return self.store(
            content=report,
            metadata={
                "type": "report",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }
        )

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近存储的文档"""
        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            formatted = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    formatted.append({
                        "id": results["ids"][i],
                        "content": doc,
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })
            return formatted
        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_model": "all-MiniLM-L6-v2" if self.embedding_fn else "chromadb-default"
        }

    def clear(self) -> None:
        """清空所有长期记忆"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )


_long_term_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """获取全局唯一的长期记忆实例（惰性初始化，跨会话保留数据）"""
    global _long_term_instance
    if _long_term_instance is None:
        init_long_term_memory(clear_first=False)
    return _long_term_instance


def init_long_term_memory(clear_first: bool = True) -> LongTermMemory:
    """
    初始化长期记忆单例。由 graph.py 在构建图时调用。
    Args:
        clear_first: 是否先清空上次运行的遗留数据
    Returns:
        全局唯一的 LongTermMemory 实例
    """
    global _long_term_instance
    _long_term_instance = LongTermMemory()
    if clear_first:
        _long_term_instance.clear()
    return _long_term_instance
