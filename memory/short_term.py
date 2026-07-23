"""
短期记忆模块 — 会话内消息缓冲 + 窗口化 + 摘要压缩
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from config import (
    MODEL_NAME, LLM_BASE_URL, LLM_API_KEY,
    SHORT_TERM_WINDOW_SIZE, SHORT_TERM_SUMMARY_THRESHOLD,
    LIGHT_MODEL_NAME, LIGHT_MODEL_TEMPERATURE
)


class ShortTermMemory:
    """
    短期记忆（工作记忆）：
    - 维护最近 N 条消息的滑动窗口
    - 超出窗口的消息由 LLM 摘要压缩
    - 保持记忆预算可控，同时不丢失关键信息
    """

    def __init__(self, window_size: int = None, name: str = "Agent"):
        self.window_size = window_size or SHORT_TERM_WINDOW_SIZE
        self.name = name
        self.buffer: List[Dict[str, Any]] = []
        self.summary: str = ""  # 被挤出窗口的消息摘要

        # 轻量模型用于摘要
        self.summary_model = ChatOpenAI(
            model=LIGHT_MODEL_NAME,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=LIGHT_MODEL_TEMPERATURE,
        )

    def add(self, message: Dict[str, Any]) -> None:
        """添加一条消息到缓冲区"""
        self.buffer.append(message)

        # 如果缓冲区超出摘要阈值，触发压缩
        if len(self.buffer) > SHORT_TERM_SUMMARY_THRESHOLD:
            self._compress()

    def get_context(self) -> str:
        """获取格式化的上下文文本（摘要 + 最近消息）"""
        parts = []

        if self.summary:
            parts.append(f"[历史摘要]\n{self.summary}")

        if self.buffer:
            recent = self.buffer[-self.window_size:]  # 只取最近 N 条
            lines = []
            for m in recent:
                role = m.get("role", "unknown")
                content = m.get("content", str(m))
                # 截断过长的内容
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"[{role}]: {content}")
            parts.append("[最近对话]\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "(无历史记录)"

    def get_recent(self, n: int = None) -> List[Dict[str, Any]]:
        """获取最近 N 条消息"""
        n = n or self.window_size
        return self.buffer[-n:]

    def clear(self) -> None:
        """清空缓冲区（保留摘要）"""
        self.buffer = []

    def _compress(self) -> None:
        """
        将被挤出窗口的消息压缩为摘要。
        保留最近 window_size 条，将其余的 LLM 总结后存入 self.summary。
        """
        if len(self.buffer) <= self.window_size:
            return

        # 取出将被压缩的旧消息
        overflow = self.buffer[:-self.window_size]
        self.buffer = self.buffer[-self.window_size:]

        # 构建压缩提示
        old_context = "\n".join([
            f"[{m.get('role', 'unknown')}]: {m.get('content', str(m))[:300]}"
            for m in overflow
        ])

        try:
            response = self.summary_model.invoke(
                f"请用2-3句话总结以下对话的关键信息，保留重要的事实、数据和结论：\n\n{old_context}\n\n关键信息摘要："
            )
            new_summary = response.content.strip()
            # 合并到已有摘要
            if self.summary:
                self.summary = self.summary + " " + new_summary
            else:
                self.summary = new_summary
        except Exception:
            # 摘要失败时保留旧消息的最后几条
            fallback = overflow[-2:]
            self.buffer = fallback + self.buffer
