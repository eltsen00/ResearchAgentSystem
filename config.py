"""
模型配置、API Key、常量定义
"""

import os

# ============ 模型配置 ============
# LLM API 配置（兼容 OpenAI / DashScope / Ollama / vLLM 等任何 OpenAI 兼容 API）
# 优先读取 LLM_API_KEY / LLM_BASE_URL
LLM_API_KEY = os.getenv("LLM_API_KEY") or ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 主模型：用于推理、规划、生成
MODEL_NAME = os.getenv("LLM_MODEL_NAME") or "qwen-plus"
MODEL_TEMPERATURE = 0.7

# 轻量模型：用于摘要、评分等辅助任务
LIGHT_MODEL_NAME = os.getenv("LLM_LIGHT_MODEL_NAME") or "qwen-turbo"
LIGHT_MODEL_TEMPERATURE = 0.3

# ============ 记忆系统配置 ============
# 短期记忆
SHORT_TERM_WINDOW_SIZE = 10  # 保留最近 N 条消息
SHORT_TERM_SUMMARY_THRESHOLD = 15  # 超过此数量时触发摘要压缩

# 长期记忆（ChromaDB）
CHROMA_PERSIST_DIR = "./chroma_db"  # ChromaDB 持久化目录
CHROMA_COLLECTION_NAME = "research_memory"
EMBEDDING_MODEL = "text-embedding-v1"  # DashScope 嵌入模型
RETRIEVAL_TOP_K = 5  # RAG 检索返回数量
RETRIEVAL_SCORE_THRESHOLD = 0.3  # 相似度阈值
CLEAR_MEMORY_ON_START = False  # 是否在每次启动时清空长期记忆（False=跨会话保留）

# ============ Agent 配置 ============
# ReAct 搜索
MAX_REACT_ITERATIONS = 5  # ReAct 最大迭代次数（充分的 Thought→Action→Observation 循环）

# ToT (Tree of Thoughts)
TOT_NUM_BRANCHES = 6  # 生成的研究角度数量
TOT_TOP_K = 3  # 选择的最佳分支数量

# Reflection 评审
MAX_REFLECTION_ROUNDS = 4  # 最大反思-重新搜索循环次数（展示渐进改进）

# ============ 工具配置 ============
# DuckDuckGo 搜索
WEB_SEARCH_MAX_RESULTS = 5

# 计算器
CALCULATOR_MAX_DIGITS = 10

# ============ 日志配置 ============
VERBOSE = True  # 是否打印详细日志
