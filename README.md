# 🧠 智能研究助手系统（Intelligent Research Assistant）

> 基于 **LangGraph** 的多智能体协作研究助手 — 输入一个主题，自动搜索、分析、评审、撰写研究报告。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

```
5 个 Agent | 3 种推理方法 | 2 种记忆系统 | 3 个工具
```

---

## ✨ 核心特性

- 🌳 **ToT 思维树规划** — 将主题分解为多个研究角度，评分后选最优分支
- 🔍 **ReAct 自主搜索** — LLM 自主决定搜什么、何时调用工具、何时停止
- 🤔 **Reflection 迭代评审** — 四维评估质量，自动发现空白并补充搜索
- 💾 **双重记忆系统** — 短期记忆（滑动窗口 + 摘要压缩）+ 长期记忆（ChromaDB 向量存储）
- 🔧 **工具即用** — Web 搜索 / RAG 检索 / 数学计算
- 🌐 **多 LLM 支持** — 兼容 OpenAI、DashScope、Ollama、vLLM 等任何 OpenAI 格式 API
- 📝 **自动报告生成** — Markdown 格式，含真实来源引用

---

## 🚀 快速开始

### 安装依赖

```bash
pip install langchain langchain-openai langgraph chromadb sentence-transformers ddgs
```

### 设置 API Key

```powershell
# DashScope（通义千问，默认）
$env:LLM_API_KEY = "你的Key"
$env:LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL_NAME = "qwen-plus"
$env:LLM_LIGHT_MODEL_NAME = "qwen-turbo"

# 或者 OpenAI
$env:LLM_API_KEY = "sk-..."
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_MODEL_NAME = "gpt-4o"
$env:LLM_LIGHT_MODEL_NAME = "gpt-4o-mini"

# 或者本地 Ollama
$env:LLM_API_KEY = "ollama"
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL_NAME = "qwen2.5:7b"
$env:LLM_LIGHT_MODEL_NAME = "qwen2.5:1.5b"
```

### 运行

```bash
# 命令行模式
python main.py -t "乒乓球运动员早田希娜（Hina Hayata）球风与打法研究"

# 完整参数
python main.py --topic "乒乓球运动员早田希娜（Hina Hayata）球风与打法研究" --clear-memory --max-rounds 3

# 交互式模式
python main.py

# 查看帮助
python main.py --help
```

---

## 🏗️ 系统架构

```
用户输入主题
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  LangGraph 工作流                │
│                                                 │
│  🌳 Orchestrator   ToT 规划    ──► 研究计划      │
│         │                                       │
│         ▼                                       │
│  🔍 Searcher       ReAct 搜索  ──► 搜索结果      │
│         │                                       │
│         ▼                                       │
│  📖 Extractor      知识提取    ──► 结构化事实     │
│         │                                       │
│         ▼                                       │
│  🤔 Critic         Reflection  ──► 评审/补充     │
│         │         ◄────────── (不通过则重新搜索)  │
│         ▼                                       │
│  ✍️ Writer         报告生成    ──► Markdown 报告  │
└─────────────────────────────────────────────────┘
```

### 五个 Agent

| Agent | 角色 | 推理方法 | 职责 |
|-------|------|----------|------|
| **Orchestrator** | 研究协调者 | **ToT** | 分解主题为多个角度 → 评分 → 选最优 |
| **Searcher** | 信息检索员 | **ReAct** | LLM 自主调用搜索/RAG 工具，收集资料 |
| **Extractor** | 知识提取员 | LLM 提取 | 从搜索结果提取事实，存入 ChromaDB |
| **Critic** | 研究评审员 | **Reflection** | 四维评审，发现空白 → 触发补充搜索 |
| **Writer** | 报告撰写员 | LLM 生成 | 综合所有发现，生成研究报告 |

### 三种推理方法

```
ToT (Tree of Thoughts)          ReAct (Reasoning+Acting)       Reflection
─────────────────────           ───────────────────────        ───────────
探索 6 个角度                   Thought → Action → Obs          评估 → 空白 → 补充
  ↓                               ↓        ↓       ↓              ↑______________↓
评估 打分                         思考   调工具   看结果         (最多 4 轮迭代)
  ↓
选择 最优 3 个
```

### 文件结构

```
ResearchAgentSystem/
├── main.py              # 主入口（CLI + 交互式）
├── config.py            # 配置中心（模型/记忆/Agent）
├── state.py             # LangGraph 全局状态定义
├── graph.py             # 图构建 + 条件路由
├── utils.py             # 日志美化、报告保存
├── agents/
│   ├── orchestrator.py  # ToT 规划 Agent
│   ├── searcher.py      # ReAct 搜索 Agent
│   ├── extractor.py     # 知识提取 Agent
│   ├── critic.py        # Reflection 评审 Agent
│   └── writer.py        # 报告撰写 Agent
├── memory/
│   ├── short_term.py    # 短期记忆（窗口缓冲+摘要压缩）
│   └── long_term.py     # 长期记忆（ChromaDB 向量存储）
├── tools/
│   ├── web_search.py    # DuckDuckGo 搜索
│   ├── rag_retrieve.py  # RAG 检索
│   └── calculator.py    # 安全计算
└── 使用说明.md           # 详细使用文档
```

---

## 📖 命令行参数

```
用法: python main.py [选项]

  -t, --topic TOPIC     研究主题。不指定则进入交互式输入
  --clear-memory        启动前清空长期记忆
  --max-rounds N        最大反思轮次（1-10，默认 4）
  -q, --quiet           静默模式
  -v, --verbose         详细输出（默认）
  -h, --help            帮助信息
```

---

## ⚙️ 配置概览

所有配置集中在 `config.py`，支持环境变量覆盖：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| API Key | `LLM_API_KEY` | — | LLM API 密钥 |
| Base URL | `LLM_BASE_URL` | DashScope | OpenAI 兼容端点 |
| 主模型 | `LLM_MODEL_NAME` | `qwen-plus` | 推理/规划/生成 |
| 轻量模型 | `LLM_LIGHT_MODEL_NAME` | `qwen-turbo` | 摘要/评分 |
| ToT 分支数 | — | 6 | 生成的候选角度 |
| ToT 选择数 | — | 3 | 最终保留的角度 |
| ReAct 迭代 | — | 5 | 最大搜索循环次数 |
| 反思轮次 | — | 4 | 最大重搜索次数 |
| 记忆保留 | `CLEAR_MEMORY_ON_START` | `False` | 跨会话保留记忆 |

---

## 📄 输出产物

系统会在运行目录生成 Markdown 格式的研究报告：

```
research_report_乒乓球运动员早田希娜（Hina Hayata）球风与打法研究_20260724_143052.md
```

报告包含：摘要、引言、研究方法、多角度分析、关键发现、参考文献、研究局限性。

---

## 📝 License

MIT
