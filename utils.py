"""
辅助工具函数 — 日志、格式化输出、控制台美化
注意：Windows 编码修复在 main.py 中统一处理，此处不再重复
"""
import sys
import io
import time
from datetime import datetime
from typing import Dict, Any, List
from state import ResearchState


# ANSI 颜色码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_BLUE = "\033[104m"
    BG_GREEN = "\033[102m"


def print_banner():
    """打印系统启动横幅"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║            Intelligent Research Assistant                    ║
║       Multi-Agent System powered by LangGraph                ║
║                                                              ║
║   5 Agents | 3 Reasoning Methods | 2 Memory Types | 3 Tools  ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(banner)

def print_end_banner():
    """打印系统结束横幅"""
    end_banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════╗
║                    🎉 研究助手执行完毕                    ║
║                                                          ║
║  🌳 ToT (Tree of Thoughts) — Orchestrator 规划           ║
║  🔍 ReAct (Reasoning+Acting) — Searcher 搜索             ║
║  🤔 Reflection — Critic 评审                             ║
║  💾 短期记忆 + 长期记忆 (ChromaDB)                        ║
║  🔧 web_search + rag_retrieve + calculate 工具           ║
╚══════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(end_banner)

def print_agent_header(agent_name: str, role: str, phase: str = ""):
    """打印 Agent 执行标题"""
    emoji_map = {
        "Orchestrator": "🌳",
        "Searcher": "🔍",
        "Extractor": "📖",
        "Critic": "🤔",
        "Writer": "✍️",
    }
    emoji = emoji_map.get(agent_name, "🤖")
    phase_str = f" [{phase}]" if phase else ""
    print(f"\n{Colors.BOLD}{Colors.BLUE}  {emoji} {agent_name} ({role}){phase_str}{Colors.RESET}")
    print(f"  {'─'*50}")


def print_section(title: str):
    """打印分隔区域"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.RESET}")


def print_info(label: str, value: str):
    """打印键值信息"""
    print(f"  {Colors.DIM}{label}:{Colors.RESET} {value}")


def print_success(msg: str):
    """打印成功信息"""
    print(f"  {Colors.GREEN}✅ {msg}{Colors.RESET}")


def print_warning(msg: str):
    """打印警告信息"""
    print(f"  {Colors.YELLOW}⚠️ {msg}{Colors.RESET}")


def print_error(msg: str):
    """打印错误信息"""
    print(f"  {Colors.RED}❌ {msg}{Colors.RESET}")


def print_state_summary(state: ResearchState):
    """打印当前状态摘要"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}  📊 状态摘要{Colors.RESET}")
    print_info("阶段", state.get("current_phase", "?"))
    print_info("研究角度", str(len(state.get("research_plan", []))))
    print_info("搜索组数", str(len(state.get("search_results", []))))
    print_info("提取知识", str(len(state.get("extracted_facts", []))))
    print_info("反思轮次", str(state.get("reflection_round", 0)))
    critique = state.get("critique", {})
    if critique:
        print_info("评审分数", f"{critique.get('overall_score', 'N/A')}/10")
    print_info("完成", "是" if state.get("research_complete", False) else "否")


def print_react_trace(trace: List[Dict[str, Any]]):
    """格式化打印 ReAct 执行轨迹"""
    print(f"\n  {Colors.BOLD}🔄 ReAct 搜索轨迹:{Colors.RESET}")
    for step in trace:
        iteration = step.get("iteration", "?")
        thought = step.get("thought", "")
        action = step.get("action", {})
        observation = step.get("observation", "")

        print(f"  {Colors.CYAN}── 迭代 {iteration} ──{Colors.RESET}")
        if thought:
            print(f"  {Colors.YELLOW}💭 思考:{Colors.RESET} {thought}")
        if action:
            tool_name = action.get("tool", "?")
            print(f"  {Colors.MAGENTA}🔧 行动:{Colors.RESET} 调用 {tool_name}")
        if observation:
            obs_preview = observation[:200].replace("\n", " ")
            print(f"  {Colors.GREEN}👁️ 观察:{Colors.RESET} {obs_preview}...")


def print_final_report(report: str):
    """美化打印最终报告"""
    print_section("📄 最终研究报告")
    print(report)
    print(f"\n{Colors.BOLD}{Colors.CYAN}  📊 报告统计: {len(report)} 字符, ~{len(report.split())} 词{Colors.RESET}\n")


def save_report_to_file(report: str, topic: str) -> str:
    """保存报告到文件"""
    # 清理文件名
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:30]
    filename = f"research_report_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = f"./{filename}"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath


def elapsed_time(start: float) -> str:
    """返回格式化的耗时"""
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}秒"
    else:
        return f"{elapsed/60:.1f}分钟"
