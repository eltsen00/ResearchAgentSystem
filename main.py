"""
主入口 — 智能研究助手系统
用法: python main.py -t "研究主题"
      python main.py --topic "研究主题" --clear-memory
      python main.py --help    查看完整帮助
"""
import sys
import io
import os
import time
import argparse

# 修复 Windows GBK 编码问题，强制使用 UTF-8（仅在交互终端中有效）
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except (ValueError, AttributeError, OSError):
        pass  # 非交互终端（如管道、后台运行）可能不支持，静默跳过

# 设置 SSL 证书路径，确保 requests 库在 Windows 上正常工作
if "SSL_CERT_FILE" in os.environ:
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass
from datetime import datetime

# 延迟导入：graph/state 依赖 langgraph，仅在 run_research() 中惰性加载，
# 确保 --help / -h 等纯 CLI 操作不需安装 langgraph 即可运行
from utils import (
    print_banner, print_end_banner, print_section, print_info, print_success,
    print_warning, print_error, print_state_summary,
    print_final_report, save_report_to_file, elapsed_time,
    Colors
)
from config import VERBOSE


def run_research(topic: str, max_rounds: int = 2):
    """
    运行研究流程。

    Args:
        topic: 研究主题
        max_rounds: 最大反思轮次（可在 config.py 中覆盖）

    Returns:
        最终状态（包含 final_report），或 None（出错时）
    """
    # 惰性导入：确保 --help 等纯 CLI 操作不需 langgraph
    from graph import build_research_graph
    from state import ResearchState
    # 构建图
    if VERBOSE:
        print_section("🔧 初始化系统")
    graph = build_research_graph()

    # 设置配置（用于检查点）（checkpointer=memory_saver）
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    # 初始化状态
    initial_state: ResearchState = {
        "user_query": topic,
        "messages": [{
            "role": "System",
            "content": f"用户查询: {topic}",
            "timestamp": datetime.now().isoformat()
        }],
        "research_plan": [],
        "search_results": [],
        "extracted_facts": [],
        "critique": {},
        "final_report": "",
        "current_phase": "orchestrating",
        "needs_search": True,
        "needs_revision": False,
        "reflection_round": 0,
        "research_complete": False,
        "current_angle_index": 0,
        "gap_queries": [],
        "previous_score": None,
        "error": None
    }

    if VERBOSE:
        print_section(f"🚀 开始研究")
        print_info("主题", topic)
        print_info("时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 运行图
    if VERBOSE:
        print_section("🔄 执行流程")

    step_count = 0

    try:
        for output in graph.stream(initial_state, config):
            # output 只包含当前这一步的变化，不是完整 state
            step_count += 1

            # 提取节点名称和输出
            node_name = list(output.keys())[0] if output else "unknown"
            node_output = output.get(node_name, {}) if output else {}

            if VERBOSE:
                print(f"\n  {Colors.BOLD}── 步骤 {step_count}: {node_name} ──{Colors.RESET}")

                if node_output.get("current_phase"):
                    print_info("→ 阶段", node_output["current_phase"])
                if node_output.get("research_plan"):
                    print_info("研究角度", f"{len(node_output['research_plan'])} 个")
                if node_output.get("search_results"):
                    print_info("搜索结果", f"{len(node_output['search_results'])} 组")
                if node_output.get("extracted_facts"):
                    print_info("提取知识", f"{len(node_output['extracted_facts'])} 条")
                if node_output.get("critique"):
                    c = node_output["critique"]
                    print_info("评审分数", f"{c.get('overall_score', 'N/A')}/10")
                    print_info("评审判定", c.get('verdict', 'N/A'))

            messages = node_output.get("messages", [])
            for msg in messages:
                content = msg.get("content", str(msg))
                role = msg.get("role", "?")
                if len(content) > 150:
                    content = content[:150] + "..."
                if VERBOSE:
                    print(f"  [{role}]: {content}")

        # stream 结束后，获取完整累积状态
        final_state = graph.get_state(config).values

    except KeyboardInterrupt:
        print_warning("用户中断研究")
        return None
    except Exception as e:
        print_error(f"研究过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    if VERBOSE:
        print_success(f"研究流程完成，共 {step_count} 步")

    return final_state


def build_cli_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="智能研究助手系统 — 基于 LangGraph 的多智能体协作研究助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -t "量子计算的最新进展"
  python main.py --topic "人工智能在教育领域的应用" --clear-memory
  python main.py                                              # 交互式输入主题
  python main.py -t "新能源汽车市场趋势" --max-rounds 3 -q     # 静默模式，最多 3 轮反思
        """
    )

    parser.add_argument(
        "-t", "--topic",
        type=str,
        default=None,
        help="研究主题（中英文均可）。不指定则进入交互式输入模式"
    )

    parser.add_argument(
        "--clear-memory",
        action="store_true",
        default=False,
        help="启动前清空长期记忆（ChromaDB），从零开始研究"
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        metavar="N",
        help="最大反思轮次（覆盖 config.py 中的 MAX_REFLECTION_ROUNDS，范围 1-10）"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="静默模式，减少控制台输出（等价于 VERBOSE=False）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=None,
        help="详细输出模式（默认开启，等同于 config.py 中 VERBOSE=True）"
    )

    return parser


def main():
    """主函数 — 支持命令行参数和交互式两种运行模式"""
    parser = build_cli_parser()

    # 解析参数，遇未知参数时友好报错而非静默忽略
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse 在 --help 时正常退出（code=0），其他均为参数错误
        if e.code != 0:
            print(f"\n{Colors.RED}命令行参数错误{Colors.RESET}")
            print(f"  {Colors.DIM}请使用 {Colors.BOLD}python main.py --help{Colors.RESET}{Colors.DIM} 查看可用参数{Colors.RESET}\n")
        sys.exit(e.code)

    # ---- 处理命令行参数 ----

    # 静默/详细模式
    if args.quiet:
        import config
        config.VERBOSE = False
    elif args.verbose:
        import config
        config.VERBOSE = True

    # 清空长期记忆
    if args.clear_memory:
        import config
        config.CLEAR_MEMORY_ON_START = True

    # 最大反思轮次
    if args.max_rounds is not None:
        if 1 <= args.max_rounds <= 10:
            import config
            config.MAX_REFLECTION_ROUNDS = args.max_rounds
        else:
            print_error("--max-rounds 取值范围为 1-10")
            return

    # 显示横幅
    print_banner()

    # 获取研究主题
    if args.topic:
        topic = args.topic
    else:
        print(f"\n{Colors.BOLD}请输入研究主题（或按 Enter 使用默认示例）:{Colors.RESET}")
        topic = input("> ").strip()
        if not topic:
            topic = "乒乓球运动员早田希娜（Hina Hayata）球风与打法分析"

    if not topic:
        print_error("未输入研究主题，退出")
        return

    # 运行研究
    start_time = time.time()
    final_state = run_research(topic)

    if final_state is None:
        print_error("研究未完成")
        return

    # 显示最终报告
    final_report = final_state.get("final_report", "")
    if final_report:
        print_final_report(final_report)

        # 保存报告
        filepath = save_report_to_file(final_report, topic)
        print_success(f"报告已保存到: {filepath}")
    else:
        print_warning("未生成最终报告")

    # 显示状态摘要
    print_state_summary(final_state)

    # 总结
    print_section("✅ 研究完成")
    print_success(f"耗时: {elapsed_time(start_time)}")
    print_success(f"报告长度: {len(final_report)} 字符")

    print_end_banner()


if __name__ == "__main__":
    main()
