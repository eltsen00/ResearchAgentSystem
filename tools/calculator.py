"""
计算工具 — 安全地计算数学表达式
"""
import math
from langchain_core.tools import tool


# 安全的数学命名空间
SAFE_MATH_NAMESPACE = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
    "ceil": math.ceil, "floor": math.floor,
    "mean": lambda x: sum(x) / len(x) if x else 0,
    "median": lambda x: sorted(x)[len(x) // 2] if x else 0,
    "std": lambda x: (sum((i - sum(x)/len(x))**2 for i in x) / len(x)) ** 0.5 if x else 0,
}


@tool
def calculate(expression: str) -> str:
    """
    安全地计算数学表达式。支持基本算术、统计函数和科学计算。
    当需要计算具体数值、统计量、百分比或进行比较分析时使用此工具。

    支持的运算:
    - 基本算术: +, -, *, /, //, %, **
    - 数学函数: sqrt, log, log10, log2, exp, pow
    - 三角函数: sin, cos, tan
    - 常量: pi, e
    - 统计函数: mean([...]), median([...]), std([...])
    - 其他: abs, round, min, max, sum, ceil, floor

    Args:
        expression: 数学表达式字符串
    Returns:
        计算结果
    """
    try:
        # 安全检查：禁止危险操作
        forbidden = ["__", "import", "exec", "eval", "compile", "open", "write","delete", "system", "subprocess", "os.", "sys.", "globals", "locals"]
        expr_lower = expression.lower()
        for keyword in forbidden:
            if keyword in expr_lower:
                return f"⚠️ 表达式包含不允许的操作: '{keyword}'。请使用安全的数学表达式。"

        # 使用受限命名空间求值
        result = eval(expression, {"__builtins__": {}}, SAFE_MATH_NAMESPACE)

        # 格式化结果
        if isinstance(result, float):
            # 避免浮点精度问题
            if abs(result) < 1e-10:
                result = 0.0
            result = round(result, 6)
        elif isinstance(result, complex):
            return f"计算结果（复数）: {result}"

        return f"📊 计算结果: {expression} = {result}"

    except SyntaxError:
        return f"❌ 表达式语法错误: '{expression}'。请检查括号配对和运算符。"
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return f"❌ 计算错误: {type(e).__name__}: {e}"
    except Exception as e:
        return f"❌ 无法计算 '{expression}': {type(e).__name__}: {e}"
