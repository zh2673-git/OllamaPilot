"""
Python Solver Skill

提供 Python 代码执行、脚本运行、数据分析等功能

主要工具:
    - python_exec: 执行 Python 代码片段
    - python_script: 执行 Python 脚本文件
    - python_notebook: 交互式代码执行
"""

import io
import sys
import traceback
import contextlib
from typing import Optional, Dict, Any, List
from pathlib import Path
from langchain_core.tools import tool, BaseTool

# 导入 Skill 基类
from base_agent.skill import Skill


# 危险操作黑名单
DANGEROUS_MODULES = [
    'os.system', 'os.popen', 'subprocess', 'sys.exit',
    '__import__', 'eval', 'exec', 'compile',
]

# 预导入的常用库
DEFAULT_IMPORTS = """
import os
import sys
import json
import csv
import re
import math
import random
import statistics
import datetime
import time
import pathlib
import shutil
import itertools
import collections
import functools
import typing
from pathlib import Path
"""


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    检查代码安全性
    
    Returns:
        (是否安全, 错误信息)
    """
    code_lower = code.lower()
    
    # 检查危险模块
    for dangerous in DANGEROUS_MODULES:
        if dangerous in code_lower:
            return False, f"🚫 安全拦截: 检测到危险操作 '{dangerous}'"
    
    # 检查文件系统危险操作
    dangerous_patterns = [
        'rm -rf', 'del /', 'format(', 'mkfs',
        'shutil.rmtree', 'os.rmdir', 'os.remove',
    ]
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            return False, f"🚫 安全拦截: 检测到危险文件操作 '{pattern}'"
    
    return True, ""


@tool
def python_exec(code: str, timeout: int = 30) -> str:
    """
    执行 Python 代码片段
    
    适合快速计算、简单数据处理、文本操作等任务。
    
    Args:
        code: Python 代码（单行或多行）
        timeout: 执行超时时间（秒），默认 30 秒
        
    Returns:
        代码执行结果（stdout + stderr + 返回值）
        
    Examples:
        >>> python_exec("sum(range(1, 101))")
        >>> python_exec("import math; math.sqrt(16)")
        >>> python_exec('''
        ... text = "Hello World"
        ... print(text.upper())
        ... ''')
    """
    # 安全检查
    is_safe, error_msg = check_code_safety(code)
    if not is_safe:
        return error_msg
    
    # 创建执行环境
    env = {}
    
    # 执行预导入
    try:
        exec(DEFAULT_IMPORTS, env)
    except Exception as e:
        return f"❌ 环境初始化失败: {e}"
    
    # 捕获输出
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    result_output = []
    result_output.append(f">>> Python 代码执行")
    result_output.append(f"{'='*50}")
    result_output.append(f"代码:\n{code}")
    result_output.append(f"{'='*50}")
    
    try:
        # 使用 contextlib 重定向输出
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            # 如果是表达式，尝试 eval
            try:
                result = eval(code, env)
                if result is not None:
                    print(f"[返回值]: {repr(result)}")
            except SyntaxError:
                # 不是表达式，使用 exec
                exec(code, env)
        
        # 获取输出
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            result_output.append(f"\n📤 标准输出:\n{stdout_content}")
        
        if stderr_content:
            result_output.append(f"\n⚠️ 标准错误:\n{stderr_content}")
        
        result_output.append(f"\n✅ 执行成功")
        
    except Exception as e:
        result_output.append(f"\n❌ 执行失败:")
        result_output.append(f"错误类型: {type(e).__name__}")
        result_output.append(f"错误信息: {str(e)}")
        result_output.append(f"\n堆栈跟踪:\n{traceback.format_exc()}")
    
    return '\n'.join(result_output)


@tool
def python_script(script_path: str, args: Optional[str] = None, timeout: int = 60) -> str:
    """
    执行 Python 脚本文件
    
    适合执行复杂的、多步骤的 Python 脚本。
    
    Args:
        script_path: Python 脚本文件路径
        args: 命令行参数（可选）
        timeout: 执行超时时间（秒），默认 60 秒
        
    Returns:
        脚本执行结果
        
    Examples:
        >>> python_script("analysis.py")
        >>> python_script("process_data.py", args="--input data.csv --output result.csv")
    """
    import subprocess
    
    path = Path(script_path)
    if not path.exists():
        return f"❌ 脚本文件不存在: {script_path}"
    
    if not path.suffix == '.py':
        return f"⚠️ 警告: 文件 '{script_path}' 可能不是 Python 脚本"
    
    # 构建命令
    cmd = [sys.executable, str(path)]
    if args:
        cmd.extend(args.split())
    
    result_output = []
    result_output.append(f">>> Python 脚本执行")
    result_output.append(f"{'='*50}")
    result_output.append(f"脚本: {script_path}")
    result_output.append(f"参数: {args or '无'}")
    result_output.append(f"{'='*50}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.stdout:
            result_output.append(f"\n📤 标准输出:\n{result.stdout}")
        
        if result.stderr:
            result_output.append(f"\n⚠️ 标准错误:\n{result.stderr}")
        
        result_output.append(f"\n[退出码: {result.returncode}]")
        
        if result.returncode == 0:
            result_output.append("✅ 执行成功")
        else:
            result_output.append("❌ 执行失败")
        
    except subprocess.TimeoutExpired:
        result_output.append(f"\n⏱️ 执行超时（>{timeout}秒）")
    except Exception as e:
        result_output.append(f"\n❌ 执行出错: {e}")
    
    return '\n'.join(result_output)


@tool
def python_notebook(cells: list, timeout: int = 30) -> str:
    """
    交互式 Python 代码执行（类似 Jupyter Notebook）
    
    按顺序执行多个代码单元格，共享执行环境。
    适合数据分析、逐步探索等场景。
    
    Args:
        cells: 代码单元格列表，每个元素是一个代码字符串
        timeout: 每个单元格的超时时间（秒），默认 30 秒
        
    Returns:
        所有单元格的执行结果
        
    Examples:
        >>> python_notebook([
        ...     "import pandas as pd",
        ...     "df = pd.read_csv('data.csv')",
        ...     "print(df.head())",
        ...     "print(df.describe())"
        ... ])
    """
    if not cells:
        return "⚠️ 没有提供代码单元格"
    
    # 创建共享执行环境
    env = {}
    
    # 执行预导入
    try:
        exec(DEFAULT_IMPORTS, env)
    except Exception as e:
        return f"❌ 环境初始化失败: {e}"
    
    result_output = []
    result_output.append(f">>> Python Notebook 执行")
    result_output.append(f"{'='*50}")
    result_output.append(f"单元格数量: {len(cells)}")
    result_output.append(f"{'='*50}")
    
    for i, cell in enumerate(cells, 1):
        result_output.append(f"\n{'─'*50}")
        result_output.append(f"单元格 [{i}/{len(cells)}]")
        result_output.append(f"{'─'*50}")
        result_output.append(f"代码:\n{cell}")
        result_output.append(f"{'─'*50}")
        
        # 安全检查
        is_safe, error_msg = check_code_safety(cell)
        if not is_safe:
            result_output.append(f"\n{error_msg}")
            result_output.append(f"❌ 单元格 {i} 执行被拦截")
            continue
        
        # 捕获输出
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # 尝试 eval，失败则用 exec
                try:
                    result = eval(cell, env)
                    if result is not None:
                        print(f"[返回值]: {repr(result)}")
                except SyntaxError:
                    exec(cell, env)
            
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            if stdout_content:
                result_output.append(f"\n📤 输出:\n{stdout_content}")
            
            if stderr_content:
                result_output.append(f"\n⚠️ 错误:\n{stderr_content}")
            
            result_output.append(f"✅ 单元格 {i} 执行成功")
            
        except Exception as e:
            result_output.append(f"\n❌ 单元格 {i} 执行失败:")
            result_output.append(f"错误类型: {type(e).__name__}")
            result_output.append(f"错误信息: {str(e)}")
            result_output.append(f"\n堆栈跟踪:\n{traceback.format_exc()}")
    
    result_output.append(f"\n{'='*50}")
    result_output.append(f"Notebook 执行完成")
    
    return '\n'.join(result_output)


@tool
def python_install(package: str) -> str:
    """
    安装 Python 包
    
    Args:
        package: 包名（支持 pip 的所有格式）
        
    Returns:
        安装结果
        
    Examples:
        >>> python_install("requests")
        >>> python_install("pandas==2.0.0")
        >>> python_install("-r requirements.txt")
    """
    import subprocess
    
    result_output = []
    result_output.append(f">>> 安装 Python 包")
    result_output.append(f"{'='*50}")
    result_output.append(f"包: {package}")
    result_output.append(f"{'='*50}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.stdout:
            result_output.append(f"\n📤 输出:\n{result.stdout}")
        
        if result.stderr:
            result_output.append(f"\n⚠️ 错误:\n{result.stderr}")
        
        result_output.append(f"\n[退出码: {result.returncode}]")
        
        if result.returncode == 0:
            result_output.append("✅ 安装成功")
        else:
            result_output.append("❌ 安装失败")
        
    except subprocess.TimeoutExpired:
        result_output.append("\n⏱️ 安装超时（>120秒）")
    except Exception as e:
        result_output.append(f"\n❌ 安装出错: {e}")
    
    return '\n'.join(result_output)


class PythonSolverSkill(Skill):
    """
    Python Solver Skill
    
    提供 Python 代码执行能力，帮助解决各类编程和数据处理问题。
    """
    
    name = "python-solver"
    description = "Python 问题解决工具，提供代码执行、数据分析、文本处理等功能"
    tags = ["python", "脚本", "数据处理", "计算", "自动化"]
    version = "1.0.0"
    author = "BaseAgent Team"
    
    def get_tools(self) -> List[BaseTool]:
        """
        返回 Skill 提供的工具列表
        
        Returns:
            List[BaseTool]: 工具列表
        """
        return [
            python_exec,
            python_script,
            python_notebook,
            python_install,
        ]
