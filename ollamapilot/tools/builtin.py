"""
内置工具集合

提供基础的文件系统、Shell 执行、网络搜索等常用工具。
所有工具均使用 LangChain 的 @tool 装饰器定义。
"""

import os
import re
import io
import sys
import json
import subprocess
import contextlib
import traceback
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool


# =============================================================================
# 文件系统工具
# =============================================================================

@tool
def read_file(file_path: str, limit: int = 1000) -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径（相对或绝对）
        limit: 最大读取行数，默认1000行
        
    Returns:
        文件内容
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"❌ 文件不存在: {file_path}"
        
        if not path.is_file():
            return f"❌ 不是文件: {file_path}"
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= limit:
                    lines.append(f"\n... (已截断，共读取 {limit} 行)")
                    break
                lines.append(line.rstrip())
            
            content = '\n'.join(lines)
            return f"📄 文件: {file_path}\n{'='*50}\n{content}"
            
    except Exception as e:
        return f"❌ 读取失败: {e}"


@tool
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    写入文件内容
    
    Args:
        file_path: 文件路径
        content: 文件内容
        append: 是否追加模式，默认False（覆盖）
        
    Returns:
        操作结果
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        
        action = "追加" if append else "写入"
        return f"✅ {action}成功: {file_path}\n   大小: {len(content)} 字符"
        
    except Exception as e:
        return f"❌ 写入失败: {e}"


@tool
def list_directory(dir_path: str = ".", recursive: bool = False) -> str:
    """
    列出目录内容
    
    Args:
        dir_path: 目录路径，默认当前目录
        recursive: 是否递归列出子目录，默认False
        
    Returns:
        目录内容列表
    """
    try:
        path = Path(dir_path)
        if not path.exists():
            return f"❌ 目录不存在: {dir_path}"
        
        if not path.is_dir():
            return f"❌ 不是目录: {dir_path}"
        
        def format_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes/1024:.1f}KB"
            else:
                return f"{size_bytes/(1024*1024):.1f}MB"
        
        lines = [f"📁 目录: {path.absolute()}", "="*50]
        
        if recursive:
            for item in path.rglob("*"):
                rel_path = item.relative_to(path)
                if item.is_dir():
                    lines.append(f"📂 {rel_path}/")
                else:
                    size = item.stat().st_size
                    lines.append(f"📄 {rel_path} ({format_size(size)})")
        else:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item in items:
                if item.is_dir():
                    lines.append(f"📂 {item.name}/")
                else:
                    size = item.stat().st_size
                    lines.append(f"📄 {item.name} ({format_size(size)})")
        
        return '\n'.join(lines)
        
    except Exception as e:
        return f"❌ 列出目录失败: {e}"


@tool
def search_files(pattern: str, dir_path: str = ".", file_ext: Optional[str] = None) -> str:
    """
    搜索文件
    
    Args:
        pattern: 搜索模式（文件名包含的字符串）
        dir_path: 搜索目录，默认当前目录
        file_ext: 文件扩展名过滤（如 .py, .md），可选
        
    Returns:
        搜索结果
    """
    try:
        path = Path(dir_path)
        if not path.exists():
            return f"❌ 目录不存在: {dir_path}"
        
        matches = []
        for item in path.rglob("*"):
            if item.is_file():
                if pattern.lower() in item.name.lower():
                    if file_ext and not item.name.endswith(file_ext):
                        continue
                    matches.append(item)
        
        if not matches:
            return f"🔍 未找到匹配的文件: '{pattern}'"
        
        def format_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes}B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes/1024:.1f}KB"
            else:
                return f"{size_bytes/(1024*1024):.1f}MB"
        
        lines = [f"🔍 搜索结果: '{pattern}'", f"📁 目录: {path.absolute()}", "="*50]
        for match in matches[:50]:
            rel_path = match.relative_to(path)
            size = match.stat().st_size
            lines.append(f"📄 {rel_path} ({format_size(size)})")
        
        if len(matches) > 50:
            lines.append(f"\n... 还有 {len(matches) - 50} 个结果")
        
        lines.append(f"\n共找到 {len(matches)} 个文件")
        return '\n'.join(lines)
        
    except Exception as e:
        return f"❌ 搜索失败: {e}"


# =============================================================================
# Shell 工具
# =============================================================================

# 危险命令黑名单
DANGEROUS_COMMANDS = [
    'rm -rf /', 'rm -rf /*', 'rm -rf ~', 
    'mkfs', 'dd if=', '>:', 'format',
    'del /f /s /q', 'rd /s /q',
]


def _is_dangerous_command(command: str) -> bool:
    """检查命令是否危险"""
    cmd_lower = command.lower().strip()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return True
    return False


@tool
def shell_exec(command: str, timeout: int = 30) -> str:
    """
    执行 Shell 命令
    
    安全限制：禁止执行危险命令（如 rm -rf /），超时限制默认30秒
    
    Args:
        command: 要执行的命令
        timeout: 超时时间（秒）
        
    Returns:
        命令执行结果
    """
    if _is_dangerous_command(command):
        return f"🚫 安全拦截: 检测到危险命令"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        output = [f"$ {command}", "="*50]
        
        if result.stdout:
            output.append(result.stdout)
        
        if result.stderr:
            output.append(f"[stderr]\n{result.stderr}")
        
        output.append(f"\n[exit code: {result.returncode}]")
        
        return '\n'.join(output)
        
    except subprocess.TimeoutExpired:
        return f"⏱️ 命令执行超时（>{timeout}秒）"
    except Exception as e:
        return f"❌ 执行失败: {e}"


@tool
def shell_script(script: str, interpreter: str = "bash", timeout: int = 60) -> str:
    """
    执行 Shell 脚本
    
    Args:
        script: 脚本内容
        interpreter: 解释器（bash/sh/python/powershell）
        timeout: 超时时间（秒）
        
    Returns:
        脚本执行结果
    """
    if _is_dangerous_command(script):
        return f"🚫 安全拦截: 检测到危险命令"
    
    try:
        if interpreter == "python":
            cmd = [sys.executable, "-c", script]
        elif interpreter == "powershell":
            cmd = ["powershell", "-Command", script]
        else:
            cmd = [interpreter, "-c", script]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        output = [f"# {interpreter} script", "="*50, script, "="*50]
        
        if result.stdout:
            output.append(result.stdout)
        
        if result.stderr:
            output.append(f"[stderr]\n{result.stderr}")
        
        output.append(f"\n[exit code: {result.returncode}]")
        
        return '\n'.join(output)
        
    except subprocess.TimeoutExpired:
        return f"⏱️ 脚本执行超时（>{timeout}秒）"
    except Exception as e:
        return f"❌ 执行失败: {e}"


# =============================================================================
# Python 执行工具
# =============================================================================

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
import itertools
import collections
from pathlib import Path
from datetime import datetime, date, timedelta
"""


def _check_code_safety(code: str) -> tuple[bool, str]:
    """检查代码安全性"""
    code_lower = code.lower()
    
    for dangerous in DANGEROUS_MODULES:
        if dangerous in code_lower:
            return False, f"🚫 安全拦截: 检测到危险操作 '{dangerous}'"
    
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
    预导入的库: os, sys, json, re, math, datetime, pathlib 等
    
    Args:
        code: Python 代码（单行或多行）
        timeout: 执行超时时间（秒）
        
    Returns:
        代码执行结果
    """
    is_safe, error_msg = _check_code_safety(code)
    if not is_safe:
        return error_msg
    
    env = {}
    try:
        exec(DEFAULT_IMPORTS, env)
    except Exception as e:
        return f"❌ 环境初始化失败: {e}"
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    result_output = [">>> Python 代码执行", "="*50, f"代码:\n{code}", "="*50]
    
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                result = eval(code, env)
                if result is not None:
                    print(f"[返回值]: {repr(result)}")
            except SyntaxError:
                exec(code, env)
        
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            result_output.append(f"\n📤 输出:\n{stdout_content}")
        
        if stderr_content:
            result_output.append(f"\n⚠️ 错误:\n{stderr_content}")
        
        result_output.append("\n✅ 执行成功")
        
    except Exception as e:
        result_output.append(f"\n❌ 执行失败: {type(e).__name__}: {str(e)}")
    
    return '\n'.join(result_output)


# =============================================================================
# Web 搜索工具
# =============================================================================

@tool
def web_search(query: str, count: int = 5) -> str:
    """
    使用 SearXNG 本地搜索引擎进行网络搜索
    
    需要本地部署 SearXNG：
        docker run -d --name searxng -p 8080:8080 searxng/searxng
    
    或使用环境变量指定远程地址：
        export SEARXNG_URL='http://your-searxng-url'
    
    Args:
        query: 搜索查询
        count: 返回结果数量（1-20）
        
    Returns:
        搜索结果
    """
    try:
        # 获取 SearXNG 地址，默认本地
        searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8080")
        
        # 限制结果数量
        count = max(1, min(20, count))
        
        # 构建 SearXNG 请求
        params = {
            "q": query,
            "format": "json",
            "language": "zh-CN",
            "safesearch": "0",
        }
        
        url = f"{searxng_url}/search?{urllib.parse.urlencode(params)}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        # 解析 SearXNG 结果
        results = data.get("results", [])
        
        if not results:
            return f"未找到搜索结果。请检查 SearXNG 是否正常运行：{searxng_url}"
        
        # 限制结果数量
        results = results[:count]
        
        # 格式化结果
        output = [f"🔍 搜索 '{query}' 的结果（{len(results)} 条）:\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            content = result.get("content", "无描述")
            engine = result.get("engine", "unknown")
            
            output.append(f"{i}. {title}")
            output.append(f"   URL: {url}")
            output.append(f"   来源: {engine}")
            output.append(f"   {content}\n")
        
        return "\n".join(output)
        
    except urllib.error.URLError as e:
        return f"❌ 无法连接到 SearXNG ({os.environ.get('SEARXNG_URL', 'http://localhost:8080')})\n请确保：\n1. SearXNG 已启动: docker run -d -p 8080:8080 searxng/searxng\n2. 或设置环境变量: export SEARXNG_URL='http://your-searxng-url'\n错误: {str(e)}"
    except Exception as e:
        return f"❌ 搜索错误: {str(e)}"


@tool
def web_fetch(url: str, max_chars: int = 5000) -> str:
    """
    获取网页内容
    
    Args:
        url: 网页 URL
        max_chars: 最大字符数，默认 5000
        
    Returns:
        网页内容
    """
    try:
        # 设置请求头模拟浏览器
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            
            # 读取内容
            data = response.read()
            
            # 尝试解码
            try:
                html = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    html = data.decode("gbk")
                except UnicodeDecodeError:
                    html = data.decode("utf-8", errors="ignore")
            
            # 简单提取正文（去除 HTML 标签）
            # 移除 script 和 style
            html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
            html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
            
            # 提取 title
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else "无标题"
            
            # 移除 HTML 标签
            text = re.sub(r"<[^>]+>", " ", html)
            
            # 清理空白
            text = re.sub(r"\s+", " ", text).strip()
            
            # 截断
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[内容已截断...]"
            
            return f"📄 标题: {title}\n🔗 URL: {url}\n\n{text}"
            
    except Exception as e:
        return f"❌ 获取网页错误: {str(e)}"
