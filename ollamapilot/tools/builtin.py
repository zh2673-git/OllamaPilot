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
import time
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool


# =============================================================================
# SearXNG 管理工具
# =============================================================================

def _check_searxng_running(url: str = "http://localhost:8080") -> bool:
    """检查 SearXNG 是否正在运行"""
    try:
        req = urllib.request.Request(
            f"{url}/healthz",
            headers={"User-Agent": "OllamaPilot/1.0"},
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _check_docker_image_exists(image_name: str = "searxng/searxng") -> bool:
    """检查 Docker 镜像是否已存在"""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _pull_docker_image(image_name: str = "searxng/searxng") -> tuple[bool, str]:
    """
    拉取 Docker 镜像
    
    Returns:
        (是否成功, 消息)
    """
    try:
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=120  # 拉取镜像可能需要较长时间
        )
        if result.returncode == 0:
            return True, f"镜像 {image_name} 拉取成功"
        else:
            return False, f"镜像拉取失败: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "镜像拉取超时"
    except Exception as e:
        return False, f"镜像拉取失败: {str(e)}"


def _start_searxng_docker() -> tuple[bool, str]:
    """
    自动启动 SearXNG Docker 容器
    
    完整流程：
    1. 检查 Docker 是否安装
    2. 检查是否已有容器在运行
    3. 检查是否有停止的容器
    4. 检查镜像是否存在（不存在则自动拉取）
    5. 创建并启动容器
    6. 等待服务就绪
    
    Returns:
        (是否成功, 消息)
    """
    try:
        # 步骤 1: 检查 Docker 是否安装
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False, "Docker 未安装或未启动"
        
        # 步骤 2: 检查是否已有容器在运行
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=searxng"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout.strip():
            # 容器已在运行，检查健康状态
            if _check_searxng_running():
                return True, "SearXNG 容器已在运行"
            else:
                return False, "SearXNG 容器存在但无法访问，请检查日志: docker logs searxng"
        
        # 步骤 3: 检查是否有停止的容器
        result = subprocess.run(
            ["docker", "ps", "-aq", "-f", "name=searxng"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.stdout.strip():
            # 启动已存在的容器
            subprocess.run(
                ["docker", "start", "searxng"],
                capture_output=True,
                timeout=30
            )
        else:
            # 步骤 4: 检查镜像是否存在，不存在则自动拉取
            if not _check_docker_image_exists("searxng/searxng"):
                success, message = _pull_docker_image("searxng/searxng")
                if not success:
                    return False, f"镜像拉取失败: {message}"
            
            # 步骤 5: 创建新容器
            subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", "searxng",
                    "-p", "8080:8080",
                    "-e", "BASE_URL=http://localhost:8080/",
                    "-e", "INSTANCE_NAME=OllamaPilot-SearXNG",
                    "searxng/searxng"
                ],
                capture_output=True,
                timeout=60
            )
        
        # 步骤 6: 等待服务启动
        for i in range(30):  # 最多等待30秒
            time.sleep(1)
            if _check_searxng_running():
                return True, "SearXNG 启动成功"
        
        return False, "SearXNG 启动超时，请检查日志: docker logs searxng"
        
    except subprocess.TimeoutExpired:
        return False, "Docker 命令执行超时"
    except FileNotFoundError:
        return False, "Docker 命令未找到，请确保 Docker 已安装"
    except Exception as e:
        return False, f"启动失败: {str(e)}"


@tool
def web_search_setup(action: str = "status") -> str:
    """
    管理 SearXNG 搜索服务的部署和配置
    
    用于自动部署、检查和管理本地 SearXNG 搜索引擎。
    
    Args:
        action: 操作类型
            - "status": 检查服务状态
            - "start": 自动启动 SearXNG Docker 容器
            - "stop": 停止 SearXNG 容器
            - "logs": 查看容器日志
            
    Returns:
        操作结果信息
        
    Examples:
        web_search_setup("status")  # 检查状态
        web_search_setup("start")   # 自动部署并启动
    """
    searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8080")
    
    if action == "status":
        is_running = _check_searxng_running(searxng_url)
        
        if is_running:
            return f"✅ SearXNG 服务正常运行\n地址: {searxng_url}\n\nweb_search 工具已可用"
        else:
            return f"⚠️ SearXNG 服务未运行\n地址: {searxng_url}\n\n解决方案:\n1. 自动部署: web_search_setup('start')\n2. 手动部署: docker run -d --name searxng -p 8080:8080 searxng/searxng\n3. 使用远程: 设置环境变量 SEARXNG_URL"
    
    elif action == "start":
        success, message = _start_searxng_docker()
        if success:
            return f"✅ {message}\n\nSearXNG 地址: http://localhost:8080\nweb_search 工具现在可以使用了"
        else:
            return f"❌ {message}\n\n手动部署命令:\ndocker run -d --name searxng -p 8080:8080 -e BASE_URL=http://localhost:8080/ searxng/searxng"
    
    elif action == "stop":
        try:
            subprocess.run(
                ["docker", "stop", "searxng"],
                capture_output=True,
                timeout=30
            )
            return "✅ SearXNG 已停止"
        except Exception as e:
            return f"❌ 停止失败: {str(e)}"
    
    elif action == "logs":
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", "50", "searxng"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return f"📋 SearXNG 日志:\n{result.stdout}"
            else:
                return f"❌ 获取日志失败: {result.stderr}"
        except Exception as e:
            return f"❌ 获取日志失败: {str(e)}"
    
    else:
        return f"❌ 未知操作: {action}\n可用操作: status, start, stop, logs"


# =============================================================================
# 文件系统工具
# =============================================================================

@tool
def read_file(file_path: str, limit: int = 1000) -> str:
    """
    读取文件内容

    支持文本文件和 PDF 文件。PDF 文件会自动提取文本内容。

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

        # 检查文件类型
        suffix = path.suffix.lower()

        # PDF 文件特殊处理
        if suffix == '.pdf':
            return _read_pdf_file(path, limit)

        # 普通文本文件
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


def _read_pdf_file(path: Path, limit: int = 1000) -> str:
    """读取 PDF 文件并提取文本"""
    try:
        import PyPDF2

        text = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                if i >= limit:
                    text.append(f"\n... (已截断，共读取 {limit} 页，总共 {total_pages} 页)")
                    break
                page_text = page.extract_text()
                if page_text:
                    text.append(f"\n--- 第 {i+1} 页 ---\n{page_text}")

        content = '\n'.join(text)
        return f"📄 PDF 文件: {path}\n{'='*50}\n{content}"

    except ImportError:
        return f"❌ 未安装 PyPDF2，无法读取 PDF。请运行: pip install PyPDF2"
    except Exception as e:
        return f"❌ 读取 PDF 失败: {e}"


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
def web_search(query: str, count: int = 5, auto_start: bool = True) -> str:
    """
    智能网络搜索 - 支持多引擎优先级和自动降级
    
    搜索策略：
    1. 优先使用 SearXNG（本地部署，质量最佳）
    2. SearXNG 不可用时自动降级到 DuckDuckGo（免费，无需配置）
    3. 如果配置了其他 API（Serper/Bing/Brave），也会按优先级尝试
    
    支持自动部署：如果检测到 SearXNG 未运行，会尝试自动启动 Docker 容器
    
    手动部署命令：
        docker run -d --name searxng -p 8080:8080 searxng/searxng
    
    或使用环境变量指定远程地址：
        export SEARXNG_URL='http://your-searxng-url'
    
    Args:
        query: 搜索查询
        count: 返回结果数量（1-20）
        auto_start: 是否自动尝试启动 SearXNG（默认True）
        
    Returns:
        搜索结果
    """
    import asyncio
    
    # 限制结果数量
    count = max(1, min(20, count))
    
    # 尝试使用增强搜索的路由器（如果可用）
    try:
        from skills.enhanced_search.engine_router import SearchEngineRouter
        
        async def _smart_search():
            router = SearchEngineRouter()
            results = await router.search(query, category="general", num_results=count)
            return results
        
        # 运行异步搜索
        results = asyncio.run(_smart_search())
        
        if results:
            # 格式化结果
            output = [f"🔍 搜索 '{query}' 的结果（{len(results)} 条）:\n"]
            
            for i, result in enumerate(results, 1):
                title = result.title if hasattr(result, 'title') else "无标题"
                url = result.url if hasattr(result, 'url') else ""
                snippet = result.snippet if hasattr(result, 'snippet') else "无描述"
                source = result.source if hasattr(result, 'source') else "unknown"
                
                output.append(f"{i}. {title}")
                output.append(f"   URL: {url}")
                output.append(f"   来源: {source}")
                output.append(f"   {snippet}\n")
            
            return "\n".join(output)
    except Exception as e:
        # 增强搜索不可用，回退到原始 SearXNG 逻辑
        print(f"⚠️ 智能搜索失败，回退到 SearXNG: {e}")
        pass
    
    # 回退到原始 SearXNG 搜索逻辑
    return _web_search_searxng(query, count, auto_start)


def _web_search_searxng(query: str, count: int = 5, auto_start: bool = True) -> str:
    """
    使用 SearXNG 进行网络搜索（原始实现）
    """
    # 获取 SearXNG 地址
    searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8080")
    
    # 检查 SearXNG 是否运行
    if not _check_searxng_running(searxng_url):
        if auto_start:
            # 尝试自动启动
            success, message = _start_searxng_docker()
            if not success:
                return f"❌ SearXNG 未运行，自动启动失败\n{message}\n\n你可以：\n1. 手动部署: docker run -d --name searxng -p 8080:8080 searxng/searxng\n2. 使用远程: export SEARXNG_URL='http://your-searxng-url'\n3. 检查状态: web_search_setup('status')"
        else:
            return f"❌ SearXNG 未运行\n地址: {searxng_url}\n\n解决方案:\n1. 自动部署: web_search_setup('start')\n2. 手动部署: docker run -d --name searxng -p 8080:8080 searxng/searxng"
    
    try:
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
        return f"❌ 无法连接到 SearXNG ({searxng_url})\n错误: {str(e)}"
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
