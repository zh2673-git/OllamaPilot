"""
内置工具 - 文件系统工具
提供基础文件读写、目录浏览、文件搜索功能
"""

import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool


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
        
        # 确保目录存在
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
                # 检查文件名匹配
                if pattern.lower() in item.name.lower():
                    # 检查扩展名
                    if file_ext and not item.name.endswith(file_ext):
                        continue
                    matches.append(item)
        
        if not matches:
            return f"🔍 未找到匹配的文件: '{pattern}'"
        
        lines = [f"🔍 搜索结果: '{pattern}'", f"📁 目录: {path.absolute()}", "="*50]
        for match in matches[:50]:  # 限制结果数量
            rel_path = match.relative_to(path)
            size = match.stat().st_size
            lines.append(f"📄 {rel_path} ({format_size(size)})")
        
        if len(matches) > 50:
            lines.append(f"\n... 还有 {len(matches) - 50} 个结果")
        
        lines.append(f"\n共找到 {len(matches)} 个文件")
        return '\n'.join(lines)
        
    except Exception as e:
        return f"❌ 搜索失败: {e}"


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    else:
        return f"{size_bytes/(1024*1024):.1f}MB"


# 导出工具列表
TOOLS = [read_file, write_file, list_directory, search_files]
