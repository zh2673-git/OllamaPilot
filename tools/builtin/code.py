"""
内置工具 - 代码工具
提供代码搜索、补丁应用、代码统计功能
"""

import re
import ast
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool


@tool
def code_search(pattern: str, dir_path: str = ".", language: Optional[str] = None) -> str:
    """
    在代码中搜索模式
    
    Args:
        pattern: 搜索模式（正则表达式或字符串）
        dir_path: 搜索目录，默认当前目录
        language: 语言过滤（如 python, javascript），可选
        
    Returns:
        搜索结果
    """
    try:
        path = Path(dir_path)
        if not path.exists():
            return f"❌ 目录不存在: {dir_path}"
        
        # 语言扩展名映射
        ext_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'go': ['.go'],
            'rust': ['.rs'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp', '.cc'],
        }
        
        extensions = ext_map.get(language.lower(), None) if language else None
        
        matches = []
        
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # 检查扩展名
            if extensions and file_path.suffix not in extensions:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # 搜索匹配
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    try:
                        if re.search(pattern, line, re.IGNORECASE):
                            matches.append({
                                'file': file_path,
                                'line': i,
                                'content': line.strip()[:100]  # 限制长度
                            })
                    except re.error:
                        # 如果不是有效正则，使用简单字符串匹配
                        if pattern.lower() in line.lower():
                            matches.append({
                                'file': file_path,
                                'line': i,
                                'content': line.strip()[:100]
                            })
            except Exception:
                continue
        
        if not matches:
            return f"🔍 未找到匹配: '{pattern}'"
        
        # 格式化输出
        lines = [f"🔍 代码搜索结果: '{pattern}'", f"📁 目录: {path.absolute()}", "="*50]
        
        # 按文件分组
        current_file = None
        for match in matches[:30]:  # 限制结果数量
            if match['file'] != current_file:
                current_file = match['file']
                rel_path = current_file.relative_to(path) if current_file.is_relative_to(path) else current_file
                lines.append(f"\n📄 {rel_path}")
            
            lines.append(f"  {match['line']:4d}: {match['content']}")
        
        if len(matches) > 30:
            lines.append(f"\n... 还有 {len(matches) - 30} 个结果")
        
        lines.append(f"\n共找到 {len(matches)} 处匹配")
        return '\n'.join(lines)
        
    except Exception as e:
        return f"❌ 搜索失败: {e}"


@tool
def apply_patch(file_path: str, search: str, replace: str) -> str:
    """
    应用代码补丁（搜索替换）
    
    Args:
        file_path: 目标文件路径
        search: 要搜索的代码块
        replace: 替换为的代码块
        
    Returns:
        操作结果
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"❌ 文件不存在: {file_path}"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查搜索内容是否存在
        if search not in content:
            return f"❌ 未找到要替换的内容\n搜索:\n{search[:200]}..."
        
        # 执行替换
        new_content = content.replace(search, replace, 1)  # 只替换第一处
        
        # 写回文件
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"✅ 补丁应用成功: {file_path}\n   替换了 {len(search)} 字符 → {len(replace)} 字符"
        
    except Exception as e:
        return f"❌ 应用补丁失败: {e}"


@tool
def code_stats(dir_path: str = ".", language: Optional[str] = None) -> str:
    """
    代码统计
    
    Args:
        dir_path: 统计目录，默认当前目录
        language: 指定语言，可选
        
    Returns:
        统计结果
    """
    try:
        path = Path(dir_path)
        if not path.exists():
            return f"❌ 目录不存在: {dir_path}"
        
        # 扩展名映射
        ext_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'go': ['.go'],
            'rust': ['.rs'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp'],
            'markdown': ['.md'],
        }
        
        extensions = ext_map.get(language.lower(), None) if language else None
        
        stats = {
            'files': 0,
            'lines': 0,
            'code_lines': 0,
            'blank_lines': 0,
            'comment_lines': 0,
        }
        
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # 跳过隐藏文件和特定目录
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'node_modules' in str(file_path) or '__pycache__' in str(file_path):
                continue
            
            # 检查扩展名
            if extensions and file_path.suffix not in extensions:
                continue
            elif not extensions:
                # 默认只统计代码文件
                if file_path.suffix not in ['.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp', '.md']:
                    continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                stats['files'] += 1
                stats['lines'] += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        stats['blank_lines'] += 1
                    elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                        stats['comment_lines'] += 1
                    else:
                        stats['code_lines'] += 1
                        
            except Exception:
                continue
        
        # 格式化输出
        lines = [
            f"📊 代码统计: {path.absolute()}",
            "="*50,
            f"📁 文件数: {stats['files']}",
            f"📄 总行数: {stats['lines']:,}",
            f"💻 代码行: {stats['code_lines']:,}",
            f"💬 注释行: {stats['comment_lines']:,}",
            f"⬜ 空行: {stats['blank_lines']:,}",
            "="*50,
        ]
        
        if stats['lines'] > 0:
            code_ratio = stats['code_lines'] / stats['lines'] * 100
            comment_ratio = stats['comment_lines'] / stats['lines'] * 100
            lines.append(f"代码占比: {code_ratio:.1f}%")
            lines.append(f"注释占比: {comment_ratio:.1f}%")
        
        return '\n'.join(lines)
        
    except Exception as e:
        return f"❌ 统计失败: {e}"


# 导出工具列表
TOOLS = [code_search, apply_patch, code_stats]
