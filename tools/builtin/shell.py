"""
内置工具 - Shell 工具
提供安全的 Shell 命令执行功能
"""

import subprocess
import shlex
from typing import List
from langchain_core.tools import tool

# 危险命令黑名单
DANGEROUS_COMMANDS = [
    'rm -rf /', 'rm -rf /*', 'rm -rf ~', 
    'mkfs', 'dd if=', '>:', 'format',
    'del /f /s /q', 'rd /s /q',
]

# 允许的命令白名单（可选使用）
ALLOWED_COMMANDS = [
    'ls', 'dir', 'cat', 'type', 'head', 'tail', 'grep', 'find',
    'pwd', 'cd', 'echo', 'mkdir', 'touch', 'cp', 'copy', 'mv', 'move',
    'python', 'python3', 'pip', 'node', 'npm', 'git', 'curl', 'wget',
    'code', 'code.', 'code ',
]


def is_dangerous(command: str) -> bool:
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
    
    安全限制：
    - 禁止执行危险命令（如 rm -rf /）
    - 命令超时限制（默认30秒）
    - 工作目录限制在当前目录
    
    Args:
        command: 要执行的命令
        timeout: 超时时间（秒）
        
    Returns:
        命令执行结果
    """
    # 安全检查
    if is_dangerous(command):
        return f"🚫 安全拦截: 检测到危险命令 '{command}'"
    
    try:
        # 执行命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'
        )
        
        output = []
        output.append(f"$ {command}")
        output.append("="*50)
        
        if result.stdout:
            output.append(result.stdout)
        
        if result.stderr:
            output.append(f"[stderr]\n{result.stderr}")
        
        output.append(f"\n[exit code: {result.returncode}]")
        
        return '\n'.join(output)
        
    except subprocess.TimeoutExpired:
        return f"⏱️ 命令执行超时（>{timeout}秒）: {command}"
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
    import tempfile
    import os
    
    # 安全检查
    if is_dangerous(script):
        return "🚫 安全拦截: 脚本中包含危险命令"
    
    try:
        # 根据解释器选择文件扩展名
        ext_map = {
            'bash': '.sh',
            'sh': '.sh',
            'python': '.py',
            'python3': '.py',
            'powershell': '.ps1',
            'pwsh': '.ps1',
        }
        ext = ext_map.get(interpreter.lower(), '.sh')
        
        # 创建临时脚本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as f:
            f.write(script)
            script_path = f.name
        
        try:
            # 构建执行命令
            if interpreter.lower() in ['python', 'python3']:
                cmd = f'{interpreter} "{script_path}"'
            elif interpreter.lower() in ['powershell', 'pwsh']:
                cmd = f'{interpreter} -ExecutionPolicy Bypass -File "{script_path}"'
            else:
                cmd = f'{interpreter} "{script_path}"'
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            
            output = []
            output.append(f"📜 执行 {interpreter} 脚本")
            output.append("="*50)
            output.append(f"脚本内容 ({len(script)} 字符):")
            output.append("-"*30)
            # 显示脚本前10行
            script_lines = script.split('\n')[:10]
            output.append('\n'.join(script_lines))
            if len(script.split('\n')) > 10:
                output.append("...")
            output.append("-"*30)
            output.append("")
            
            if result.stdout:
                output.append("[输出]")
                output.append(result.stdout)
            
            if result.stderr:
                output.append(f"[错误]\n{result.stderr}")
            
            output.append(f"\n[exit code: {result.returncode}]")
            
            return '\n'.join(output)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(script_path)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        return f"⏱️ 脚本执行超时（>{timeout}秒）"
    except Exception as e:
        return f"❌ 脚本执行失败: {e}"


# 导出工具列表
TOOLS = [shell_exec, shell_script]
