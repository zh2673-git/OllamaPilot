"""
BashTool - Shell 命令执行工具

借鉴 Claude Code 的 BashTool 设计，支持：
- 安全校验
- 超时控制
- 进度回调
- 沙箱执行（可选）
"""

import subprocess
from typing import Any, Callable, Dict, Optional

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)


class BashTool(Tool):
    """
    Bash 命令执行工具
    
    支持安全校验、超时控制、沙箱执行。
    """
    
    name = "bash"
    description = "执行 Bash shell 命令"
    is_read_only = False
    is_destructive = True
    
    # 危险命令黑名单
    DANGEROUS_COMMANDS = [
        'rm -rf /', 'rm -rf /*', 'rm -rf ~', 
        'mkfs', 'dd if=/dev/zero', '>:', 'format',
        'del /f /s /q', 'rd /s /q',
    ]
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Bash 命令"
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒），默认30秒",
                    "default": 30
                },
                "description": {
                    "type": "string",
                    "description": "命令描述（用于日志）"
                }
            },
            "required": ["command"]
        }
    
    def _is_dangerous(self, command: str) -> bool:
        """检查命令是否危险"""
        cmd_lower = command.lower().strip()
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in cmd_lower:
                return True
        return False
    
    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        """验证输入"""
        # 基类验证
        base_result = await super().validate(input_data)
        if not base_result.valid:
            return base_result
        
        command = input_data.get('command', '')
        
        if not command.strip():
            return ValidationResult.failure("命令不能为空")
        
        if self._is_dangerous(command):
            return ValidationResult.failure("检测到危险命令，已拦截")
        
        return ValidationResult.success()
    
    async def check_permission(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext
    ) -> PermissionResult:
        """权限检查"""
        command = input_data.get('command', '')
        
        # 检查是否需要沙箱
        if context.sandbox and self._needs_sandbox(command):
            return PermissionResult.allow({
                **input_data,
                "use_sandbox": True
            })
        
        return PermissionResult.allow()
    
    def _needs_sandbox(self, command: str) -> bool:
        """判断是否需要沙箱执行"""
        # 文件写入、网络操作等需要沙箱
        sandbox_patterns = [
            'curl', 'wget', 'fetch',
            '> ', '>> ', 'tee',
            'sudo', 'su -',
        ]
        cmd_lower = command.lower()
        return any(p in cmd_lower for p in sandbox_patterns)
    
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        """执行命令"""
        command = input_data.get('command', '')
        timeout = input_data.get('timeout', context.timeout)
        use_sandbox = input_data.get('use_sandbox', False)
        
        # 如果使用沙箱
        if use_sandbox and context.sandbox:
            return await self._execute_in_sandbox(command, context, on_progress)
        
        # 本地执行
        try:
            if on_progress:
                on_progress(f"执行: {command}", 0.1)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='ignore'
            )
            
            if on_progress:
                on_progress("执行完成", 1.0)
            
            output = []
            output.append(f"$ {command}")
            output.append("=" * 50)
            
            if result.stdout:
                output.append(result.stdout)
            
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr}")
            
            output.append(f"\n[exit code: {result.returncode}]")
            
            return ToolResult.success_result(
                '\n'.join(output),
                {"exit_code": result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult.error_result(f"命令执行超时（>{timeout}秒）")
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    async def _execute_in_sandbox(
        self, 
        command: str, 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]]
    ) -> ToolResult:
        """在沙箱中执行"""
        try:
            if on_progress:
                on_progress(f"在沙箱中执行: {command}", 0.1)
            
            result = await context.sandbox.execute(command)
            
            if on_progress:
                on_progress("沙箱执行完成", 1.0)
            
            return result
        except Exception as e:
            return ToolResult.error_result(f"沙箱执行失败: {e}")
