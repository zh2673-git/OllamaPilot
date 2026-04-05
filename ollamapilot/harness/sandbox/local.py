"""
LocalSandbox - 本地沙箱

基于本地进程限制实现的轻量级沙箱
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

from ollamapilot.harness.sandbox.base import Sandbox, SandboxResult


class LocalSandbox(Sandbox):
    """
    本地沙箱
    
    使用临时目录和进程限制实现的轻量级沙箱。
    适用于：
    - 文件隔离
    - 命令执行限制
    - 资源限制
    """
    
    def __init__(
        self, 
        workspace_dir: str = "./workspace",
        temp_dir: Optional[str] = None,
        restricted_commands: Optional[list] = None
    ):
        super().__init__(workspace_dir)
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="ollamapilot_sandbox_")
        self.restricted_commands = restricted_commands or [
            'rm -rf /', 'mkfs', 'dd if=/dev/zero',
            'sudo', 'su -',
        ]
    
    def _is_restricted(self, command: str) -> bool:
        """检查命令是否被限制"""
        cmd_lower = command.lower()
        for restricted in self.restricted_commands:
            if restricted in cmd_lower:
                return True
        return False
    
    async def execute(self, command: str, timeout: int = 30) -> SandboxResult:
        """在沙箱中执行命令"""
        if self._is_restricted(command):
            return SandboxResult.error_result("命令被沙箱限制")
        
        try:
            # 在临时目录中执行
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.temp_dir,
                encoding='utf-8',
                errors='ignore'
            )
            
            return SandboxResult.success_result(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return SandboxResult.error_result(f"执行超时（>{timeout}秒）")
        except Exception as e:
            return SandboxResult.error_result(str(e))
    
    async def read_file(self, path: str) -> str:
        """读取沙箱中的文件"""
        # 将路径限制在沙箱目录内
        safe_path = self._safe_path(path)
        
        try:
            with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"读取失败: {e}"
    
    async def write_file(self, path: str, content: str) -> bool:
        """写入沙箱中的文件"""
        # 将路径限制在沙箱目录内
        safe_path = self._safe_path(path)
        
        try:
            # 确保目录存在
            Path(safe_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    def _safe_path(self, path: str) -> str:
        """
        获取安全的文件路径
        
        确保路径在沙箱目录内，防止目录遍历攻击
        """
        # 解析路径
        requested_path = Path(self.temp_dir) / path
        requested_path = requested_path.resolve()
        
        # 确保在沙箱目录内
        sandbox_root = Path(self.temp_dir).resolve()
        
        try:
            requested_path.relative_to(sandbox_root)
            return str(requested_path)
        except ValueError:
            # 路径在沙箱外，返回沙箱根目录
            return str(sandbox_root / "unsafe_file")
    
    def cleanup(self):
        """清理沙箱"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
