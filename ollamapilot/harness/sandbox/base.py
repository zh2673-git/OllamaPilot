"""
Sandbox 基类 - 沙箱执行环境

借鉴 DeerFlow 的 Sandbox 设计
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str = ""
    
    @classmethod
    def success_result(
        cls, 
        stdout: str, 
        stderr: str = "", 
        exit_code: int = 0
    ) -> "SandboxResult":
        return cls(
            success=True,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code
        )
    
    @classmethod
    def error_result(cls, error: str, stderr: str = "") -> "SandboxResult":
        return cls(
            success=False,
            error=error,
            stderr=stderr,
            exit_code=-1
        )


class Sandbox(ABC):
    """
    沙箱基类
    
    提供隔离的执行环境，支持：
    - 命令执行
    - 文件操作
    - 虚拟路径映射
    """
    
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = workspace_dir
    
    @abstractmethod
    async def execute(self, command: str, timeout: int = 30) -> SandboxResult:
        """
        在沙箱中执行命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            
        Returns:
            SandboxResult: 执行结果
        """
        pass
    
    @abstractmethod
    async def read_file(self, path: str) -> str:
        """
        读取沙箱中的文件
        
        Args:
            path: 文件路径（沙箱内）
            
        Returns:
            文件内容
        """
        pass
    
    @abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """
        写入沙箱中的文件
        
        Args:
            path: 文件路径（沙箱内）
            content: 文件内容
            
        Returns:
            是否成功
        """
        pass
    
    def map_path(self, host_path: str) -> str:
        """
        将主机路径映射到沙箱路径
        
        Args:
            host_path: 主机路径
            
        Returns:
            沙箱路径
        """
        return host_path
    
    def unmap_path(self, sandbox_path: str) -> str:
        """
        将沙箱路径映射回主机路径
        
        Args:
            sandbox_path: 沙箱路径
            
        Returns:
            主机路径
        """
        return sandbox_path
