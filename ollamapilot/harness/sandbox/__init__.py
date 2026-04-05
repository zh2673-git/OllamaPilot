"""
沙箱系统 - 可选的安全执行环境

借鉴 DeerFlow 的 Sandbox 设计
支持 Local 和 Docker 两种模式
"""

from ollamapilot.harness.sandbox.base import Sandbox, SandboxResult
from ollamapilot.harness.sandbox.local import LocalSandbox

__all__ = [
    "Sandbox",
    "SandboxResult",
    "LocalSandbox",
]
