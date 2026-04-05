"""
SubAgent 基类 - 子 Agent 定义

借鉴 DeerFlow 的子 Agent 设计
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SubAgentResult:
    """子 Agent 执行结果"""
    success: bool
    output: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    sub_tasks: List[str] = field(default_factory=list)
    
    @classmethod
    def success_result(
        cls, 
        output: str, 
        data: Optional[Dict[str, Any]] = None,
        sub_tasks: Optional[List[str]] = None
    ) -> "SubAgentResult":
        return cls(
            success=True,
            output=output,
            data=data or {},
            sub_tasks=sub_tasks or []
        )
    
    @classmethod
    def error_result(cls, error: str, output: str = "") -> "SubAgentResult":
        return cls(
            success=False,
            error=error,
            output=output
        )


class SubAgent(ABC):
    """
    子 Agent 基类
    
    用于处理特定类型的子任务。
    
    使用场景：
    1. 复杂任务分解
    2. 专业化处理
    3. 并行执行
    """
    
    name: str = "base_subagent"
    description: str = "基础子 Agent"
    
    def __init__(self, model: Any, **kwargs):
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> SubAgentResult:
        """
        执行子任务
        
        Args:
            task: 任务描述
            context: 上下文信息
            
        Returns:
            SubAgentResult: 执行结果
        """
        pass
    
    def can_handle(self, task: str) -> bool:
        """
        判断是否能处理该任务
        
        Args:
            task: 任务描述
            
        Returns:
            是否能处理
        """
        return True
