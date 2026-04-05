"""
Tool 基类 - 借鉴 Claude Code 的 Tool 生命周期设计

完整生命周期：
1. validate() - 输入校验
2. check_permission() - 权限检查
3. execute() - 执行逻辑
4. render_result() - 结果渲染
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import logging

logger = logging.getLogger("ollamapilot.harness.tools")


class PermissionStatus(Enum):
    """权限状态"""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    message: str = ""
    error_code: int = 0
    
    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True)
    
    @classmethod
    def failure(cls, message: str, error_code: int = 400) -> "ValidationResult":
        return cls(valid=False, message=message, error_code=error_code)


@dataclass
class PermissionResult:
    """权限检查结果"""
    status: PermissionStatus
    message: str = ""
    updated_input: Optional[Dict[str, Any]] = None
    
    @classmethod
    def allow(cls, updated_input: Optional[Dict[str, Any]] = None) -> "PermissionResult":
        return cls(status=PermissionStatus.ALLOW, updated_input=updated_input)
    
    @classmethod
    def deny(cls, message: str) -> "PermissionResult":
        return cls(status=PermissionStatus.DENY, message=message)
    
    @classmethod
    def ask(cls, message: str) -> "PermissionResult":
        return cls(status=PermissionStatus.ASK, message=message)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    
    @classmethod
    def success_result(cls, output: str, data: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(success=True, output=output, data=data or {})
    
    @classmethod
    def error_result(cls, error: str, output: str = "") -> "ToolResult":
        return cls(success=False, error=error, output=output)


@dataclass
class ToolContext:
    """工具执行上下文"""
    workspace_dir: str = "./workspace"
    thread_id: str = "default"
    sandbox: Optional[Any] = None
    verbose: bool = True
    
    # 执行控制
    timeout: int = 30
    max_output_size: int = 10000
    
    def log(self, message: str):
        """记录日志"""
        if self.verbose:
            logger.info(message)


# 进度回调类型
ProgressCallback = Callable[[str, float], None]


class Tool(ABC):
    """
    Tool 基类 - 借鉴 Claude Code 设计
    
    完整生命周期：
    1. validate(input) -> ValidationResult
    2. check_permission(input, context) -> PermissionResult
    3. execute(input, context, on_progress) -> ToolResult
    4. render_result(result) -> str
    
    属性：
    - name: 工具名称
    - description: 工具描述
    - input_schema: 输入参数定义
    """
    
    # 工具元数据
    name: str = "base_tool"
    description: str = "基础工具"
    
    # 行为配置
    is_enabled: bool = True
    is_read_only: bool = False
    is_destructive: bool = False
    max_result_size: int = 10000
    
    def __init__(self):
        pass
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        输入参数 Schema
        
        返回 JSON Schema 格式的参数定义
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        输入校验
        
        Args:
            input_data: 输入参数
            
        Returns:
            ValidationResult: 校验结果
        """
        # 默认实现：检查必填参数
        schema = self.input_schema
        required = schema.get("required", [])
        
        for field in required:
            if field not in input_data:
                return ValidationResult.failure(f"缺少必填参数: {field}")
        
        return ValidationResult.success()
    
    async def check_permission(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext
    ) -> PermissionResult:
        """
        权限检查
        
        Args:
            input_data: 输入参数
            context: 执行上下文
            
        Returns:
            PermissionResult: 权限检查结果
        """
        # 默认实现：允许执行
        return PermissionResult.allow()
    
    @abstractmethod
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[ProgressCallback] = None
    ) -> ToolResult:
        """
        执行工具
        
        Args:
            input_data: 经过校验的输入参数
            context: 执行上下文
            on_progress: 进度回调
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    def render_result(self, result: ToolResult) -> str:
        """
        渲染执行结果
        
        Args:
            result: 执行结果
            
        Returns:
            str: 渲染后的结果字符串
        """
        if result.success:
            return result.output
        else:
            return f"❌ 错误: {result.error}"
    
    async def run(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[ToolContext] = None,
        on_progress: Optional[ProgressCallback] = None
    ) -> str:
        """
        运行工具的完整生命周期
        
        Args:
            input_data: 输入参数
            context: 执行上下文（可选）
            on_progress: 进度回调（可选）
            
        Returns:
            str: 渲染后的结果
        """
        context = context or ToolContext()
        
        # 1. 输入校验
        validation = await self.validate(input_data)
        if not validation.valid:
            result = ToolResult.error_result(validation.message)
            return self.render_result(result)
        
        # 2. 权限检查
        permission = await self.check_permission(input_data, context)
        if permission.status == PermissionStatus.DENY:
            result = ToolResult.error_result(f"权限拒绝: {permission.message}")
            return self.render_result(result)
        elif permission.status == PermissionStatus.ASK:
            # 返回询问消息
            return f"🤔 需要确认: {permission.message}"
        
        # 使用更新后的输入
        if permission.updated_input:
            input_data = permission.updated_input
        
        # 3. 执行
        try:
            result = await self.execute(input_data, context, on_progress)
        except Exception as e:
            logger.exception(f"工具 {self.name} 执行失败")
            result = ToolResult.error_result(str(e))
        
        # 4. 渲染结果
        return self.render_result(result)
    
    def to_langchain_tool(self):
        """
        转换为 LangChain 工具格式
        
        Returns:
            BaseTool: LangChain 工具
        """
        from langchain_core.tools import StructuredTool
        
        async def _run(**kwargs):
            return await self.run(kwargs)
        
        return StructuredTool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
        )
