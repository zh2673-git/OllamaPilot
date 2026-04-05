"""
BuiltinToolAdapter - 包装原有 @tool 装饰的工具

将 ollamapilot.tools.builtin 中的工具包装为 Harness Tool 基类
"""

import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints

from langchain_core.tools import BaseTool

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)


class BuiltinToolAdapter(Tool):
    """
    内置工具适配器

    将使用 @tool 装饰器的原有工具（StructuredTool）包装为 Harness Tool 基类。
    保留原有工具的所有功能，添加 Harness 的生命周期管理。

    使用示例：
        from ollamapilot.tools import builtin as builtin_tools

        read_file_tool = BuiltinToolAdapter(builtin_tools.read_file)
        write_file_tool = BuiltinToolAdapter(builtin_tools.write_file)
        shell_tool = BuiltinToolAdapter(builtin_tools.shell_exec)
    """

    def __init__(self, builtin_tool: BaseTool, **config):
        """
        初始化适配器

        Args:
            builtin_tool: 原有工具（@tool 装饰后的 StructuredTool）
            **config: 额外配置
        """
        super().__init__()
        self.builtin_tool = builtin_tool
        self.config = config

        # 从 LangChain 工具提取元数据
        self.name = getattr(builtin_tool, 'name', 'unknown_tool')
        self._description = getattr(builtin_tool, 'description', '') or ''

        # 尝试获取原始函数
        self._original_func = None
        if hasattr(builtin_tool, 'func'):
            self._original_func = builtin_tool.func

    @property
    def description(self) -> str:
        """工具描述"""
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        从 LangChain 工具提取输入参数 Schema
        """
        # 尝试获取 args_schema
        if hasattr(self.builtin_tool, 'args_schema') and self.builtin_tool.args_schema:
            try:
                schema = self.builtin_tool.args_schema.schema()
                return {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            except Exception:
                pass

        # 回退到默认 schema
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询或输入"
                }
            },
            "required": ["query"]
        }

    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        验证输入参数
        """
        # 检查必填参数
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

        对于危险操作（如删除文件），可以在这里添加确认逻辑
        """
        # 默认允许执行
        return PermissionResult.allow()

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        """
        执行原有工具

        调用被包装的原有工具
        """
        try:
            # 报告进度
            if on_progress:
                on_progress(f"执行 {self.name}...", 0.1)

            # 调用原有工具
            import asyncio

            # 在线程池中执行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.builtin_tool.invoke(input_data)
            )

            # 报告完成
            if on_progress:
                on_progress("完成", 1.0)

            # 包装结果
            return ToolResult.success_result(
                output=str(result) if result else "执行成功",
                data={"raw_result": result}
            )

        except Exception as e:
            return ToolResult.error_result(
                error=str(e),
                output=f"执行 {self.name} 时出错"
            )

    def render_result(self, result: ToolResult) -> str:
        """
        渲染执行结果

        直接使用原有工具的输出格式
        """
        if result.success:
            return result.output
        else:
            return f"❌ 错误: {result.error}"

    def to_langchain_tool(self):
        """
        转换为 LangChain 工具

        直接返回原有的工具
        """
        return self.builtin_tool
