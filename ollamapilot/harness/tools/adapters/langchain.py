"""
LangChainToolAdapter - 包装 LangChain BaseTool

将 Skill 提供的 LangChain BaseTool 包装为 Harness Tool 基类
"""

from typing import Any, Callable, Dict, Optional

from langchain_core.tools import BaseTool

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)


class LangChainToolAdapter(Tool):
    """
    LangChain 工具适配器

    将 LangChain BaseTool（包括 Skill 提供的工具）包装为 Harness Tool 基类。
    保留原有工具的所有功能，添加 Harness 的生命周期管理。

    使用示例：
        from some_skill import some_tool

        adapted_tool = LangChainToolAdapter(some_tool)
    """

    def __init__(self, langchain_tool: BaseTool, **config):
        """
        初始化适配器

        Args:
            langchain_tool: LangChain BaseTool 实例
            **config: 额外配置
        """
        super().__init__()
        self.langchain_tool = langchain_tool
        self.config = config

        # 从 LangChain 工具提取元数据
        self.name = getattr(langchain_tool, 'name', 'unknown_tool')
        self._description = getattr(langchain_tool, 'description', '') or ''

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
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            try:
                schema = self.langchain_tool.args_schema.schema()
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
        执行 LangChain 工具
        """
        try:
            # 报告进度
            if on_progress:
                on_progress(f"执行 {self.name}...", 0.1)

            # 调用 LangChain 工具
            import asyncio

            # 在线程池中执行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.langchain_tool.invoke(input_data)
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
        """
        if result.success:
            return result.output
        else:
            return f"❌ 错误: {result.error}"

    def to_langchain_tool(self):
        """
        转换为 LangChain 工具

        直接返回原有的 LangChain 工具
        """
        return self.langchain_tool
