"""
SkillToolAdapter - 将现有 Skill 适配为 Tool 基类

Layer 2: 适配器层
- 将现有 Skill 包装为新的 Tool 基类
- 100% 兼容，无需修改现有 Skill
"""

from typing import Any, Dict, List, Optional, Callable
import logging

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)

logger = logging.getLogger("ollamapilot.harness.tools.adapter")


class SkillToolAdapter(Tool):
    """
    Skill 工具适配器

    将现有的 Skill 适配为 Tool 基类，获得完整生命周期管理能力。
    
    适配逻辑：
    - Skill.get_tools() -> Tool.execute() 包装
    - Skill 参数 -> Tool.input_schema 转换
    - Skill 执行结果 -> ToolResult 转换
    
    100% 兼容，无需修改现有 Skill！
    """

    def __init__(self, skill: Any):
        """
        初始化适配器
        
        Args:
            skill: 现有的 Skill 实例（BaseSkill 子类）
        """
        super().__init__()
        self.skill = skill
        
        # 从 Skill 提取元数据
        self.name = getattr(skill, 'name', 'unknown_skill')
        self.description = getattr(skill, 'description', 'Skill 适配工具')
        
        # 提取参数 schema（如果 Skill 有定义）
        self._parameters_schema = getattr(skill, 'parameters_schema', None)
        
        # 缓存 Skill 的工具
        self._skill_tools: List[Any] = []
        self._load_skill_tools()

    def _load_skill_tools(self):
        """加载 Skill 提供的工具"""
        try:
            if hasattr(self.skill, 'get_tools'):
                self._skill_tools = self.skill.get_tools() or []
        except Exception as e:
            logger.warning(f"加载 Skill {self.name} 的工具失败: {e}")
            self._skill_tools = []

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        从 Skill 提取输入参数 Schema
        
        如果 Skill 定义了 parameters_schema，直接使用。
        否则返回通用 schema。
        """
        if self._parameters_schema:
            return self._parameters_schema
        
        # 通用 schema：接受任意参数
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询或输入"
                },
                "args": {
                    "type": "object",
                    "description": "额外参数"
                }
            },
            "required": ["query"]
        }

    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        调用 Skill 的验证逻辑
        
        如果 Skill 有 validate_parameters 方法，调用它。
        否则使用基类的默认验证。
        """
        # 先执行基类验证
        base_result = await super().validate(input_data)
        if not base_result.valid:
            return base_result
        
        # 如果 Skill 有自定义验证，调用它
        if hasattr(self.skill, 'validate_parameters'):
            try:
                self.skill.validate_parameters(input_data)
                return ValidationResult.success()
            except Exception as e:
                return ValidationResult.failure(str(e))
        
        return ValidationResult.success()

    async def check_permission(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext
    ) -> PermissionResult:
        """
        权限检查
        
        如果 Skill 有 check_permission 方法，调用它。
        否则默认允许。
        """
        if hasattr(self.skill, 'check_permission'):
            try:
                result = self.skill.check_permission(input_data)
                if isinstance(result, bool):
                    return PermissionResult.allow() if result else PermissionResult.deny("Skill 权限检查失败")
                return result
            except Exception as e:
                logger.warning(f"Skill {self.name} 权限检查失败: {e}")
        
        return PermissionResult.allow()

    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        """
        调用 Skill 的执行逻辑
        
        根据 Skill 类型选择执行方式：
        1. 如果 Skill 有 execute 方法，直接调用
        2. 如果 Skill 有工具，调用第一个匹配的工具
        """
        try:
            # 优先调用 Skill 的 execute 方法
            if hasattr(self.skill, 'execute'):
                # 构建进度回调
                progress_callback = None
                if on_progress:
                    def progress_callback(msg: str):
                        on_progress(msg, 0.5)
                
                # 执行 Skill
                if progress_callback:
                    result = await self.skill.execute(**input_data, on_progress=progress_callback)
                else:
                    result = await self.skill.execute(**input_data)
                
                # 转换结果为 ToolResult
                return self._convert_result(result)
            
            # 如果没有 execute，尝试调用工具
            if self._skill_tools:
                query = input_data.get('query', '')
                for tool in self._skill_tools:
                    try:
                        if hasattr(tool, 'invoke'):
                            result = tool.invoke(query)
                        elif hasattr(tool, 'run'):
                            result = await tool.run(query)
                        else:
                            result = tool(query)
                        return self._convert_result(result)
                    except Exception as e:
                        logger.warning(f"Skill 工具执行失败: {e}")
                        continue
            
            return ToolResult.error_result("Skill 没有可执行的方法或工具")
            
        except Exception as e:
            logger.exception(f"Skill {self.name} 执行失败")
            return ToolResult.error_result(str(e))

    def _convert_result(self, result: Any) -> ToolResult:
        """
        将 Skill 的执行结果转换为 ToolResult
        
        支持多种结果格式：
        - ToolResult: 直接返回
        - dict: 提取 output 和 data
        - str: 作为 output
        - 其他: 转为字符串
        """
        if isinstance(result, ToolResult):
            return result
        
        if isinstance(result, dict):
            output = result.get('output', '')
            if not output and 'result' in result:
                output = str(result['result'])
            data = {k: v for k, v in result.items() if k != 'output'}
            return ToolResult.success_result(output, data)
        
        if isinstance(result, str):
            return ToolResult.success_result(result)
        
        # 其他类型转为字符串
        return ToolResult.success_result(str(result))

    def render_result(self, result: ToolResult) -> str:
        """
        渲染执行结果
        
        如果 Skill 有自定义渲染，调用它。
        否则使用基类默认渲染。
        """
        if hasattr(self.skill, 'render_result'):
            try:
                return self.skill.render_result(result)
            except Exception as e:
                logger.warning(f"Skill 自定义渲染失败: {e}")
        
        return super().render_result(result)

    def to_langchain_tool(self):
        """
        转换为 LangChain 工具
        
        优先返回 Skill 自己的工具（如果有）。
        否则返回适配器包装的工具。
        """
        # 如果 Skill 已经有 LangChain 工具，直接返回第一个
        if self._skill_tools:
            from langchain_core.tools import BaseTool
            for tool in self._skill_tools:
                if isinstance(tool, BaseTool):
                    return tool
        
        # 否则使用适配器
        return super().to_langchain_tool()
