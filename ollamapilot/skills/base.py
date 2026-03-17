"""
Skill 基类定义

定义 Skill 的标准接口和元数据结构。

Skill 是 Context 的模块化单元，包含：
- 工具定义 → 告诉模型"你能做什么"
- 提示词模板 → 告诉模型"怎么做"
- 示例 → 告诉模型"期望的输出格式"
- 知识 → Skill 特有的领域知识
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool


class SkillMetadata(BaseModel):
    """Skill 元数据模型"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Skill 名称，唯一标识")
    description: str = Field(..., description="Skill 描述")
    tags: List[str] = Field(default=[], description="Skill 标签")
    version: str = Field(default="1.0.0", description="Skill 版本")
    author: str = Field(default="", description="作者")
    triggers: List[str] = Field(default=[], description="触发关键词列表")


class Skill(ABC):
    """
    Skill 抽象基类
    
    所有 Skill 必须继承此类并实现必要的方法。
    Skill 是可独立开发、部署的功能模块。
    
    Skill 是 Context 的模块化单元，不是工具集合，而是结构化的 Context 片段。
    
    示例:
        class WeatherSkill(Skill):
            name = "weather"
            description = "查询城市天气信息"
            tags = ["实用工具", "天气"]
            
            def get_tools(self) -> List[BaseTool]:
                return [get_weather_tool]
    """
    
    # Skill 元数据（类属性，子类应覆盖）
    name: str = ""
    description: str = ""
    tags: List[str] = []
    version: str = "1.0.0"
    author: str = ""
    triggers: List[str] = []
    
    def __init__(self):
        """初始化 Skill，验证必要属性"""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} 必须设置 description 属性")
    
    @property
    def metadata(self) -> SkillMetadata:
        """获取 Skill 元数据"""
        return SkillMetadata(
            name=self.name,
            description=self.description,
            tags=self.tags,
            version=self.version,
            author=self.author,
            triggers=self.triggers,
        )
    
    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """
        返回 Skill 提供的工具列表
        
        Returns:
            List[BaseTool]: 工具列表
        """
        pass
    
    def get_system_prompt(self) -> Optional[str]:
        """
        返回 Skill 的系统提示词
        
        Returns:
            str | None: 系统提示词，None 表示不提供
        """
        return None
    
    def get_middleware(self) -> Optional[Any]:
        """
        返回自定义中间件（可选）

        Skill 可以提供自己的中间件，在模型调用前后执行自定义逻辑。
        例如 GraphRAG Skill 提供检索增强中间件。

        Returns:
            AgentMiddleware 实例或 None

        Example:
            >>> def get_middleware(self) -> Optional[AgentMiddleware]:
            ...     return MyCustomMiddleware()
        """
        return None
    
    def on_activate(self) -> None:
        """Skill 被激活时调用"""
        pass
    
    def on_deactivate(self) -> None:
        """Skill 被停用时调用"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
    
    # ========== Context 集成（新增）==========
    
    def to_context(self) -> "SkillContext":
        """
        将 Skill 转换为 Context 片段
        
        Skill 是 Context 的模块化单元，包含：
        - 工具定义 → 告诉模型"你能做什么"
        - 提示词模板 → 告诉模型"怎么做"
        - 示例 → 告诉模型"期望的输出格式"
        - 知识 → Skill 特有的领域知识
        
        Returns:
            SkillContext: Skill Context 片段
        """
        from ollamapilot.context.types import SkillContext, ToolDefinition, Example
        
        return SkillContext(
            tool_definitions=self.get_tool_definitions(),
            system_prompt=self.get_system_prompt(),
            examples=self.get_examples(),
            knowledge=self.get_knowledge(),
        )
    
    def get_tool_definitions(self) -> List["ToolDefinition"]:
        """
        获取工具定义列表
        
        Returns:
            List[ToolDefinition]: 工具定义列表
        """
        from ollamapilot.context.types import ToolDefinition
        
        tools = self.get_tools()
        return [ToolDefinition.from_tool(t) for t in tools]
    
    def get_examples(self) -> List["Example"]:
        """
        获取示例列表
        
        子类可以覆盖此方法提供示例。
        
        Returns:
            List[Example]: 示例列表
        """
        return []
    
    def get_knowledge(self) -> Optional[str]:
        """
        获取 Skill 特有的领域知识
        
        子类可以覆盖此方法提供领域知识。
        
        Returns:
            str | None: 领域知识，None 表示不提供
        """
        return None
