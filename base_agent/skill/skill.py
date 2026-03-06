"""
Skill 基类和元数据模型
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class SkillToolConfig(BaseModel):
    """Skill 工具配置 - 只配置额外工具，内置工具自动加载"""
    name: str = Field(..., description="工具名称")
    type: str = Field(default="local", description="工具类型: local/mcp/builtin")
    # local 工具配置
    module: Optional[str] = Field(default=None, description="本地工具模块路径（type=local 时必填）")
    # mcp 工具配置
    server: Optional[str] = Field(default=None, description="MCP 服务器名称（type=mcp 时必填）")


class SkillMetadata(BaseModel):
    """Skill 元数据模型"""
    
    name: str = Field(..., description="Skill 名称，唯一标识")
    description: str = Field(..., description="Skill 描述")
    tags: list[str] = Field(default=[], description="Skill 标签，用于分类和筛选")
    version: str = Field(default="1.0.0", description="Skill 版本")
    author: str = Field(default="", description="作者")
    dependencies: list[str] = Field(default=[], description="依赖的其他 Skill 名称列表")
    triggers: list[str] = Field(default=[], description="触发关键词列表")
    md_file: Optional[Any] = Field(default=None, description="SKILL.md 文件路径")
    tools: list[SkillToolConfig] = Field(default=[], description="Skill 需要的工具配置")
    
    class Config:
        arbitrary_types_allowed = True


class Skill(ABC):
    """
    Skill 抽象基类
    
    所有 Skill 必须继承此类并实现必要的方法。
    Skill 是可独立开发、部署的功能模块。
    
    示例:
        class WeatherSkill(Skill):
            name = "weather"
            description = "查询城市天气信息"
            tags = ["实用工具", "天气"]
            version = "1.0.0"
            
            def get_tools(self) -> list[BaseTool]:
                return [self.get_weather]
            
            @tool
            def get_weather(self, city: str) -> str:
                return f"{city}今天晴，25°C"
    """
    
    # Skill 元数据（类属性，子类应覆盖）
    name: str = ""
    description: str = ""
    tags: list[str] = []
    version: str = "1.0.0"
    author: str = ""
    dependencies: list[str] = []
    
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
            dependencies=self.dependencies,
        )
    
    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """
        返回 Skill 提供的工具列表
        
        Returns:
            list[BaseTool]: 工具列表
        """
        pass
    
    def get_system_prompt(self) -> Optional[str]:
        """
        返回 Skill 的系统提示词
        
        Returns:
            str | None: 系统提示词，None 表示不提供
        """
        return None
    
    def get_examples(self) -> Optional[list[dict]]:
        """
        返回示例对话
        
        Returns:
            list[dict] | None: 示例对话列表，每个元素包含 input 和 output
        """
        return None
    
    def on_activate(self) -> None:
        """
        Skill 被激活时调用
        
        可以在这里进行初始化操作，如加载资源、建立连接等
        """
        pass
    
    def on_deactivate(self) -> None:
        """
        Skill 被停用时调用
        
        可以在这里进行清理操作，如释放资源、关闭连接等
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Skill):
            return False
        return self.name == other.name and self.version == other.version
    
    def __hash__(self) -> int:
        return hash((self.name, self.version))
