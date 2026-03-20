"""
Skill 基类模块

定义 Skill 抽象基类和 Skill 系统核心组件。
包含：Skill、MarkdownSkill、SkillRegistry
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    """

    name: str = "base_skill"
    description: str = "基础 Skill"
    tags: List[str] = []
    triggers: List[str] = []

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """返回 Skill 提供的工具列表"""
        pass

    def get_system_prompt(self) -> Optional[str]:
        """返回 Skill 的系统提示词"""
        return None

    def to_context(self) -> Optional[Dict[str, Any]]:
        """转换为 Context 格式"""
        return None


class DefaultSkill(Skill):
    """默认 Skill"""

    name = "default"
    description = "OllamaPilot 默认 Skill"
    tags = ["默认"]
    triggers = []

    def get_tools(self) -> List[BaseTool]:
        return []


class MarkdownSkill(Skill):
    """
    Markdown Skill - 从 SKILL.md 配置加载

    基于 Markdown 配置文件的 Skill，支持：
    - trigger_keywords: 触发关键词
    - tool_names: 需要的工具列表
    - system_prompt: 系统提示词
    """

    def __init__(
        self,
        name: str,
        description: str,
        triggers: List[str],
        tool_names: List[str],
        system_prompt: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        knowledge: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.triggers = triggers
        self.tool_names = tool_names
        self._system_prompt = system_prompt
        self.examples = examples or []
        self.knowledge = knowledge

    def get_tools(self) -> List[BaseTool]:
        return []

    def get_system_prompt(self) -> Optional[str]:
        return self._system_prompt

    def get_required_tools(self) -> List[str]:
        """获取需要的工具名称列表"""
        return self.tool_names


class SkillRegistry:
    """
    Skill 注册中心

    管理所有可用的 Skill，支持动态发现和加载。
    支持两种格式：
    1. Python Skill (skill.py) - 自定义工具和逻辑
    2. Markdown Skill (SKILL.md) - 纯配置，使用内置工具
    """

    def __init__(self, enable_default_skill: bool = True, skill_config: Optional[Dict[str, Any]] = None):
        self._skills: Dict[str, Skill] = {}
        self._markdown_skills: Dict[str, MarkdownSkill] = {}
        self._default_skill: Optional[DefaultSkill] = None
        self._skill_config: Dict[str, Any] = skill_config or {}

        if enable_default_skill:
            self._default_skill = DefaultSkill()

    def register(self, skill: Skill) -> None:
        """注册 Skill 实例"""
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取 Skill"""
        if name in self._skills:
            return self._skills[name]
        if name in self._markdown_skills:
            return self._markdown_skills[name]
        return None

    def get_all_skills(self) -> List[Skill]:
        """获取所有已注册的 Skill"""
        skills = list(self._skills.values())
        skills.extend(self._markdown_skills.values())
        return skills

    def get_default_skill(self) -> Optional[Skill]:
        """获取默认 Skill"""
        return self._default_skill

    def discover_skills(self, skills_dir: str) -> int:
        """发现并加载 skills 目录下的 Skill"""
        skills_path = Path(skills_dir)
        if not skills_path.exists():
            return 0

        count = 0

        for item in skills_path.iterdir():
            if item.is_dir():
                skill_path = item / "SKILL.md"
                if skill_path.exists():
                    try:
                        md_skill = self._load_markdown_skill(skill_path)
                        if md_skill:
                            self._markdown_skills[md_skill.name] = md_skill
                            count += 1
                    except Exception:
                        pass

        return count

    def _load_markdown_skill(self, skill_path: Path) -> Optional[MarkdownSkill]:
        """加载 Markdown Skill"""
        import re

        content = skill_path.read_text(encoding='utf-8')

        name = skill_path.parent.name
        description = ""
        triggers = []
        tool_names = []
        system_prompt = None

        desc_match = re.search(r'description:\s*(.+)', content)
        if desc_match:
            description = desc_match.group(1).strip()

        trigger_match = re.search(r'trigger_keywords:\s*\[([^\]]+)\]', content)
        if trigger_match:
            triggers = [t.strip() for t in trigger_match.group(1).split(',')]

        tool_match = re.search(r'tool_names:\s*\[([^\]]+)\]', content)
        if tool_match:
            tool_names = [t.strip() for t in tool_match.group(1).split(',')]

        prompt_match = re.search(r'system_prompt:\s*\|?\s*\n([\s\S]*?)(?=\n\w+_names:|$)', content)
        if prompt_match:
            system_prompt = prompt_match.group(1).strip()

        return MarkdownSkill(
            name=name,
            description=description,
            triggers=triggers,
            tool_names=tool_names,
            system_prompt=system_prompt,
        )

    def find_skill_by_trigger(self, query: str) -> List[str]:
        """根据触发词查找 Skill"""
        query_lower = query.lower()
        matches = []

        for skill in self.get_all_skills():
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    matches.append(skill.name)
                    break

        return matches

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有 Skill 提供的工具"""
        tools = []
        for skill in self.get_all_skills():
            try:
                tools.extend(skill.get_tools())
            except Exception:
                pass
        return tools


def load_markdown_skill(skill_path: str) -> Optional[MarkdownSkill]:
    """加载单个 Markdown Skill"""
    path = Path(skill_path)
    if not path.exists():
        return None

    registry = SkillRegistry(enable_default_skill=False)
    return registry._load_markdown_skill(path)
