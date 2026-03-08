"""
Skill 注册中心

管理 Skill 的注册、发现和加载。
支持 Python Skill 和 Markdown Skill。
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type
from ollamapilot.skills.base import Skill, SkillMetadata
from ollamapilot.skills.loader import load_markdown_skill
from ollamapilot.skills.default_skill import DefaultSkill


class SkillRegistry:
    """
    Skill 注册中心
    
    管理所有可用的 Skill，支持动态发现和加载。
    支持两种格式：
    1. Python Skill (skill.py) - 自定义工具和逻辑
    2. Markdown Skill (SKILL.md) - 纯配置，使用内置工具
    
    示例:
        >>> registry = SkillRegistry()
        >>> registry.discover_skills("skills")
        >>> skills = registry.get_all_skills()
    """
    
    def __init__(self, enable_default_skill: bool = True):
        self._skills: Dict[str, Skill] = {}
        self._skill_classes: Dict[str, Type[Skill]] = {}
        self._markdown_skills: Dict[str, Skill] = {}
        self._default_skill: Optional[DefaultSkill] = None

        # 注册默认 Skill
        if enable_default_skill:
            self._default_skill = DefaultSkill()
    
    def register(self, skill_class: Type[Skill]) -> None:
        """
        注册 Skill 类
        
        Args:
            skill_class: Skill 类（不是实例）
        """
        # 创建临时实例获取元数据
        temp_instance = skill_class()
        name = temp_instance.name
        
        self._skill_classes[name] = skill_class
        print(f"✅ 注册 Skill: {name}")
    
    def register_markdown_skill(self, skill: Skill) -> None:
        """
        注册 Markdown Skill
        
        Args:
            skill: MarkdownSkill 实例
        """
        self._markdown_skills[skill.name] = skill
        print(f"✅ 注册 Markdown Skill: {skill.name}")
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """
        获取 Skill 实例（延迟加载）
        
        Args:
            name: Skill 名称
            
        Returns:
            Skill 实例或 None
        """
        # 检查是否是 Markdown Skill
        if name in self._markdown_skills:
            return self._markdown_skills[name]
        
        # 检查是否已实例化
        if name in self._skills:
            return self._skills[name]
        
        # 否则实例化 Python Skill
        if name in self._skill_classes:
            skill_class = self._skill_classes[name]
            skill_instance = skill_class()
            skill_instance.on_activate()
            self._skills[name] = skill_instance
            return skill_instance
        
        return None
    
    def get_all_skills(self) -> List[Skill]:
        """
        获取所有已加载的 Skill 实例
        
        Returns:
            Skill 实例列表
        """
        # 确保所有已注册的 Python Skill 都已实例化
        for name in self._skill_classes:
            if name not in self._skills:
                self.get_skill(name)
        
        # 合并 Python Skill 和 Markdown Skill
        all_skills = list(self._skills.values()) + list(self._markdown_skills.values())
        return all_skills
    
    def get_all_metadata(self) -> List[SkillMetadata]:
        """
        获取所有 Skill 的元数据（不需要实例化）
        
        Returns:
            元数据列表
        """
        metadata = []
        
        # Python Skill 元数据
        for skill_class in self._skill_classes.values():
            temp = skill_class()
            metadata.append(temp.metadata)
        
        # Markdown Skill 元数据
        for skill in self._markdown_skills.values():
            metadata.append(skill.metadata)
        
        return metadata
    
    def discover_skills(self, skills_dir: str) -> int:
        """
        从目录自动发现 Skill
        
        遍历目录下的所有子目录，查找 skill.py 或 SKILL.md 文件并加载。
        
        Args:
            skills_dir: Skill 目录路径
            
        Returns:
            发现的 Skill 数量
        
        示例目录结构:
            skills/
            ├── web/                      # Python Skill
            │   └── skill.py
            ├── weather/                  # Markdown Skill
            │   └── SKILL.md
            └── python/
                └── skill.py
        """
        skills_path = Path(skills_dir)
        if not skills_path.exists():
            print(f"⚠️ Skill 目录不存在: {skills_dir}")
            return 0
        
        count = 0
        
        for item in skills_path.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                skill_file = item / "skill.py"
                skill_md = item / "SKILL.md"
                
                if skill_file.exists():
                    # 加载 Python Skill
                    try:
                        self._load_skill_module(skill_file, item.name)
                        count += 1
                    except Exception as e:
                        print(f"❌ 加载 Skill '{item.name}' 失败: {e}")
                
                elif skill_md.exists():
                    # 加载 Markdown Skill
                    try:
                        skill = load_markdown_skill(item)
                        if skill:
                            self.register_markdown_skill(skill)
                            count += 1
                    except Exception as e:
                        print(f"❌ 加载 Markdown Skill '{item.name}' 失败: {e}")
        
        return count
    
    def _load_skill_module(self, file_path: Path, module_name: str) -> None:
        """
        加载 Skill 模块
        
        Args:
            file_path: skill.py 文件路径
            module_name: 模块名称
        """
        # 添加 Skill 目录到 Python 路径
        skill_dir = file_path.parent
        if str(skill_dir) not in sys.path:
            sys.path.insert(0, str(skill_dir))
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(
            f"skill_{module_name}", 
            file_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"skill_{module_name}"] = module
        spec.loader.exec_module(module)
        
        # 查找 Skill 类
        found = False
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, Skill) and 
                attr is not Skill):
                self.register(attr)
                found = True
        
        if not found:
            raise ValueError(f"在 {file_path} 中未找到 Skill 类")
    
    def find_skill_by_trigger(self, query: str) -> List[str]:
        """
        根据触发词查找 Skill

        Args:
            query: 用户查询

        Returns:
            匹配的 Skill 名称列表（不包含 default skill）
        """
        matches = []
        query_lower = query.lower()

        # 检查 Python Skill
        for skill_class in self._skill_classes.values():
            temp = skill_class()
            for trigger in temp.triggers:
                if trigger.lower() in query_lower:
                    matches.append(temp.name)
                    break

        # 检查 Markdown Skill
        for skill in self._markdown_skills.values():
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    matches.append(skill.name)
                    break

        return matches

    def get_default_skill(self) -> Optional[DefaultSkill]:
        """
        获取默认 Skill

        Returns:
            DefaultSkill 实例或 None
        """
        return self._default_skill
    
    def get_skill_tools(self, skill_name: str) -> List[str]:
        """
        获取 Skill 需要的工具名称列表
        
        Args:
            skill_name: Skill 名称
            
        Returns:
            工具名称列表
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return []
        
        # 如果是 Markdown Skill，返回配置的工具
        if hasattr(skill, 'get_required_tools'):
            return skill.get_required_tools()
        
        # Python Skill 返回所有工具
        return []
