"""
Skill Registry 注册中心

统一管理所有 Skill 的注册、发现和加载
"""

import os
import sys
import json
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from .skill import Skill


def simplify_tool_description(tool: BaseTool) -> str:
    """
    为本地小模型简化工具描述
    
    小模型对复杂的 JSON Schema 和长篇描述理解能力有限，
    此函数将工具描述简化为更易理解的格式。
    
    Args:
        tool: 工具实例
        
    Returns:
        简化后的描述字符串
    """
    # 基础信息
    lines = [
        f"工具: {tool.name}",
        f"用途: {tool.description[:150]}"  # 限制描述长度
    ]
    
    # 简化参数描述
    if hasattr(tool, 'args_schema') and tool.args_schema:
        try:
            schema = tool.args_schema.schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            if properties:
                lines.append("参数:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    
                    # 简化类型描述
                    type_map = {
                        'string': '文本',
                        'integer': '整数', 
                        'number': '数字',
                        'boolean': '是/否',
                        'array': '列表',
                        'object': '对象'
                    }
                    simple_type = type_map.get(param_type, param_type)
                    
                    # 标记必需参数
                    is_required = param_name in required
                    required_mark = "(必需)" if is_required else "(可选)"
                    
                    # 参数描述限制长度
                    short_desc = param_desc[:80] if param_desc else "无描述"
                    
                    lines.append(f"  - {param_name}: {simple_type} {required_mark} - {short_desc}")
        except Exception:
            # 如果解析失败，使用简单描述
            lines.append("参数: 见工具定义")
    
    # 添加使用示例
    lines.append("示例:")
    lines.append(f"  {tool.name}(...)")
    
    return "\n".join(lines)


def format_tools_for_small_model(tools: list[BaseTool]) -> str:
    """
    为本地小模型格式化所有工具描述
    
    生成适合小模型理解的工具列表描述，包含：
    - 简化的工具描述
    - 清晰的参数说明
    - 使用示例
    
    Args:
        tools: 工具列表
        
    Returns:
        格式化后的工具描述字符串
    """
    if not tools:
        return "无可用工具"
    
    sections = ["=" * 50, "可用工具列表", "=" * 50]
    
    for i, tool in enumerate(tools, 1):
        sections.append(f"\n【工具 {i}】")
        sections.append(simplify_tool_description(tool))
    
    sections.append("\n" + "=" * 50)
    sections.append("使用说明: 如需使用工具，请明确指定工具名称和参数")
    sections.append("=" * 50)
    
    return "\n".join(sections)


class SkillRegistry:
    """
    Skill 注册中心
    
    统一管理所有 Skill 的注册、发现和加载。
    支持手动注册和自动扫描目录加载。
    
    示例:
        # 创建注册中心
        registry = SkillRegistry()
        
        # 手动注册 Skill
        registry.register(WeatherSkill())
        
        # 自动加载目录中的所有 Skill
        registry.load_from_directory("./skills")
        
        # 获取所有工具
        tools = registry.get_all_tools()
    """
    
    def __init__(self):
        """初始化注册中心"""
        self._skills: dict[str, Skill] = {}
        self._tools_cache: dict[str, list[BaseTool]] = {}
    
    def register(self, skill: Skill) -> None:
        """
        注册 Skill
        
        Args:
            skill: Skill 实例
            
        Raises:
            ValueError: 如果同名 Skill 已存在且版本不同
        """
        name = skill.name
        
        # 检查是否已存在
        if name in self._skills:
            existing = self._skills[name]
            if existing.version != skill.version:
                raise ValueError(
                    f"Skill '{name}' 已存在 (版本 {existing.version})，"
                    f"无法注册版本 {skill.version}"
                )
            # 相同版本，忽略
            return
        
        # 检查依赖
        for dep in skill.dependencies:
            if dep not in self._skills:
                raise ValueError(
                    f"Skill '{name}' 依赖 '{dep}'，但该 Skill 尚未注册"
                )
        
        # 激活 Skill
        skill.on_activate()
        
        # 注册
        self._skills[name] = skill
        
        # 清空工具缓存
        self._tools_cache.clear()
    
    def unregister(self, skill_name: str) -> None:
        """
        注销 Skill
        
        Args:
            skill_name: Skill 名称
            
        Raises:
            KeyError: 如果 Skill 不存在
            ValueError: 如果有其他 Skill 依赖此 Skill
        """
        if skill_name not in self._skills:
            raise KeyError(f"Skill '{skill_name}' 不存在")
        
        # 检查是否有其他 Skill 依赖
        for name, skill in self._skills.items():
            if name != skill_name and skill_name in skill.dependencies:
                raise ValueError(
                    f"无法注销 Skill '{skill_name}'，因为 '{name}' 依赖它"
                )
        
        # 停用 Skill
        self._skills[skill_name].on_deactivate()
        
        # 注销
        del self._skills[skill_name]
        
        # 清空工具缓存
        self._tools_cache.clear()
    
    def get(self, skill_name: str) -> Optional[Skill]:
        """
        获取指定 Skill
        
        Args:
            skill_name: Skill 名称
            
        Returns:
            Skill | None: Skill 实例，不存在则返回 None
        """
        return self._skills.get(skill_name)
    
    def list_skills(self, tag: Optional[str] = None) -> list[Skill]:
        """
        列出所有 Skill
        
        Args:
            tag: 标签筛选，None 表示不过滤
            
        Returns:
            list[Skill]: Skill 列表
        """
        skills = list(self._skills.values())
        
        if tag:
            skills = [s for s in skills if tag in s.tags]
        
        return skills
    
    def get_all_tools(
        self, 
        skill_names: Optional[list[str]] = None
    ) -> list[BaseTool]:
        """
        获取所有工具
        
        Args:
            skill_names: 指定 Skill 列表，None 表示所有已注册 Skill
            
        Returns:
            list[BaseTool]: 工具列表
        """
        cache_key = ",".join(sorted(skill_names)) if skill_names else "__all__"
        
        if cache_key in self._tools_cache:
            return self._tools_cache[cache_key]
        
        tools = []
        
        if skill_names is None:
            # 获取所有 Skill 的工具
            for skill in self._skills.values():
                tools.extend(skill.get_tools())
        else:
            # 获取指定 Skill 的工具
            for name in skill_names:
                if name in self._skills:
                    tools.extend(self._skills[name].get_tools())
        
        self._tools_cache[cache_key] = tools
        return tools
    
    def load_from_directory(self, directory: str) -> None:
        """
        从目录自动加载 Skill
        
        扫描目录中的所有 Python 模块，查找并注册 Skill 类。
        目录结构应为：
            directory/
                skill_name/
                    __init__.py
                    skill.py
        
        Args:
            directory: 目录路径
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是目录: {directory}")
        
        # 遍历目录中的所有子目录
        for item in dir_path.iterdir():
            if not item.is_dir():
                continue
            
            # 检查是否是 Python 包
            init_file = item / "__init__.py"
            skill_file = item / "skill.py"
            
            if not init_file.exists():
                continue
            
            # 尝试加载
            try:
                self._load_skill_package(item)
            except Exception as e:
                print(f"加载 Skill 包 '{item.name}' 失败: {e}")
    
    def _load_skill_package(self, package_path: Path) -> None:
        """
        加载单个 Skill 包
        
        Args:
            package_path: 包路径
        """
        package_name = package_path.name
        
        # 添加到 sys.path
        parent_dir = str(package_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            # 导入包
            spec = importlib.util.spec_from_file_location(
                package_name, 
                package_path / "__init__.py"
            )
            if spec is None or spec.loader is None:
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
            
            # 查找 Skill 类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) 
                    and issubclass(attr, Skill) 
                    and attr is not Skill
                    and not getattr(attr, '_abstract', False)
                ):
                    try:
                        skill_instance = attr()
                        self.register(skill_instance)
                        print(f"已注册 Skill: {skill_instance.name}")
                    except Exception as e:
                        print(f"实例化 Skill '{attr_name}' 失败: {e}")
        
        finally:
            # 从 sys.path 移除
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
    
    def clear(self) -> None:
        """清空所有注册的 Skill"""
        for skill in self._skills.values():
            skill.on_deactivate()
        
        self._skills.clear()
        self._tools_cache.clear()
    
    def __len__(self) -> int:
        """返回已注册的 Skill 数量"""
        return len(self._skills)
    
    def __contains__(self, skill_name: str) -> bool:
        """检查是否包含指定 Skill"""
        return skill_name in self._skills
    
    def __repr__(self) -> str:
        return f"<SkillRegistry(skills={list(self._skills.keys())})>"


def auto_register(registry: SkillRegistry, skills_dir: str = "skills") -> None:
    """
    自动注册指定目录中的所有 Skill
    
    Args:
        registry: Skill 注册中心
        skills_dir: Skill 目录路径
    """
    registry.load_from_directory(skills_dir)
