"""
Skill 加载器

支持多种格式的 Skill 定义：
1. Python 模块 (skill.py) - 包含自定义工具和逻辑
2. Markdown 配置 (SKILL.md) - 纯配置，使用内置工具

基于 LangChain 1.0+ 设计，优雅简洁。
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from ollamapilot.skills.base import Skill, SkillMetadata


class MarkdownSkill(Skill):
    """
    基于 Markdown 配置的 Skill

    不需要编写 Python 代码，只需配置 SKILL.md 文件。
    支持多种工具来源：内置工具、MCP 工具、自定义工具。

    示例 SKILL.md:
        ---
        name: weather
        description: 查询天气信息
        tags: [实用工具, 天气]
        triggers: [天气, 温度, 下雨]
        tools:
          # 内置工具
          - web_search
          - web_fetch
          # MCP 工具
          - mcp://weather_api/get_current
          - mcp://weather_api/get_forecast
          # 自定义工具（从当前目录加载）
          - custom://local_tool.py:get_weather
        ---

        # 天气查询助手

        你是天气查询专家，帮助用户获取天气信息。

        ## 工作流程
        1. 使用 web_search 搜索城市天气
        2. 使用 web_fetch 获取详细页面
        3. 整理天气信息返回给用户
    """

    # 从配置文件加载的属性
    _config: Dict[str, Any] = {}
    _system_prompt: str = ""
    _allowed_tools: List[str] = []
    _mcp_tools: List[str] = []
    _custom_tools: List[str] = []

    def __init__(self, config: Dict[str, Any], system_prompt: str, skill_dir: Path):
        """
        从配置初始化 Skill

        Args:
            config: YAML 配置字典
            system_prompt: 系统提示词（Markdown 正文）
            skill_dir: Skill 目录路径
        """
        self._config = config
        self._system_prompt = system_prompt
        self._skill_dir = skill_dir

        # 设置类属性
        self.name = config.get("name", "")
        self.description = config.get("description", "")
        self.tags = config.get("tags", [])
        self.version = config.get("version", "1.0.0")
        self.author = config.get("author", "")
        self.triggers = config.get("triggers", [])

        # 解析工具配置
        tools_config = config.get("tools", [])
        self._parse_tools(tools_config)

        # 验证
        if not self.name:
            raise ValueError("SKILL.md 必须包含 name 字段")
        if not self.description:
            raise ValueError("SKILL.md 必须包含 description 字段")

    def _parse_tools(self, tools: List[str]) -> None:
        """
        解析工具配置，分类存储

        支持格式:
        - 内置工具: "web_search", "python_exec"
        - MCP 工具: "mcp://server_name/tool_name"
        - 自定义工具: "custom://file.py:function_name"
        """
        self._allowed_tools = []  # 内置工具
        self._mcp_tools = []      # MCP 工具
        self._custom_tools = []   # 自定义工具

        for tool in tools:
            if tool.startswith("mcp://"):
                self._mcp_tools.append(tool)
            elif tool.startswith("custom://"):
                self._custom_tools.append(tool)
            else:
                # 内置工具
                self._allowed_tools.append(tool)
    
    def get_tools(self) -> List[Any]:
        """
        返回 Skill 需要的工具
        
        对于 Markdown Skill，返回的是工具名称列表，
        Agent 会根据名称从内置工具中筛选。
        """
        # Markdown Skill 不直接提供工具实例
        # 而是返回工具名称，由 Agent 筛选
        return []
    
    def get_system_prompt(self) -> Optional[str]:
        """返回系统提示词"""
        return self._system_prompt
    
    def get_required_tools(self) -> List[str]:
        """返回需要的内置工具名称列表"""
        return self._allowed_tools

    def get_mcp_tools(self) -> List[str]:
        """返回需要的 MCP 工具列表"""
        return self._mcp_tools

    def get_custom_tools(self) -> List[str]:
        """返回需要的自定义工具列表"""
        return self._custom_tools

    def get_all_tool_refs(self) -> Dict[str, List[str]]:
        """
        返回所有工具引用

        Returns:
            {
                "builtin": [...],  # 内置工具
                "mcp": [...],      # MCP 工具
                "custom": [...]    # 自定义工具
            }
        """
        return {
            "builtin": self._allowed_tools,
            "mcp": self._mcp_tools,
            "custom": self._custom_tools,
        }


def parse_skill_md(content: str) -> tuple[Dict[str, Any], str]:
    """
    解析 SKILL.md 文件
    
    Args:
        content: Markdown 文件内容
        
    Returns:
        (config, system_prompt): 配置字典和系统提示词
    """
    # 提取 YAML Front Matter
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("SKILL.md 格式错误：缺少 YAML Front Matter")
    
    yaml_content = match.group(1)
    markdown_content = match.group(2).strip()
    
    # 解析 YAML
    config = yaml.safe_load(yaml_content)
    if not isinstance(config, dict):
        raise ValueError("SKILL.md YAML 配置格式错误")
    
    return config, markdown_content


def load_markdown_skill(skill_dir: Path) -> Optional[MarkdownSkill]:
    """
    从目录加载 Markdown Skill
    
    Args:
        skill_dir: Skill 目录路径
        
    Returns:
        MarkdownSkill 实例或 None
    """
    skill_md = skill_dir / "SKILL.md"
    
    if not skill_md.exists():
        return None
    
    try:
        content = skill_md.read_text(encoding='utf-8')
        config, system_prompt = parse_skill_md(content)
        
        return MarkdownSkill(config, system_prompt, skill_dir)
        
    except Exception as e:
        print(f"❌ 加载 Markdown Skill '{skill_dir.name}' 失败: {e}")
        return None


def discover_skills(skills_dir: str) -> List[Skill]:
    """
    发现所有 Skill
    
    支持格式：
    - skill.py: Python 模块
    - SKILL.md: Markdown 配置
    
    Args:
        skills_dir: Skill 目录路径
        
    Returns:
        Skill 实例列表
    """
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        return []
    
    skills = []
    
    for item in skills_path.iterdir():
        if not item.is_dir() or item.name.startswith("__"):
            continue
        
        # 优先尝试加载 Python Skill
        skill_py = item / "skill.py"
        skill_md = item / "SKILL.md"
        
        if skill_py.exists():
            # Python Skill - 通过 registry 加载
            # 这里只返回 None，实际加载由 SkillRegistry 处理
            pass
        elif skill_md.exists():
            # Markdown Skill
            skill = load_markdown_skill(item)
            if skill:
                skills.append(skill)
                print(f"✅ 加载 Markdown Skill: {skill.name}")
    
    return skills
