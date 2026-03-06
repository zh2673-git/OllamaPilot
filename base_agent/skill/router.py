"""
Skill Router - 模型驱动的 Skill 路由

根据需求文档 2.2.1 基座智能体职责：
- 任务调度: 解析用户意图，决定调用哪些 Skill
- Skill路由: 将请求路由到合适的 Skill 执行
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import re

from .skill import SkillMetadata


class SkillRouter:
    """
    Skill 路由器
    
    职责：
    1. 启动时只加载 Skill 元信息
    2. 让模型根据元信息决定调用哪个 Skill
    3. 按需加载 Skill 的完整内容（SKILL.md）
    """
    
    def __init__(self):
        self._skills_metadata: Dict[str, SkillMetadata] = {}
        self._loaded_skills: Dict[str, Any] = {}  # 缓存已加载的 Skill
    
    def register_metadata(self, metadata: SkillMetadata) -> None:
        """注册 Skill 元数据"""
        self._skills_metadata[metadata.name] = metadata
    
    def get_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """获取 Skill 元数据"""
        return self._skills_metadata.get(skill_name)
    
    def list_metadata(self) -> List[SkillMetadata]:
        """列出所有 Skill 元数据"""
        return list(self._skills_metadata.values())
    
    def get_router_prompt(self) -> str:
        """
        生成路由提示词，让模型决定调用哪个 Skill
        
        根据需求文档，这是任务调度的核心
        """
        skill_list = []
        for meta in self._skills_metadata.values():
            triggers_str = ", ".join(meta.triggers[:3]) if hasattr(meta, 'triggers') and meta.triggers else "无"
            skill_list.append(
                f"- {meta.name}: {meta.description}\n"
                f"  触发词: {triggers_str}"
            )
        
        return f"""你是 Skill 路由决策器。根据用户输入，决定是否需要调用特定 Skill。

## 可用 Skill 列表

{chr(10).join(skill_list)}

## 决策规则

1. 分析用户输入的意图
2. 判断是否需要调用某个 Skill
3. 如果需要，回复格式: [CALL_SKILL: Skill名称]
4. 如果不需要，直接回复用户

## 示例

用户: "帮我查一下北京天气"
分析: 用户需要查询天气，应该调用天气相关 Skill
回复: [CALL_SKILL: weather]

用户: "你好"
分析: 普通问候，不需要调用 Skill
回复: 你好！有什么可以帮助你的吗？

请分析以下用户输入，决定是否需要调用 Skill。"""
    
    def load_skill_content(self, skill_name: str, level: str = "auto") -> Optional[str]:
        """
        按需加载 Skill 的内容（SKILL.md），支持分层加载
        
        Args:
            skill_name: Skill 名称
            level: 加载级别
                - "core": 只加载核心指令（快速执行）
                - "standard": 加载标准内容（平衡）
                - "full": 加载完整内容（最详细）
                - "auto": 自动根据内容长度选择（默认）
        
        分层策略：
        1. Core 层：标题 + 前两个主要章节
        2. Standard 层：标题 + 前四个主要章节
        3. Full 层：完整内容（无限制）
        """
        metadata = self._skills_metadata.get(skill_name)
        if not metadata or not hasattr(metadata, 'md_file') or not metadata.md_file:
            return None
        
        cache_key = f"{skill_name}_{level}"
        
        # 如果已经加载过，直接返回
        if cache_key in self._loaded_skills:
            return self._loaded_skills[cache_key]
        
        try:
            full_content = metadata.md_file.read_text(encoding='utf-8')
            
            # 移除 frontmatter，只保留正文
            if full_content.startswith('---'):
                parts = full_content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
                else:
                    content = full_content
            else:
                content = full_content
            
            # 自动选择级别
            if level == "auto":
                if len(content) > 10000:
                    level = "core"
                elif len(content) > 5000:
                    level = "standard"
                else:
                    level = "full"
            
            # 根据级别截取内容
            if level == "core":
                # Core 层：提取核心部分（标题 + 前两个二级标题）
                extracted = self._extract_core_layer(content)
                result = f"""【核心指令模式】

{extracted}

---
💡 提示：当前使用核心指令模式。如需完整功能，请使用标准模式。
"""
            elif level == "standard":
                # Standard 层：提取到前四个二级标题
                extracted = self._extract_standard_layer(content)
                result = f"""【标准模式】

{extracted}

---
💡 提示：当前使用标准模式。如需查看完整故障排除信息，请使用完整模式。
"""
            else:  # full
                result = content
            
            # 缓存
            self._loaded_skills[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"加载 Skill {skill_name} 失败: {e}")
            return None
    
    def _extract_core_layer(self, content: str) -> str:
        """提取核心层内容（前两个二级标题）"""
        lines = content.split('\n')
        result = []
        h2_count = 0
        
        for line in lines:
            if line.startswith('## '):
                h2_count += 1
                if h2_count > 2:
                    break
            result.append(line)
        
        return '\n'.join(result).strip()
    
    def _extract_standard_layer(self, content: str) -> str:
        """提取标准层内容（前四个二级标题）"""
        lines = content.split('\n')
        result = []
        h2_count = 0
        
        for line in lines:
            if line.startswith('## '):
                h2_count += 1
                if h2_count > 4:
                    break
            result.append(line)
        
        return '\n'.join(result).strip()
    
    def decide_skill(self, user_input: str) -> Optional[str]:
        """
        根据用户输入决定调用哪个 Skill
        
        简单实现：基于关键词匹配
        未来可以改为模型决策
        """
        user_input_lower = user_input.lower()
        
        # 关键词映射
        keyword_map = {
            'browser': ['浏览器', '打开网页', '访问', '浏览'],
            'filesystem': ['文件', '目录', '文件夹', '读取', '保存'],
            'shell': ['命令', '执行', '运行', 'shell', 'bash'],
            'web': ['搜索', '查找', '查询', 'google', '百度'],
            'code': ['代码', '编程', 'python', '分析代码'],
            '自动笔记': ['笔记', '记录', '保存到obsidian', 'obsidian'],
        }
        
        for skill_name, keywords in keyword_map.items():
            if any(kw in user_input_lower for kw in keywords):
                if skill_name in self._skills_metadata:
                    return skill_name
        
        return None


def discover_skills_metadata(skills_dir: str | Path = "skills") -> List[SkillMetadata]:
    """
    自动发现 skills 目录下的所有 Skill 元数据
    
    扫描每个子目录，查找 SKILL.md 或 skill.py，提取元数据
    
    Args:
        skills_dir: Skill 目录路径
        
    Returns:
        List[SkillMetadata]: Skill 元数据列表
    """
    import yaml
    
    skills_dir = Path(skills_dir)
    metadata_list = []
    
    if not skills_dir.exists():
        print(f"⚠️ Skill 目录不存在: {skills_dir}")
        return metadata_list
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        
        skill_name = skill_dir.name
        md_file = skill_dir / "SKILL.md"
        py_file = skill_dir / "skill.py"
        
        try:
            if md_file.exists():
                # 从 SKILL.md 解析元数据
                content = md_file.read_text(encoding='utf-8')
                metadata = _parse_metadata_from_md(content, skill_name)
                metadata.md_file = md_file
                metadata_list.append(metadata)
                print(f"✓ 发现 Skill (Markdown): {skill_name}")
                
            elif py_file.exists():
                # 从 skill.py 提取元数据
                metadata = _extract_metadata_from_py(py_file, skill_name)
                if metadata:
                    metadata_list.append(metadata)
                    print(f"✓ 发现 Skill (Python): {skill_name}")
                    
        except Exception as e:
            print(f"✗ 发现 {skill_name} 失败: {e}")
    
    return metadata_list


def _parse_metadata_from_md(content: str, skill_name: str) -> SkillMetadata:
    """从 Markdown 内容解析元数据"""
    import yaml
    from .skill import SkillToolConfig
    
    # 尝试解析 frontmatter
    frontmatter_pattern = r"^---\n(.*?)\n---"
    match = re.search(frontmatter_pattern, content, re.DOTALL)
    
    metadata_dict = {}
    if match:
        yaml_content = match.group(1)
        try:
            metadata_dict = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            pass
    
    # 解析工具配置
    tools_config = []
    tools_data = metadata_dict.get('tools', [])
    if tools_data:
        for tool_data in tools_data:
            if isinstance(tool_data, dict):
                tools_config.append(SkillToolConfig(**tool_data))
            elif isinstance(tool_data, str):
                # 简写格式：只写工具名，默认为 local 类型
                tools_config.append(SkillToolConfig(name=tool_data, type="local"))
    
    # 构建元数据
    return SkillMetadata(
        name=metadata_dict.get('name', skill_name),
        description=metadata_dict.get('description', f'{skill_name} Skill'),
        tags=metadata_dict.get('tags', []),
        version=metadata_dict.get('version', '1.0.0'),
        author=metadata_dict.get('author', ''),
        triggers=metadata_dict.get('triggers', []),
        tools=tools_config,
    )


def _extract_metadata_from_py(py_file: Path, skill_name: str) -> Optional[SkillMetadata]:
    """从 Python 文件提取元数据"""
    import importlib.util
    
    try:
        spec = importlib.util.spec_from_file_location(f"skills.{skill_name}", py_file)
        if not spec or not spec.loader:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找 Skill 类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                hasattr(attr, 'name') and 
                hasattr(attr, 'description')):
                return SkillMetadata(
                    name=getattr(attr, 'name', skill_name),
                    description=getattr(attr, 'description', ''),
                    tags=getattr(attr, 'tags', []),
                    version=getattr(attr, 'version', '1.0.0'),
                    author=getattr(attr, 'author', ''),
                )
    except Exception:
        pass
    
    return None
