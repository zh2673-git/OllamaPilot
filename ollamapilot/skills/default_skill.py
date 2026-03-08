"""
默认 Skill

当没有匹配到其他 Skill 时，使用 Default Skill 提供基础能力。
包含所有内置工具，作为 Agent 的"保底"能力。
"""

from typing import List, Optional
from langchain_core.tools import BaseTool
from ollamapilot.skills.base import Skill
from ollamapilot.tools.builtin import (
    read_file, write_file, list_directory, search_files,
    shell_exec, shell_script, python_exec, web_search, web_fetch
)


class DefaultSkill(Skill):
    """
    默认 Skill

    当用户查询没有匹配到其他 Skill 的触发词时，使用此 Skill。
    提供所有内置工具的基础能力。

    特点:
    - 无特定触发词（最低优先级匹配）
    - 包含所有内置工具
    - 通用的系统提示词
    """

    name = "default"
    description = "默认技能，提供基础工具能力"
    tags = ["基础", "默认"]
    version = "1.0.0"
    triggers: List[str] = []  # 无特定触发词

    # 默认系统提示词
    DEFAULT_PROMPT = """你是一个智能助手，帮助用户完成各种任务。

你可以使用以下工具:
- read_file: 读取文件内容
- write_file: 写入文件内容
- list_directory: 列出目录内容
- search_files: 搜索文件内容
- shell_exec: 执行 Shell 命令
- shell_script: 执行 Shell 脚本
- python_exec: 执行 Python 代码
- web_search: 搜索网络信息
- web_fetch: 获取网页内容

重要规则:
1. 理解用户需求后，直接调用合适的工具
2. 如果工具调用失败，尝试其他方法或工具
3. 如果所有方法都失败，诚实告知用户无法完成
4. 完成任务后，给出简洁的结果说明
5. 使用工具时，确保参数正确
"""

    def __init__(self, custom_prompt: Optional[str] = None):
        """
        初始化 Default Skill

        Args:
            custom_prompt: 自定义系统提示词，覆盖默认提示词
        """
        super().__init__()
        self._custom_prompt = custom_prompt

    def get_tools(self) -> List[BaseTool]:
        """
        返回所有内置工具

        Returns:
            内置工具列表
        """
        return [
            read_file,
            write_file,
            list_directory,
            search_files,
            shell_exec,
            shell_script,
            python_exec,
            web_search,
            web_fetch,
        ]

    def get_system_prompt(self) -> Optional[str]:
        """
        返回系统提示词

        Returns:
            系统提示词
        """
        return self._custom_prompt or self.DEFAULT_PROMPT

    def on_activate(self) -> None:
        """Skill 被激活时调用"""
        pass

    def on_deactivate(self) -> None:
        """Skill 被停用时调用"""
        pass
