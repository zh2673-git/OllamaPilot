"""
ContextBuilder - 上下文构建器 v2

基于"Context 总纲"理念，协调所有子模块构建最优 Context。
支持四层 Context 架构（L3/L2/L1/L0）。
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ollamapilot.skills import Skill
from ollamapilot.context.types import (
    Context as ContextType,
    ContextPart,
    Layer,
    RuntimeContext,
    WorkingContext,
    KnowledgeContext,
    SkillContext,
)
from ollamapilot.context.optimizer import TokenOptimizer


@dataclass
class Context:
    """Context 四层结构"""
    knowledge: str = ""
    working: str = ""
    realtime: str = ""
    memory: str = ""

    def to_prompt(self) -> str:
        """将四层 Context 合并为 System Prompt"""
        parts = []
        if self.knowledge:
            parts.append(f"[人设与知识]\n{self.knowledge}")
        if self.working:
            parts.append(f"[工作上下文]\n{self.working}")
        if self.realtime:
            parts.append(f"[当前任务]\n{self.realtime}")
        if self.memory:
            parts.append(f"[相关记忆]\n{self.memory}")
        return "\n\n".join(parts)


class ContextBuilder:
    """
    Context 构建器 - 四层 Context 架构

    构建四层 Context：
    - L3 知识层：SOUL/IDENTITY/USER（静态知识）
    - L2 工作层：AGENTS.md + 对话历史
    - L1 实时层：当前用户输入
    - L0 记忆层：MEMORY.md + 语义检索

    性能优化：
    - L3 缓存：TTL + 哈希检测
    - 延迟加载：各层按需构建
    """

    def __init__(
        self,
        workspace_dir: Path = None,
        max_tokens: int = 8192,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        enable_system_memory: bool = False,
        system_memory=None,
        budget: Optional[Dict[str, float]] = None,
    ):
        if workspace_dir is None:
            workspace_dir = Path("./workspace")

        self.workspace = Path(workspace_dir)
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        self.token_optimizer = TokenOptimizer(max_tokens=max_tokens, budget=budget)
        self.system_memory = system_memory
        self.enable_system_memory = enable_system_memory
        self.memory_manager = None

        self._l3_cache: Optional[str] = None
        self._l3_cache_time: float = 0
        self._l3_cache_hash: str = ""

    def build(
        self,
        query: str,
        skill: Skill,
        history: List[Any] = None,
        thread_id: Optional[str] = None,
    ) -> ContextType:
        """
        构建完整上下文（兼容旧接口）

        Args:
            query: 用户查询
            skill: 激活的 Skill
            history: 对话历史
            thread_id: 对话线程 ID

        Returns:
            构建好的 Context
        """
        context_parts: List[ContextPart] = []

        runtime_context = self._build_runtime(query, skill)
        context_parts.append(runtime_context)

        working_context = self._build_working(history or [])
        if working_context.history:
            context_parts.append(working_context)

        if self.enable_system_memory and self.system_memory:
            knowledge_context = self._build_knowledge(query)
            if knowledge_context.memories or knowledge_context.kb_results:
                context_parts.append(knowledge_context)

        context = ContextType(parts=context_parts)
        optimized_context = self.token_optimizer.optimize(context)

        return optimized_context

    def build_four_layer(
        self,
        query: str,
        history: Optional[List[Any]] = None,
        knowledge: bool = True,
        working: bool = True,
        realtime: bool = True,
        memory: bool = True,
    ) -> Context:
        """
        构建四层 Context（优化版本）

        优化点：
        1. L3 缓存：知识层文件不常变化，使用缓存避免重复读取
        2. 延迟加载：各层按需构建，减少不必要的计算
        3. 文件变化检测：通过哈希检测文件变化，自动刷新缓存

        Args:
            query: 当前用户查询
            history: 对话历史
            knowledge: 是否包含知识层
            working: 是否包含工作层
            realtime: 是否包含实时层
            memory: 是否包含记忆层

        Returns:
            Context 对象
        """
        context = Context()

        if knowledge:
            context.knowledge = self._build_knowledge_layer_cached()

        if working:
            context.working = self._build_working_layer(history or [])

        if realtime:
            context.realtime = self._build_realtime_layer(query)

        if memory:
            context.memory = self._build_memory_layer(query)

        return context

    def _build_knowledge_layer_cached(self) -> str:
        """L3: 知识层 - 带缓存的版本"""
        if not self.enable_cache:
            return self._build_knowledge_layer()

        current_time = time.time()
        if self._l3_cache and (current_time - self._l3_cache_time) < self.cache_ttl:
            current_hash = self._calculate_l3_hash()
            if current_hash == self._l3_cache_hash:
                return self._l3_cache

        content = self._build_knowledge_layer()
        self._l3_cache = content
        self._l3_cache_time = current_time
        self._l3_cache_hash = self._calculate_l3_hash()

        return content

    def _calculate_l3_hash(self) -> str:
        """计算 L3 文件的内容哈希"""
        hasher = hashlib.md5()
        for file in ["SOUL.md", "IDENTITY.md", "USER.md"]:
            path = self.workspace / file
            if path.exists():
                hasher.update(path.read_bytes())
        return hasher.hexdigest()

    def _build_knowledge_layer(self) -> str:
        """L3: 知识层 - SOUL/IDENTITY/USER"""
        parts = []
        for file in ["SOUL.md", "IDENTITY.md", "USER.md"]:
            path = self.workspace / file
            if path.exists():
                parts.append(path.read_text(encoding='utf-8'))
        return "\n\n".join(parts)

    def invalidate_cache(self):
        """手动使缓存失效"""
        self._l3_cache = None
        self._l3_cache_time = 0
        self._l3_cache_hash = ""

    def _build_working_layer(self, history: List[Any]) -> str:
        """L2: 工作层 - AGENTS.md + 对话历史"""
        parts = []

        agents_path = self.workspace / "AGENTS.md"
        if agents_path.exists():
            parts.append(agents_path.read_text(encoding='utf-8'))

        if history:
            history_text = self._format_history(history)
            parts.append(f"\n## 对话历史\n{history_text}")

        return "\n\n".join(parts)

    def _format_history(self, history: List[Any]) -> str:
        """格式化对话历史"""
        lines = []
        for msg in history:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', str(msg))
            if content:
                lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _build_realtime_layer(self, query: str) -> str:
        """L1: 实时层 - 当前查询 + 时间信息"""
        from datetime import datetime, timedelta
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_weekday = now.strftime("%A")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        today = now.strftime('%Y-%m-%d')

        time_info = f"""[当前时间]
{current_time} ({current_weekday})
今天是{today}，明天是{tomorrow}，后天是{day_after}，昨天是{yesterday}。

[用户输入]
{query}"""

        return time_info

    def _build_memory_layer(self, query: str) -> str:
        """L0: 记忆层 - 由 Context 统管"""
        if not self.memory_manager:
            return ""

        try:
            memories = self.memory_manager.recall(query, top_k=5)

            if not memories:
                return ""

            memory_parts = ["[相关记忆]"]
            for mem in memories:
                memory_parts.append(f"- {mem}")

            return "\n".join(memory_parts)
        except Exception:
            return ""

    def _build_runtime(self, query: str, skill: Skill) -> RuntimeContext:
        """构建实时层 Context"""
        skill_context = self._skill_to_context(skill)
        system_state = self._get_system_state()

        return RuntimeContext(
            user_input=query,
            skill_context=skill_context,
            system_state=system_state,
            timestamp=datetime.now(),
        )

    def _build_working(self, history: List[Any]) -> WorkingContext:
        """构建工作层 Context"""
        return WorkingContext(
            history=history,
            intermediate_results=[],
        )

    def _build_knowledge(self, query: str) -> KnowledgeContext:
        """构建知识层 Context"""
        memories = []
        kb_results = []

        if self.system_memory:
            try:
                memories = self.system_memory.recall(query, top_k=5)
            except Exception:
                pass

        return KnowledgeContext(
            memories=memories,
            kb_results=kb_results,
        )

    def _skill_to_context(self, skill: Skill) -> SkillContext:
        """将 Skill 转换为 SkillContext"""
        if hasattr(skill, 'to_context') and callable(getattr(skill, 'to_context')):
            try:
                return skill.to_context()
            except Exception:
                pass

        from ollamapilot.context.types import ToolDefinition

        tools = skill.get_tools() if hasattr(skill, 'get_tools') else []
        tool_definitions = [ToolDefinition.from_tool(t) for t in tools]

        system_prompt = None
        if hasattr(skill, 'get_system_prompt'):
            system_prompt = skill.get_system_prompt()

        return SkillContext(
            tool_definitions=tool_definitions,
            system_prompt=system_prompt,
            examples=[],
            knowledge=None,
        )

    def _get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "timestamp": datetime.now().isoformat(),
            "timezone": datetime.now().astimezone().tzname(),
        }

    def set_system_memory(self, system_memory):
        """设置系统记忆"""
        self.system_memory = system_memory
        self.enable_system_memory = True

    def set_memory_manager(self, memory_manager):
        """设置记忆管理器"""
        self.memory_manager = memory_manager

    def enable_memory(self):
        """启用系统记忆"""
        self.enable_system_memory = True

    def disable_memory(self):
        """禁用系统记忆"""
        self.enable_system_memory = False