"""
ContextBuilder - 上下文构建器

基于"Context 总纲"理念，协调所有子模块构建最优 Context。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ollamapilot.skills.base import Skill
from ollamapilot.context.types import (
    Context,
    ContextPart,
    Layer,
    RuntimeContext,
    WorkingContext,
    KnowledgeContext,
    SkillContext,
)
from ollamapilot.context.optimizer import TokenOptimizer


class ContextBuilder:
    """
    Context 构建器 - 所有模块为 Context 服务
    
    构建三层 Context：
    - 实时层（必须）：当前输入 + Skill + 系统状态
    - 工作层（会话级）：对话历史（Checkpoint）
    - 知识层（按需）：系统记忆 + 知识库
    """
    
    def __init__(
        self,
        max_tokens: int = 8192,
        enable_system_memory: bool = False,
        system_memory=None,
        budget: Optional[Dict[str, float]] = None,
    ):
        """
        初始化 ContextBuilder
        
        Args:
            max_tokens: 最大 token 数
            enable_system_memory: 是否启用系统记忆
            system_memory: 系统记忆实例（可选）
            budget: Token 预算分配比例，如 {"runtime": 0.3, "working": 0.4, "knowledge": 0.3}
        """
        self.max_tokens = max_tokens
        self.token_optimizer = TokenOptimizer(max_tokens=max_tokens, budget=budget)
        self.system_memory = system_memory
        self.enable_system_memory = enable_system_memory
    
    def build(
        self,
        query: str,
        skill: Skill,
        history: List[Any] = None,
        thread_id: Optional[str] = None,
    ) -> Context:
        """
        构建完整上下文
        
        Args:
            query: 用户查询
            skill: 激活的 Skill
            history: 对话历史
            thread_id: 对话线程 ID
            
        Returns:
            构建好的 Context
        """
        context_parts: List[ContextPart] = []
        
        # 1. 构建实时层（必须）
        runtime_context = self._build_runtime(query, skill)
        context_parts.append(runtime_context)
        
        # 2. 构建工作层（会话级）
        working_context = self._build_working(history or [])
        if working_context.history:
            context_parts.append(working_context)
        
        # 3. 构建知识层（按需）- 系统记忆
        if self.enable_system_memory and self.system_memory:
            knowledge_context = self._build_knowledge(query)
            if knowledge_context.memories or knowledge_context.kb_results:
                context_parts.append(knowledge_context)
        
        # 4. Token 优化
        context = Context(parts=context_parts)
        optimized_context = self.token_optimizer.optimize(context)
        
        return optimized_context
    
    def _build_runtime(self, query: str, skill: Skill) -> RuntimeContext:
        """
        构建实时层 Context
        
        Args:
            query: 用户查询
            skill: 激活的 Skill
            
        Returns:
            RuntimeContext
        """
        # 获取 Skill 的 Context 片段
        skill_context = self._skill_to_context(skill)
        
        # 获取系统状态
        system_state = self._get_system_state()
        
        return RuntimeContext(
            user_input=query,
            skill_context=skill_context,
            system_state=system_state,
            timestamp=datetime.now(),
        )
    
    def _build_working(self, history: List[Any]) -> WorkingContext:
        """
        构建工作层 Context
        
        Args:
            history: 对话历史
            
        Returns:
            WorkingContext
        """
        return WorkingContext(
            history=history,
            intermediate_results=[],
        )
    
    def _build_knowledge(self, query: str) -> KnowledgeContext:
        """
        构建知识层 Context
        
        Args:
            query: 用户查询
            
        Returns:
            KnowledgeContext
        """
        memories = []
        kb_results = []
        
        # 从系统记忆检索
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
        """
        将 Skill 转换为 SkillContext
        
        Args:
            skill: Skill 实例
            
        Returns:
            SkillContext
        """
        # 如果 Skill 有 to_context 方法，使用它
        if hasattr(skill, 'to_context') and callable(getattr(skill, 'to_context')):
            try:
                return skill.to_context()
            except Exception:
                pass
        
        # 否则，手动构建
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
        """
        获取系统状态
        
        Returns:
            系统状态字典
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "timezone": datetime.now().astimezone().tzname(),
        }
    
    def set_system_memory(self, system_memory):
        """
        设置系统记忆
        
        Args:
            system_memory: 系统记忆实例
        """
        self.system_memory = system_memory
        self.enable_system_memory = True
    
    def enable_memory(self):
        """启用系统记忆"""
        self.enable_system_memory = True
    
    def disable_memory(self):
        """禁用系统记忆"""
        self.enable_system_memory = False
