"""
TokenOptimizer - Token 优化器

按三层优先级分配预算：
- 实时层（必须完整保留）
- 工作层（滑动窗口）
- 知识层（按相关性筛选）
"""

from typing import Dict, List

from ollamapilot.context.types import (
    Context,
    ContextPart,
    Layer,
    RuntimeContext,
    WorkingContext,
    KnowledgeContext,
)


class TokenOptimizer:
    """
    Token 优化器 - 按三层优先级分配预算
    
    优先级：实时层 > 工作层 > 知识层
    """
    
    # 默认预算分配比例
    DEFAULT_BUDGET = {
        Layer.RUNTIME: 0.3,    # 30% - 实时层（必须保留）
        Layer.WORKING: 0.4,    # 40% - 工作层（高优先级）
        Layer.KNOWLEDGE: 0.3,  # 30% - 知识层（动态调整）
    }
    
    def __init__(
        self,
        max_tokens: int = 8192,
        budget: Dict[Layer, float] = None,
    ):
        """
        初始化 TokenOptimizer
        
        Args:
            max_tokens: 最大 token 数
            budget: 预算分配比例，默认使用 DEFAULT_BUDGET
        """
        self.max_tokens = max_tokens
        self.budget = budget or self.DEFAULT_BUDGET.copy()
    
    def optimize(self, context: Context) -> Context:
        """
        按优先级优化 Context
        
        策略：
        1. 实时层必须完整保留
        2. 工作层优先保留近期历史
        3. 知识层按相关性筛选
        
        Args:
            context: 原始 Context
            
        Returns:
            优化后的 Context
        """
        optimized_parts: List[ContextPart] = []
        remaining_tokens = self.max_tokens
        
        # 1. 实时层（必须完整保留）
        runtime = context.get_layer(Layer.RUNTIME)
        if runtime:
            optimized_parts.append(runtime)
            remaining_tokens -= runtime.token_count
        
        # 2. 工作层（滑动窗口）
        working = context.get_layer(Layer.WORKING)
        if working and remaining_tokens > 0:
            working_budget = int(self.max_tokens * self.budget[Layer.WORKING])
            working_optimized = self._optimize_working(working, min(working_budget, remaining_tokens))
            if working_optimized.history:
                optimized_parts.append(working_optimized)
                remaining_tokens -= working_optimized.token_count
        
        # 3. 知识层（按相关性筛选）
        knowledge = context.get_layer(Layer.KNOWLEDGE)
        if knowledge and remaining_tokens > 0:
            knowledge_optimized = self._optimize_knowledge(knowledge, remaining_tokens)
            if knowledge_optimized.memories or knowledge_optimized.kb_results:
                optimized_parts.append(knowledge_optimized)
        
        return Context(parts=optimized_parts)
    
    def _optimize_working(self, working: WorkingContext, max_tokens: int) -> WorkingContext:
        """
        优化工作层 - 滑动窗口保留近期历史
        
        Args:
            working: 工作层 Context
            max_tokens: 最大 token 数
            
        Returns:
            优化后的 WorkingContext
        """
        if not working.history:
            return WorkingContext(history=[], intermediate_results=[])
        
        # 从最近的历史开始保留
        optimized_history = []
        current_tokens = 0
        
        for msg in reversed(working.history):
            content = getattr(msg, 'content', str(msg))
            msg_tokens = len(content) * 1.5
            
            if current_tokens + msg_tokens > max_tokens:
                break
            
            optimized_history.insert(0, msg)
            current_tokens += msg_tokens
        
        return WorkingContext(
            history=optimized_history,
            intermediate_results=working.intermediate_results,
        )
    
    def _optimize_knowledge(self, knowledge: KnowledgeContext, max_tokens: int) -> KnowledgeContext:
        """
        优化知识层 - 按优先级筛选
        
        Args:
            knowledge: 知识层 Context
            max_tokens: 最大 token 数
            
        Returns:
            优化后的 KnowledgeContext
        """
        optimized_memories = []
        optimized_kb_results = []
        current_tokens = 0
        
        # 优先保留记忆（通常更相关）
        for memory in knowledge.memories:
            memory_tokens = len(memory) * 1.5
            if current_tokens + memory_tokens > max_tokens:
                break
            optimized_memories.append(memory)
            current_tokens += memory_tokens
        
        # 然后保留知识库结果
        for result in knowledge.kb_results:
            content = result.get('content', '')
            result_tokens = len(content) * 1.5
            if current_tokens + result_tokens > max_tokens:
                break
            optimized_kb_results.append(result)
            current_tokens += result_tokens
        
        return KnowledgeContext(
            memories=optimized_memories,
            kb_results=optimized_kb_results,
        )
    
    def set_budget(self, layer: Layer, ratio: float):
        """
        设置某层的预算比例
        
        Args:
            layer: 层级
            ratio: 比例（0-1之间）
        """
        if 0 <= ratio <= 1:
            self.budget[layer] = ratio
    
    def get_budget(self, layer: Layer) -> float:
        """
        获取某层的预算比例
        
        Args:
            layer: 层级
            
        Returns:
            预算比例
        """
        return self.budget.get(layer, 0.3)
