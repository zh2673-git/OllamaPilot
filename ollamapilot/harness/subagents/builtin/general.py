"""
GeneralSubAgent - 通用子 Agent

处理一般性任务，使用 LLM 直接回答
"""

from typing import Any, Dict, Optional

from ollamapilot.harness.subagents.base import SubAgent, SubAgentResult


class GeneralSubAgent(SubAgent):
    """
    通用子 Agent
    
    处理一般性查询和任务，直接使用 LLM 回答。
    
    适用场景：
    - 简单问答
    - 信息查询
    - 文本生成
    """
    
    name = "general"
    description = "通用子 Agent，处理一般性任务"
    
    def __init__(self, model: Any, **kwargs):
        super().__init__(model, **kwargs)
        self.max_tokens = kwargs.get('max_tokens', 2000)
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> SubAgentResult:
        """
        执行通用任务
        
        Args:
            task: 任务描述
            context: 上下文信息
            
        Returns:
            SubAgentResult: 执行结果
        """
        try:
            # 构建提示
            prompt = self._build_prompt(task, context)
            
            # 调用 LLM
            from langchain_core.messages import HumanMessage
            response = self.model.invoke(
                [HumanMessage(content=prompt)],
                max_tokens=self.max_tokens
            )
            
            output = response.content if hasattr(response, 'content') else str(response)
            
            return SubAgentResult.success_result(
                output=output,
                data={"task_type": "general", "task": task}
            )
            
        except Exception as e:
            return SubAgentResult.error_result(
                error=str(e),
                output=f"执行任务时出错: {e}"
            )
    
    def _build_prompt(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """构建提示"""
        prompt_parts = []
        
        # 系统提示
        prompt_parts.append("你是一个专业的AI助手，请认真完成以下任务。")
        
        # 添加上下文
        if context:
            parent_task = context.get('parent_task')
            if parent_task:
                prompt_parts.append(f"\n这是主任务的一部分：{parent_task}")
        
        # 任务
        prompt_parts.append(f"\n任务：{task}")
        prompt_parts.append("\n请提供详细、准确的回答。")
        
        return "\n".join(prompt_parts)
    
    def can_handle(self, task: str) -> bool:
        """通用 Agent 可以处理任何任务"""
        return True
