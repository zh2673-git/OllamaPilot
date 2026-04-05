"""
SubAgentFactory - 子 Agent 工厂

管理和创建子 Agent 实例
"""

from typing import Any, Dict, List, Optional, Type
import logging

from ollamapilot.harness.subagents.base import SubAgent, SubAgentResult

logger = logging.getLogger("ollamapilot.harness.subagents")


class SubAgentFactory:
    """
    子 Agent 工厂
    
    管理子 Agent 的注册和创建。
    """
    
    def __init__(self):
        self._agents: Dict[str, Type[SubAgent]] = {}
        self._instances: Dict[str, SubAgent] = {}
    
    def register(self, name: str, agent_class: Type[SubAgent]):
        """
        注册子 Agent 类
        
        Args:
            name: Agent 名称
            agent_class: Agent 类
        """
        self._agents[name] = agent_class
        logger.debug(f"注册子 Agent: {name}")
    
    def create(self, name: str, model: Any, **kwargs) -> Optional[SubAgent]:
        """
        创建子 Agent 实例
        
        Args:
            name: Agent 名称
            model: 语言模型
            
        Returns:
            SubAgent 实例或 None
        """
        if name not in self._agents:
            logger.warning(f"未注册的子 Agent: {name}")
            return None
        
        agent_class = self._agents[name]
        agent = agent_class(model, **kwargs)
        self._instances[name] = agent
        
        return agent
    
    def get(self, name: str) -> Optional[SubAgent]:
        """获取已创建的 Agent 实例"""
        return self._instances.get(name)
    
    def find_agent_for_task(self, task: str) -> Optional[str]:
        """
        查找能处理任务的 Agent
        
        Args:
            task: 任务描述
            
        Returns:
            Agent 名称或 None
        """
        for name, agent in self._instances.items():
            if agent.can_handle(task):
                return name
        return None
    
    async def delegate(self, task: str, model: Any, context: Optional[Dict] = None) -> SubAgentResult:
        """
        委托任务给合适的子 Agent
        
        Args:
            task: 任务描述
            model: 语言模型
            context: 上下文
            
        Returns:
            SubAgentResult: 执行结果
        """
        agent_name = self.find_agent_for_task(task)
        
        if not agent_name:
            return SubAgentResult.error_result("未找到能处理该任务的子 Agent")
        
        agent = self._instances.get(agent_name)
        if not agent:
            agent = self.create(agent_name, model)
        
        if not agent:
            return SubAgentResult.error_result(f"无法创建子 Agent: {agent_name}")
        
        return await agent.execute(task, context)
    
    def list_agents(self) -> List[str]:
        """列出所有注册的 Agent"""
        return list(self._agents.keys())
