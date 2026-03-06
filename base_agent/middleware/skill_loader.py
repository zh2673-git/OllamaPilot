"""
Skill Loader 中间件 - LangChain v1+ 兼容版本

在 Agent 执行前根据用户意图自动加载相应的 Skill
"""

from typing import Any, Optional
from langchain_core.messages import SystemMessage
from .base import AgentMiddleware, AgentState
from ..skill import SkillRouter, SkillRegistry


class SkillLoaderMiddleware(AgentMiddleware):
    """
    Skill 加载中间件
    
    在模型调用前根据用户输入自动识别并加载相应的 Skill。
    使用 SkillRouter 进行智能路由决策。
    
    示例:
        middleware = SkillLoaderMiddleware(router, registry)
        
        agent = ModelDrivenAgent(
            model=model,
            skill_router=router,
            skill_registry=registry,
            middleware=[middleware]
        )
    """
    
    def __init__(
        self, 
        skill_router: SkillRouter,
        skill_registry: SkillRegistry
    ):
        """
        初始化中间件
        
        Args:
            skill_router: Skill 路由器，用于决策加载哪些 Skill
            skill_registry: Skill 注册中心，用于获取工具
        """
        self.skill_router = skill_router
        self.skill_registry = skill_registry
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前加载 Skill
        
        分析用户输入，决定加载哪些 Skill，并将工具注入状态。
        
        Args:
            state: 当前状态，包含 messages
            config: 运行配置，可能包含 active_skills 指定强制加载的 Skill
            
        Returns:
            状态更新字典，包含 tools 和 selected_skills
        """
        if not state.messages:
            return None
        
        # 获取最后一条用户消息
        last_message = state.messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # 检查配置中是否指定了强制加载的 Skill
        active_skills = None
        if config and "configurable" in config:
            active_skills = config["configurable"].get("active_skills")
        
        # 决策加载哪些 Skill
        if active_skills:
            # 使用配置指定的 Skill
            selected_skills = active_skills
            if isinstance(selected_skills, str):
                selected_skills = [selected_skills]
        else:
            # 使用 SkillRouter 智能决策
            selected_skills = self.skill_router.decide_skill(user_input)
            if isinstance(selected_skills, str):
                selected_skills = [selected_skills]
        
        if not selected_skills:
            return None
        
        # 获取这些 Skill 的工具
        tools = self.skill_registry.get_tools(selected_skills)
        
        # 构建 Skill 提示词
        skill_prompts = []
        for skill_name in selected_skills:
            metadata = self.skill_router.get_skill_metadata(skill_name)
            if metadata:
                skill_prompts.append(f"- {metadata.name}: {metadata.description}")
        
        # 注入系统提示词
        if skill_prompts:
            system_content = f"""你可以使用以下 Skill 来完成任务：

{chr(10).join(skill_prompts)}

请根据用户需求选择合适的 Skill 工具来完成任务。
"""
            # 检查是否已有系统消息
            has_system = any(
                msg.type == "system" if hasattr(msg, 'type') else False
                for msg in state.messages
            )
            
            if not has_system:
                # 在消息列表开头添加系统消息
                state.messages.insert(0, SystemMessage(content=system_content))
        
        return {
            "messages": state.messages,
            "tools": tools,
            "selected_skills": selected_skills
        }
