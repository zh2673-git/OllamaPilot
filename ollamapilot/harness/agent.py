"""
OllamaPilotHarnessAgent - Harness 架构 Agent

核心原则：增强而非替换
- 包装现有的 OllamaPilotAgent
- 使用 LangChain 中间件机制
- 添加三层工具架构
- 保持 100% 向后兼容
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from ollamapilot.harness.middlewares import (
    ContextInjectionMiddleware,
    MemoryRetrievalMiddleware,
    CompactionMiddleware,
    ClarificationMiddleware,
)
from ollamapilot.harness.tools.registry import ToolRegistry
from ollamapilot.harness.tools.builtin import BashTool, FileReadTool, FileWriteTool, TaskTool
from ollamapilot.harness.subagents.factory import SubAgentFactory

logger = logging.getLogger("ollamapilot.harness.agent")


class OllamaPilotHarnessAgent:
    """
    OllamaPilot Harness Agent

    基于 Harness 架构的增强型 Agent，支持：
    1. LangChain 中间件机制
    2. 三层工具架构
    3. 可选沙箱执行
    4. 可选记忆增强

    向后兼容：
    - 现有代码无需修改
    - 渐进式启用新功能
    """

    def __init__(
        self,
        model: BaseChatModel,
        skills_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        enable_memory: bool = True,
        max_tool_calls: int = 50,
        verbose: bool = True,
        # Harness 特有配置
        use_middleware_chain: bool = True,
        use_enhanced_tools: bool = False,
        sandbox_config: Optional[Dict] = None,
        memory_config: Optional[Dict] = None,
        # 现有 Agent 实例（可选）
        existing_agent: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化 Harness Agent

        Args:
            model: 语言模型
            skills_dir: Skill 目录
            workspace_dir: 工作目录
            enable_memory: 是否启用记忆
            max_tool_calls: 最大工具调用次数
            verbose: 是否详细输出
            use_middleware_chain: 是否使用中间件链（LangChain 机制）
            use_enhanced_tools: 是否使用增强型工具架构
            sandbox_config: 沙箱配置（可选）
            memory_config: 记忆增强配置（可选）
            existing_agent: 现有的 OllamaPilotAgent 实例（可选）
        """
        self.model = model
        self.verbose = verbose
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path("./workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.max_tool_calls = max_tool_calls

        # 功能开关
        self._use_middleware_chain = use_middleware_chain
        self._use_enhanced_tools = use_enhanced_tools

        # 初始化或包装现有 Agent
        if existing_agent:
            self._base_agent = existing_agent
        else:
            self._base_agent = self._create_base_agent(
                model, skills_dir, workspace_dir, enable_memory, max_tool_calls, verbose, **kwargs
            )

        # 中间件列表（LangChain 机制）
        self.middlewares: List[Any] = []
        if use_middleware_chain:
            self._init_middlewares()

        # 初始化工具注册表
        self.tool_registry = ToolRegistry()
        if use_enhanced_tools:
            self._init_enhanced_tools()

        # 沙箱配置
        self.sandbox_config = sandbox_config or {}
        self._sandbox = None

        # 记忆增强配置
        self.memory_config = memory_config or {}
        self._enhanced_memory = None

        # 如果使用中间件链，创建新的 Agent
        if use_middleware_chain and self.middlewares:
            self._agent_with_middleware = self._create_agent_with_middleware()
        else:
            self._agent_with_middleware = None

        if verbose:
            print(f"🔧 Harness Agent 初始化完成")
            print(f"   中间件链: {'启用' if use_middleware_chain else '禁用'} ({len(self.middlewares)} 个中间件)")
            print(f"   增强工具: {'启用' if use_enhanced_tools else '禁用'}")

    def _create_base_agent(self, model, skills_dir, workspace_dir, enable_memory, max_tool_calls, verbose, **kwargs):
        """创建基础 Agent"""
        from ollamapilot.agent import create_ollama_agent
        return create_ollama_agent(
            model=model,
            skills_dir=skills_dir,
            workspace_dir=workspace_dir,
            enable_memory=enable_memory,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
            **kwargs
        )

    def _init_middlewares(self):
        """初始化 LangChain 中间件列表"""
        # Context 注入中间件
        if hasattr(self._base_agent, 'context_builder'):
            self.middlewares.append(
                ContextInjectionMiddleware(self._base_agent.context_builder)
            )

        # 记忆检索中间件
        if hasattr(self._base_agent, 'memory_manager'):
            self.middlewares.append(
                MemoryRetrievalMiddleware(self._base_agent.memory_manager)
            )

        # 上下文压缩中间件
        if hasattr(self._base_agent, 'compactor'):
            self.middlewares.append(
                CompactionMiddleware(self._base_agent.compactor)
            )

        # 澄清请求中间件（默认禁用，需要时启用）
        # self.middlewares.append(ClarificationMiddleware(auto_clarify=False))

    def _create_agent_with_middleware(self):
        """创建使用 LangChain 中间件的 Agent"""
        # 合并基础 Agent 的中间件和 Harness 中间件
        all_middlewares = list(self.middlewares)
        
        # 创建新的 Agent，传入中间件
        return create_agent(
            model=self.model.bind_tools(self._base_agent.all_tools),
            tools=self._base_agent.all_tools,
            middleware=all_middlewares,
        )

    def _init_enhanced_tools(self):
        """初始化增强型工具 - 使用适配器模式包装原有工具"""
        from ollamapilot.tools import builtin as builtin_tools
        from ollamapilot.harness.tools.builtin import TaskTool
        from ollamapilot.harness.subagents.factory import SubAgentFactory
        
        # 1. 注册原有内置工具（通过适配器包装）
        builtin_tool_list = [
            builtin_tools.read_file,
            builtin_tools.write_file,
            builtin_tools.shell_exec,
            builtin_tools.shell_script,
            builtin_tools.web_search,
            builtin_tools.web_fetch,
            builtin_tools.web_search_setup,
        ]
        
        for tool_obj in builtin_tool_list:
            try:
                tool_name = getattr(tool_obj, 'name', str(tool_obj))
                self.tool_registry.register_builtin_tool(tool_obj)
            except Exception as e:
                logger.warning(f"注册内置工具 {tool_name} 失败: {e}")
        
        # 2. 注册 TaskTool（Harness 新增功能）
        subagent_factory = SubAgentFactory()
        self.tool_registry.register_tool(TaskTool(subagent_factory))

        # 3. 注册现有 Skill（通过适配器）
        if hasattr(self._base_agent, 'skill_registry'):
            for skill in self._base_agent.skill_registry.get_all_skills():
                try:
                    self.tool_registry.register_skill(skill)
                except Exception as e:
                    logger.warning(f"注册 Skill 失败: {e}")

    def invoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        执行用户查询

        Args:
            query: 用户查询
            thread_id: 会话 ID

        Returns:
            响应文本
        """
        if self.verbose:
            print(f"🤖 用户: {query}")

        # 如果使用中间件链且有中间件
        if self._use_middleware_chain and self._agent_with_middleware:
            return self._invoke_with_langchain_middleware(query, thread_id)
        else:
            # 回退到基础 Agent
            return self._base_agent.invoke(query, thread_id)

    def _invoke_with_langchain_middleware(self, query: str, thread_id: Optional[str] = None) -> str:
        """使用 LangChain 中间件机制执行"""
        try:
            # 使用带有中间件的 Agent 执行
            response = self._agent_with_middleware.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id or "default"}}
            )
            
            # 提取响应内容
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'output' in response:
                return response['output']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.warning(f"使用中间件的 Agent 执行失败: {e}，回退到基础 Agent")
            return self._base_agent.invoke(query, thread_id)

    def get_all_tools(self) -> List[Any]:
        """获取所有工具"""
        if self._use_enhanced_tools:
            return self.tool_registry.get_langchain_tools()
        else:
            return self._base_agent.all_tools

    def get_stats(self) -> Dict[str, Any]:
        """获取 Agent 统计信息"""
        stats = {
            "middlewares": {
                "enabled": self._use_middleware_chain,
                "count": len(self.middlewares),
                "names": [m.name for m in self.middlewares],
            },
            "tools": {
                "enhanced": self._use_enhanced_tools,
            },
        }

        if self._use_enhanced_tools:
            stats["tools"].update(self.tool_registry.get_stats())

        return stats

    # 代理基础 Agent 的方法
    def __getattr__(self, name):
        """代理到基础 Agent"""
        return getattr(self._base_agent, name)


def create_harness_agent(
    model: BaseChatModel,
    skills_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    enable_memory: bool = True,
    max_tool_calls: int = 50,
    verbose: bool = True,
    use_middleware_chain: bool = True,
    use_enhanced_tools: bool = False,
    **kwargs
) -> OllamaPilotHarnessAgent:
    """
    创建 Harness Agent 的工厂函数

    Args:
        model: 语言模型
        skills_dir: Skill 目录
        workspace_dir: 工作目录
        enable_memory: 是否启用记忆
        max_tool_calls: 最大工具调用次数
        verbose: 是否详细输出
        use_middleware_chain: 是否使用中间件链（LangChain 机制）
        use_enhanced_tools: 是否使用增强型工具架构

    Returns:
        OllamaPilotHarnessAgent 实例
    """
    return OllamaPilotHarnessAgent(
        model=model,
        skills_dir=skills_dir,
        workspace_dir=workspace_dir,
        enable_memory=enable_memory,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        use_middleware_chain=use_middleware_chain,
        use_enhanced_tools=use_enhanced_tools,
        **kwargs
    )
