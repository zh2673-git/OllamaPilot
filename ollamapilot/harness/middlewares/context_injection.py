"""
ContextInjectionMiddleware - Context 注入中间件

包装现有的 ContextBuilder，保留四层 Context 架构！
继承 LangChain AgentMiddleware
"""

from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage


class ContextInjectionMiddleware(AgentMiddleware):
    """
    Context 注入中间件

    包装现有的 ContextBuilder，在模型调用前构建四层 Context 并注入。
    
    四层架构：
    - L3 知识层：SOUL/IDENTITY/USER（静态知识）
    - L2 工作层：AGENTS.md + 对话历史
    - L1 实时层：当前用户输入 + 时间信息
    - L0 记忆层：MEMORY.md + 语义检索
    
    继承 LangChain AgentMiddleware，统一使用 LangChain 中间件机制
    """

    def __init__(self, context_builder: Any):
        self.builder = context_builder

    @property
    def name(self) -> str:
        return "ContextInjectionMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """在模型调用前注入 Context"""
        messages = state.get("messages", [])
        if not messages:
            return state

        # 提取用户查询（支持对象和字典两种格式）
        query = ""
        for msg in reversed(messages):
            # 支持对象格式
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    query = msg.content
                    break
            # 支持字典格式
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    query = msg.get("content", "")
                    break

        if not query:
            return state

        # 从 runtime 或 state 获取 thread_id
        thread_id = "default"
        if runtime and hasattr(runtime, "config") and runtime.config:
            thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")
        elif "thread_id" in state:
            thread_id = state["thread_id"]

        # 调用现有的四层 Context 构建
        try:
            context = self.builder.build_four_layer(
                query=query,
                history=messages,
                knowledge=True,
                working=True,
                realtime=True,
                memory=True,
                thread_id=thread_id
            )
            
            system_prompt = context.to_prompt()
            
            # 注入 System Message
            new_messages = []
            has_system = False
            for msg in messages:
                if isinstance(msg, SystemMessage) and not has_system:
                    new_messages.append(SystemMessage(content=system_prompt))
                    has_system = True
                else:
                    new_messages.append(msg)
            
            if not has_system:
                new_messages.insert(0, SystemMessage(content=system_prompt))
            
            state["messages"] = new_messages
            state["context"] = context
            
        except Exception as e:
            # 构建失败不影响主流程
            import logging
            logging.getLogger("ollamapilot.harness.middlewares").warning(f"Context 构建失败: {e}")

        return state
