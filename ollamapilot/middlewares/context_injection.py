"""
ContextInjectionMiddleware - Context 注入中间件

在模型调用前，构建四层 Context 并注入到 Agent 状态中。
"""

import time
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, SystemMessage

from ollamapilot.context.builder import ContextBuilder


class ContextInjectionMiddleware(AgentMiddleware):
    """
    Context 注入中间件

    职责：
    1. 调用 ContextBuilder 构建四层 Context
    2. 将 Context 转换为 System Prompt
    3. 注入到 Agent 状态

    执行时机：before_model 钩子
    """

    def __init__(self, context_builder: ContextBuilder):
        self.builder = context_builder

    @property
    def name(self) -> str:
        return "ContextInjectionMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        query = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    query = msg.content
                    break

        if not query:
            return state

        # 从 runtime 或 state 获取 thread_id
        thread_id = "default"
        if runtime and hasattr(runtime, "config") and runtime.config:
            thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")
        elif "thread_id" in state:
            thread_id = state["thread_id"]

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

        return state
