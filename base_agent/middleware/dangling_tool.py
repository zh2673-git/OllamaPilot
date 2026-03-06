"""
DanglingToolCallMiddleware - 修复悬空的工具调用

解决小模型在工具调用后"忘记"返回结果的问题
当对话中断或模型输出不完整时，自动修复悬空的工具调用
"""

import json
from typing import Any, Optional
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from .base import AgentMiddleware, AgentState


class DanglingToolCallMiddleware(AgentMiddleware):
    """
    悬空工具调用修复中间件
    
    问题场景：
    1. 用户: "帮我查天气"
    2. AI: 调用 get_weather(city="北京")
    3. 工具返回结果，但AI没有生成最终回复（中断/超时/格式错误）
    4. 用户再次输入时，历史记录中工具调用缺少AI回复
    
    解决方案：
    - 检测悬空的工具调用（有ToolMessage但没有对应的AIMessage回复）
    - 自动注入占位符响应，让对话可以继续
    
    示例:
        middleware = DanglingToolCallMiddleware(
            placeholder_response="我已收到工具执行结果，请继续。"
        )
    """
    
    def __init__(
        self,
        placeholder_response: str = "工具执行完成。",
        auto_summarize: bool = True
    ):
        """
        初始化中间件
        
        Args:
            placeholder_response: 悬空工具调用的占位符回复
            auto_summarize: 是否自动总结工具结果
        """
        self.placeholder_response = placeholder_response
        self.auto_summarize = auto_summarize
        self._fixed_count = 0
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前修复悬空的工具调用
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        messages = list(state.messages)
        if not messages:
            return None
        
        # 检测悬空工具调用
        dangling_calls = self._detect_dangling_calls(messages)
        
        if not dangling_calls:
            return None
        
        # 修复悬空调用
        fixed_messages = self._fix_dangling_calls(messages, dangling_calls)
        
        self._fixed_count += len(dangling_calls)
        
        return {"messages": fixed_messages}
    
    def _detect_dangling_calls(self, messages: list) -> list[dict]:
        """
        检测悬空的工具调用
        
        悬空定义：
        - AI 消息中包含 tool_calls
        - 后面有对应的 ToolMessage 返回结果
        - 但 ToolMessage 之后没有 AI 的最终回复
        
        Args:
            messages: 消息列表
            
        Returns:
            悬空工具调用信息列表
        """
        dangling = []
        pending_calls = {}  # call_id -> tool_name
        
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                # 记录 AI 的工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        call_id = tc.get('id') or tc.get('name', '')
                        if call_id:
                            pending_calls[call_id] = {
                                'index': i,
                                'name': tc.get('name', 'unknown'),
                                'args': tc.get('args', {}),
                                'call_id': call_id
                            }
                # 如果有内容回复，清除之前的 pending calls
                if msg.content and not msg.tool_calls:
                    pending_calls.clear()
                    
            elif isinstance(msg, ToolMessage):
                # 工具返回结果，从 pending 中移除
                call_id = getattr(msg, 'tool_call_id', None) or getattr(msg, 'name', '')
                if call_id in pending_calls:
                    # 检查后面是否有 AI 回复
                    has_ai_response = False
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], AIMessage):
                            has_ai_response = True
                            break
                        elif isinstance(messages[j], HumanMessage):
                            # 遇到用户新输入，说明确实悬空了
                            break
                    
                    if not has_ai_response:
                        dangling.append(pending_calls[call_id])
                    
                    del pending_calls[call_id]
        
        return dangling
    
    def _fix_dangling_calls(
        self, 
        messages: list, 
        dangling_calls: list[dict]
    ) -> list:
        """
        修复悬空的工具调用
        
        Args:
            messages: 原始消息列表
            dangling_calls: 悬空调用信息
            
        Returns:
            修复后的消息列表
        """
        fixed_messages = list(messages)
        
        # 在最后一个 ToolMessage 后插入 AI 回复
        last_tool_idx = -1
        for i in range(len(fixed_messages) - 1, -1, -1):
            if isinstance(fixed_messages[i], ToolMessage):
                last_tool_idx = i
                break
        
        if last_tool_idx >= 0:
            # 生成回复内容
            if self.auto_summarize and len(dangling_calls) > 0:
                content = self._generate_summary(dangling_calls)
            else:
                tool_names = [call['name'] for call in dangling_calls]
                content = f"{self.placeholder_response} 已执行工具: {', '.join(tool_names)}"
            
            # 插入 AI 回复
            ai_response = AIMessage(content=content)
            fixed_messages.insert(last_tool_idx + 1, ai_response)
        
        return fixed_messages
    
    def _generate_summary(self, dangling_calls: list[dict]) -> str:
        """
        生成工具执行总结
        
        Args:
            dangling_calls: 悬空调用列表
            
        Returns:
            总结文本
        """
        if len(dangling_calls) == 1:
            call = dangling_calls[0]
            return f"我已执行 {call['name']} 工具，请继续对话。"
        else:
            tool_names = [call['name'] for call in dangling_calls]
            return f"我已依次执行了以下工具: {', '.join(tool_names)}。请继续对话。"
    
    def get_stats(self) -> dict:
        """获取修复统计"""
        return {
            "fixed_count": self._fixed_count,
            "placeholder_response": self.placeholder_response
        }
