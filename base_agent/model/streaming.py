"""
流式输出处理器

支持：
1. 思维链实时显示（<think>标签内容）
2. 工具调用过程显示
3. 普通内容流式输出
4. 适配 Ollama 模型
"""

import re
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """内容类型"""
    THINKING = "thinking"      # 思维链
    TOOL_CALL = "tool_call"    # 工具调用
    CONTENT = "content"        # 普通内容
    TOOL_RESULT = "tool_result" # 工具结果


@dataclass
class StreamEvent:
    """流式事件"""
    type: ContentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingProcessor:
    """
    流式输出处理器
    
    解析模型输出，分离思维链、工具调用和普通内容
    支持实时显示和后续处理
    """
    
    def __init__(
        self,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict], None]] = None,
        on_content: Optional[Callable[[str], None]] = None,
        on_tool_result: Optional[Callable[[str], None]] = None,
        show_thinking: bool = True,
        show_tool_calls: bool = True,
    ):
        """
        初始化处理器
        
        Args:
            on_thinking: 思维链回调
            on_tool_call: 工具调用回调
            on_content: 普通内容回调
            on_tool_result: 工具结果回调
            show_thinking: 是否显示思维链
            show_tool_calls: 是否显示工具调用过程
        """
        self.on_thinking = on_thinking or self._default_thinking_handler
        self.on_tool_call = on_tool_call or self._default_tool_call_handler
        self.on_content = on_content or self._default_content_handler
        self.on_tool_result = on_tool_result or self._default_tool_result_handler
        
        self.show_thinking = show_thinking
        self.show_tool_calls = show_tool_calls
        
        # 缓冲区
        self._buffer = ""
        self._thinking_buffer = ""
        self._in_thinking = False
        self._events: list[StreamEvent] = []
    
    def process_chunk(self, chunk: str) -> Optional[StreamEvent]:
        """
        处理一个文本块
        
        Args:
            chunk: 文本块
            
        Returns:
            StreamEvent 或 None
        """
        self._buffer += chunk
        
        # 检查是否是完整的 <think>...</think> 块（一次性传入）
        if not self._in_thinking and "<think>" in self._buffer and "</think>" in self._buffer:
            # 提取思维链内容
            think_match = re.search(r'<think>(.*?)</think>', self._buffer, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1)
                
                # 输出思维链
                if self.show_thinking and thinking_content.strip():
                    event = StreamEvent(
                        ContentType.THINKING,
                        thinking_content,
                        {"complete": True}
                    )
                    self._events.append(event)
                    self.on_thinking(thinking_content)
                
                # 移除思维链部分，保留其他内容
                self._buffer = re.sub(r'<think>.*?</think>', '', self._buffer, flags=re.DOTALL)
                return None
        
        # 检查是否进入思维链（流式情况）
        if not self._in_thinking and "<think>" in self._buffer:
            self._in_thinking = True
            # 提取 <think> 之前的内容
            parts = self._buffer.split("<think>", 1)
            if parts[0]:
                # 输出之前的内容
                event = StreamEvent(ContentType.CONTENT, parts[0])
                self._events.append(event)
                self.on_content(parts[0])
            
            self._buffer = parts[1] if len(parts) > 1 else ""
            return None
        
        # 检查是否结束思维链
        if self._in_thinking and "</think>" in self._buffer:
            parts = self._buffer.split("</think>", 1)
            self._thinking_buffer += parts[0]
            
            # 输出思维链
            if self.show_thinking and self._thinking_buffer.strip():
                event = StreamEvent(
                    ContentType.THINKING, 
                    self._thinking_buffer,
                    {"complete": True}
                )
                self._events.append(event)
                self.on_thinking(self._thinking_buffer)
            
            self._in_thinking = False
            self._thinking_buffer = ""
            self._buffer = parts[1] if len(parts) > 1 else ""
            
            # 继续处理剩余内容
            if self._buffer:
                return self.process_chunk("")
            return None
        
        # 在思维链中
        if self._in_thinking:
            self._thinking_buffer += chunk
            # 实时显示思维链（流式）
            if self.show_thinking:
                event = StreamEvent(
                    ContentType.THINKING,
                    chunk,
                    {"complete": False, "streaming": True}
                )
                self._events.append(event)
                self.on_thinking(chunk)
            return None
        
        # 检查是否包含工具调用
        tool_call = self._try_parse_tool_call(self._buffer)
        if tool_call:
            # 提取工具调用之前的内容
            tool_pattern = self._get_tool_pattern(tool_call)
            match = re.search(tool_pattern, self._buffer, re.IGNORECASE)
            
            if match:
                before_tool = self._buffer[:match.start()]
                if before_tool:
                    event = StreamEvent(ContentType.CONTENT, before_tool)
                    self._events.append(event)
                    self.on_content(before_tool)
                
                # 输出工具调用
                if self.show_tool_calls:
                    event = StreamEvent(
                        ContentType.TOOL_CALL,
                        match.group(0),
                        tool_call
                    )
                    self._events.append(event)
                    self.on_tool_call(tool_call)
                
                self._buffer = self._buffer[match.end():]
                return event
        
        # 普通内容，检查是否可以输出
        # 等待完整的句子或足够的缓冲区
        if len(self._buffer) > 50 or chunk.endswith(('.', '。', '!', '！', '?', '？', '\n')):
            content = self._buffer
            self._buffer = ""
            event = StreamEvent(ContentType.CONTENT, content)
            self._events.append(event)
            self.on_content(content)
            return event
        
        return None
    
    def flush(self) -> list[StreamEvent]:
        """刷新缓冲区，返回所有事件"""
        # 处理剩余内容
        if self._buffer:
            if self._in_thinking:
                # 未闭合的思维链
                if self.show_thinking and self._thinking_buffer.strip():
                    event = StreamEvent(
                        ContentType.THINKING,
                        self._thinking_buffer,
                        {"complete": True, "truncated": True}
                    )
                    self._events.append(event)
                    self.on_thinking(self._thinking_buffer)
            else:
                event = StreamEvent(ContentType.CONTENT, self._buffer)
                self._events.append(event)
                self.on_content(self._buffer)
        
        self._buffer = ""
        self._thinking_buffer = ""
        self._in_thinking = False
        
        return self._events
    
    def _try_parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """尝试解析工具调用"""
        # 匹配各种工具调用格式
        patterns = [
            # JSON 格式
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
            # 函数格式
            r'(\w+)\s*\(\s*([^)]*)\s*\)',
            # 代码块格式
            r'```tool\s*\n(.*?)\n```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                if pattern == patterns[0]:  # JSON
                    return {"tool": match.group(1), "arguments": match.group(2)}
                elif pattern == patterns[1]:  # 函数
                    return {"tool": match.group(1), "arguments": match.group(2)}
                else:  # 代码块
                    return {"raw": match.group(1)}
        
        return None
    
    def _get_tool_pattern(self, tool_call: Dict) -> str:
        """获取工具调用的正则模式"""
        tool = tool_call.get("tool", "")
        return re.escape(tool_call.get("raw", tool))
    
    # 默认处理器
    def _default_thinking_handler(self, content: str):
        """默认思维链处理器"""
        print(f"\n💭 [思考] {content}", end="", flush=True)
    
    def _default_tool_call_handler(self, tool_call: Dict):
        """默认工具调用处理器"""
        tool_name = tool_call.get("tool", "unknown")
        print(f"\n🔧 [工具调用] {tool_name}", flush=True)
    
    def _default_content_handler(self, content: str):
        """默认内容处理器"""
        print(content, end="", flush=True)
    
    def _default_tool_result_handler(self, result: str):
        """默认工具结果处理器"""
        print(f"\n📊 [工具结果] {result[:200]}...", flush=True)


class StreamingDisplay:
    """
    流式显示管理器
    
    管理控制台输出，提供清晰的视觉分隔
    """
    
    def __init__(self, show_thinking: bool = True, show_tools: bool = True):
        self.show_thinking = show_thinking
        self.show_tools = show_tools
        self._thinking_started = False
        self._tool_started = False
    
    def start_thinking(self):
        """开始显示思维链"""
        if self.show_thinking and not self._thinking_started:
            print("\n" + "─" * 50)
            print("💭 思维链:")
            print("─" * 50)
            self._thinking_started = True
    
    def add_thinking_chunk(self, chunk: str):
        """添加思维链片段"""
        if self.show_thinking:
            if not self._thinking_started:
                self.start_thinking()
            print(chunk, end="", flush=True)
    
    def end_thinking(self):
        """结束思维链显示"""
        if self.show_thinking and self._thinking_started:
            print("\n" + "─" * 50)
            self._thinking_started = False
    
    def show_tool_call(self, tool_name: str, arguments: Dict):
        """显示工具调用"""
        if self.show_tools:
            print(f"\n🔧 调用工具: {tool_name}")
            if arguments:
                print(f"   参数: {arguments}")
    
    def show_tool_result(self, result: str):
        """显示工具结果"""
        if self.show_tools:
            print(f"\n📊 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")
    
    def add_content(self, content: str):
        """添加普通内容"""
        # 确保不在思维链中
        if self._thinking_started:
            self.end_thinking()
        print(content, end="", flush=True)


def create_streaming_handler(
    verbose: bool = True,
    show_thinking: bool = True,
    show_tools: bool = True
) -> StreamingProcessor:
    """
    创建流式处理器
    
    Args:
        verbose: 是否显示详细输出
        show_thinking: 是否显示思维链
        show_tools: 是否显示工具调用
        
    Returns:
        StreamingProcessor 实例
    """
    if not verbose:
        # 非详细模式，只显示内容
        return StreamingProcessor(
            on_thinking=lambda x: None,
            on_tool_call=lambda x: None,
            on_content=lambda x: print(x, end="", flush=True),
            on_tool_result=lambda x: None,
            show_thinking=False,
            show_tool_calls=False,
        )
    
    # 创建显示管理器
    display = StreamingDisplay(show_thinking, show_tools)
    
    # 处理器
    def on_thinking(content: str):
        display.add_thinking_chunk(content)
    
    def on_tool_call(tool_call: Dict):
        tool_name = tool_call.get("tool", "unknown")
        arguments = tool_call.get("arguments", {})
        display.show_tool_call(tool_name, arguments)
    
    def on_content(content: str):
        display.add_content(content)
    
    def on_tool_result(result: str):
        display.show_tool_result(result)
    
    return StreamingProcessor(
        on_thinking=on_thinking,
        on_tool_call=on_tool_call,
        on_content=on_content,
        on_tool_result=on_tool_result,
        show_thinking=show_thinking,
        show_tool_calls=show_tools,
    )
