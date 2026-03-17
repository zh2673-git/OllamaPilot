"""
Context 类型定义

定义 Context 三层架构的所有类型。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field


class Layer(Enum):
    """Context 层级"""
    RUNTIME = "runtime"      # 实时层
    WORKING = "working"      # 工作层
    KNOWLEDGE = "knowledge"  # 知识层


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_tool(cls, tool: Any) -> "ToolDefinition":
        """从 LangChain 工具创建定义"""
        return cls(
            name=getattr(tool, 'name', 'unknown'),
            description=getattr(tool, 'description', ''),
            parameters=getattr(tool, 'args_schema', {})
        )


@dataclass
class Example:
    """示例"""
    input: str
    output: str
    description: Optional[str] = None


@dataclass
class SkillContext:
    """
    Skill Context 片段
    
    Skill 是 Context 的模块化单元，包含：
    - 工具定义 → 告诉模型"你能做什么"
    - 提示词模板 → 告诉模型"怎么做"
    - 示例 → 告诉模型"期望的输出格式"
    - 知识 → Skill 特有的领域知识
    """
    tool_definitions: List[ToolDefinition] = field(default_factory=list)
    system_prompt: Optional[str] = None
    examples: List[Example] = field(default_factory=list)
    knowledge: Optional[str] = None
    
    def to_text(self) -> str:
        """转换为文本格式"""
        parts = []
        
        if self.system_prompt:
            parts.append(f"## 系统提示\n{self.system_prompt}")
        
        if self.tool_definitions:
            parts.append("## 可用工具")
            for tool in self.tool_definitions:
                parts.append(f"- {tool.name}: {tool.description}")
        
        if self.examples:
            parts.append("## 示例")
            for ex in self.examples:
                parts.append(f"输入: {ex.input}")
                parts.append(f"输出: {ex.output}")
                if ex.description:
                    parts.append(f"说明: {ex.description}")
        
        if self.knowledge:
            parts.append(f"## 领域知识\n{self.knowledge}")
        
        return "\n\n".join(parts)


class ContextPart(ABC):
    """Context 片段基类"""
    
    @property
    @abstractmethod
    def layer(self) -> Layer:
        """返回所属层级"""
        pass
    
    @property
    @abstractmethod
    def token_count(self) -> int:
        """估算 token 数量"""
        pass
    
    @abstractmethod
    def to_text(self) -> str:
        """转换为文本"""
        pass


@dataclass
class RuntimeContext(ContextPart):
    """
    实时层 Context
    
    包含当前输入、激活的 Skill、系统状态
    """
    user_input: str
    skill_context: SkillContext
    system_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def layer(self) -> Layer:
        return Layer.RUNTIME
    
    @property
    def token_count(self) -> int:
        # 简单估算：1个汉字 ≈ 1.5 tokens
        total = len(self.user_input) * 1.5
        total += len(self.skill_context.to_text()) * 1.5
        return int(total)
    
    def to_text(self) -> str:
        parts = [
            f"# 用户输入\n{self.user_input}",
            f"# Skill Context\n{self.skill_context.to_text()}",
        ]
        if self.system_state:
            state_text = "\n".join([f"{k}: {v}" for k, v in self.system_state.items()])
            parts.append(f"# 系统状态\n{state_text}")
        return "\n\n".join(parts)


@dataclass
class WorkingContext(ContextPart):
    """
    工作层 Context
    
    包含对话历史、中间执行结果
    """
    history: List[Any] = field(default_factory=list)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def layer(self) -> Layer:
        return Layer.WORKING
    
    @property
    def token_count(self) -> int:
        # 估算历史消息的 token 数
        total = 0
        for msg in self.history:
            content = getattr(msg, 'content', str(msg))
            total += len(content) * 1.5
        return int(total)
    
    def to_text(self) -> str:
        if not self.history:
            return ""
        
        parts = ["# 对话历史"]
        for msg in self.history:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', str(msg))
            parts.append(f"[{role}] {content}")
        
        return "\n".join(parts)


@dataclass
class KnowledgeContext(ContextPart):
    """
    知识层 Context
    
    包含系统记忆、知识库检索结果
    """
    memories: List[str] = field(default_factory=list)
    kb_results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def layer(self) -> Layer:
        return Layer.KNOWLEDGE
    
    @property
    def token_count(self) -> int:
        total = sum(len(m) for m in self.memories) * 1.5
        for result in self.kb_results:
            content = result.get('content', '')
            total += len(content) * 1.5
        return int(total)
    
    def to_text(self) -> str:
        parts = []
        
        if self.memories:
            parts.append("# 相关记忆")
            for i, memory in enumerate(self.memories, 1):
                parts.append(f"{i}. {memory}")
        
        if self.kb_results:
            parts.append("# 知识库检索结果")
            for i, result in enumerate(self.kb_results, 1):
                content = result.get('content', '')
                source = result.get('source', 'unknown')
                parts.append(f"[{i}] 来源: {source}")
                parts.append(content)
        
        return "\n\n".join(parts)


@dataclass
class Context:
    """
    完整 Context
    
    包含所有层级的 Context 片段
    """
    parts: List[ContextPart] = field(default_factory=list)
    
    @property
    def token_count(self) -> int:
        """总 token 数"""
        return sum(part.token_count for part in self.parts)
    
    def to_text(self) -> str:
        """转换为完整文本"""
        return "\n\n".join([part.to_text() for part in self.parts if part.to_text()])
    
    def get_layer(self, layer: Layer) -> Optional[ContextPart]:
        """获取指定层级的 Context"""
        for part in self.parts:
            if part.layer == layer:
                return part
        return None


class ContextLayer(ABC):
    """
    Context 层抽象基类
    
    用于扩展自定义 Context 层
    """
    
    @abstractmethod
    def build(self, query: str, **kwargs) -> ContextPart:
        """构建该层的 Context"""
        pass
    
    @abstractmethod
    def optimize(self, context: ContextPart, max_tokens: int) -> ContextPart:
        """优化该层 Context 以适应 token 限制"""
        pass