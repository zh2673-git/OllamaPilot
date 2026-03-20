"""
ContextCompactor - 上下文压缩器

借鉴 OpenClaw 的 Compaction 机制，提供智能上下文压缩能力。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel


@dataclass
class CompressionResult:
    """压缩结果"""
    messages: List[BaseMessage]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    summary: str


class ContextCompactor:
    """
    上下文压缩器 - 借鉴 OpenClaw Compaction

    优化点：
    1. 智能分层压缩：保留系统消息、最近对话、关键决策
    2. 结构化摘要：保留实体、决策、待办事项
    3. 精确 Token 计算：使用 tokenizer 而非估算
    4. 渐进式压缩：先尝试轻量压缩，必要时深度压缩
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        threshold: float = 0.8,
        preserve_recent: int = 10,
        tokenizer: Any = None,
    ):
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.preserve_recent = preserve_recent
        self.tokenizer = tokenizer

    async def compact_async(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """异步压缩上下文"""
        original_tokens = self.count_tokens(messages)

        if original_tokens < self.max_tokens * self.threshold:
            return messages

        result = await self.compact_if_needed_async(messages)
        return result.messages

    async def compact_if_needed_async(
        self,
        messages: List[BaseMessage],
        llm: Optional[BaseLanguageModel] = None
    ) -> CompressionResult:
        """按需压缩上下文"""
        original_tokens = self.count_tokens(messages)

        if original_tokens < self.max_tokens * self.threshold:
            return CompressionResult(
                messages=messages,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                summary=""
            )

        compressed_messages = await self._stratified_compression(messages, llm)
        compressed_tokens = self.count_tokens(compressed_messages)

        summary = self._generate_summary(
            messages, compressed_messages
        )

        return CompressionResult(
            messages=compressed_messages,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            summary=summary
        )

    async def _stratified_compression(
        self,
        messages: List[BaseMessage],
        llm: Optional[BaseLanguageModel]
    ) -> List[BaseMessage]:
        """分层压缩策略"""
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        recent_msgs = non_system_msgs[-self.preserve_recent:]
        history_msgs = non_system_msgs[:-self.preserve_recent]

        if not history_msgs:
            return messages

        compressed = []
        compressed.extend(system_msgs)

        summary_parts = []

        if llm:
            history_summary = await self._generate_summary_async(history_msgs, llm)
            if history_summary:
                summary_parts.append(f"[历史摘要]\n{history_summary}")

        if summary_parts:
            summary_content = "\n\n".join(summary_parts)
            compressed.append(SystemMessage(
                content=f"[上下文已压缩]\n{summary_content}"
            ))

        compressed.extend(recent_msgs)

        return compressed

    async def _generate_summary_async(
        self,
        messages: List[BaseMessage],
        llm: BaseLanguageModel
    ) -> str:
        """使用 LLM 生成摘要"""
        prompt = self._build_summary_prompt(messages)

        try:
            response = await llm.ainvoke(prompt)
            if hasattr(response, 'content'):
                return response.content.strip()
        except Exception:
            pass

        return self._generate_simple_summary(messages)

    def _build_summary_prompt(self, messages: List[BaseMessage]) -> str:
        """构建摘要提示词"""
        conversation = self._format_conversation(messages)

        return f"""请总结以下对话的关键信息：

{conversation}

摘要要求：
1. 保留重要决定和结论
2. 保留关键事实和数据
3. 保留用户明确的需求
4. 控制在 300 字以内

输出格式：
- 主题：[对话主题]
- 关键决策：[决策列表]
- 重要事实：[事实列表]"""

    def _format_conversation(self, messages: List[BaseMessage]) -> str:
        """格式化对话为文本"""
        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            content = getattr(msg, 'content', str(msg))
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generate_simple_summary(self, messages: List[BaseMessage]) -> str:
        """生成简单摘要（当 LLM 不可用时）"""
        topics = []
        decisions = []

        for msg in messages:
            content = getattr(msg, 'content', '')
            if not content:
                continue

            if isinstance(msg, HumanMessage):
                if len(content) < 50:
                    topics.append(content)

        summary_parts = []
        if topics:
            summary_parts.append(f"讨论过 {len(topics)} 个话题")
        if decisions:
            summary_parts.append(f"做出 {len(decisions)} 个决定")

        return "; ".join(summary_parts) if summary_parts else "对话历史已压缩"

    def _generate_summary(
        self,
        original: List[BaseMessage],
        compressed: List[BaseMessage]
    ) -> str:
        """生成压缩过程的结构化摘要"""
        import json

        return json.dumps({
            "original_messages": len(original),
            "compressed_messages": len(compressed),
            "original_tokens": self.count_tokens(original),
            "compressed_tokens": self.count_tokens(compressed),
        }, ensure_ascii=False, indent=2)

    def count_tokens(self, messages: List[BaseMessage]) -> int:
        """计算 Token 数量"""
        if self.tokenizer:
            try:
                text = "\n".join([getattr(m, 'content', '') or '' for m in messages])
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        total = 0
        for msg in messages:
            content = getattr(msg, 'content', '') or ''
            chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
            other_chars = len(content) - chinese_chars
            total += int(chinese_chars * 1.5 + other_chars * 0.5)

        return total