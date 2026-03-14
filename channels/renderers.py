"""
Markdown 渲染器模块

将 Markdown 文本转换为各平台支持的富文本格式。
支持平台：QQ、飞书、钉钉
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RenderedMessage:
    """渲染后的消息"""
    def __init__(self, content: Any, message_type: str = "text"):
        self.content = content
        self.message_type = message_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "message_type": self.message_type
        }


class MarkdownRenderer(ABC):
    """
    Markdown 渲染器基类

    子类需要实现各平台特定的渲染逻辑。
    """

    @abstractmethod
    def render(self, markdown_text: str) -> RenderedMessage:
        """
        将 Markdown 渲染为平台特定的格式

        Args:
            markdown_text: Markdown 格式的文本

        Returns:
            RenderedMessage: 渲染后的消息
        """
        pass

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """提取代码块"""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = []
        for match in re.finditer(pattern, text, re.DOTALL):
            matches.append({
                'language': match.group(1) or 'text',
                'code': match.group(2).strip(),
                'full_match': match.group(0)
            })
        return matches

    def _extract_inline_code(self, text: str) -> List[Dict[str, str]]:
        """提取行内代码"""
        pattern = r'`([^`]+)`'
        matches = []
        for match in re.finditer(pattern, text):
            matches.append({
                'code': match.group(1),
                'full_match': match.group(0)
            })
        return matches


class QQMarkdownRenderer(MarkdownRenderer):
    """
    QQ 富文本渲染器

    将 Markdown 转换为 QQ 支持的富文本格式（JSON 数组）。
    支持样式：粗体、斜体、删除线、颜色

    参考文档：https://bot.q.qq.com/wiki/develop/api/openapi/message/post_messages.html
    """

    def render(self, markdown_text: str) -> RenderedMessage:
        """
        将 Markdown 渲染为 QQ 富文本格式

        Args:
            markdown_text: Markdown 文本

        Returns:
            RenderedMessage: QQ 富文本格式
        """
        if not markdown_text:
            return RenderedMessage([], "rich_text")

        # 处理代码块（先处理，避免内部格式被解析）
        text = self._process_code_blocks(markdown_text)

        # 处理行内代码
        text = self._process_inline_code(text)

        # 处理标题
        text = self._process_headers(text)

        # 处理粗体和斜体
        text = self._process_emphasis(text)

        # 处理删除线
        text = self._process_strikethrough(text)

        # 处理链接
        text = self._process_links(text)

        # 处理列表
        text = self._process_lists(text)

        # 处理引用
        text = self._process_quotes(text)

        # 处理分隔线
        text = self._process_hr(text)

        # 将文本转换为 QQ 富文本格式
        segments = self._text_to_segments(text)

        return RenderedMessage(segments, "rich_text")

    def _process_code_blocks(self, text: str) -> str:
        """处理代码块，转换为带标记的文本"""
        def replace_code_block(match):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            # 代码块用特殊标记包裹，后续转换为 QQ 格式
            return f"【CODE_BLOCK:{language}】{code}【/CODE_BLOCK】"

        return re.sub(r'```(\w+)?\n(.*?)```', replace_code_block, text, flags=re.DOTALL)

    def _process_inline_code(self, text: str) -> str:
        """处理行内代码"""
        def replace_inline_code(match):
            code = match.group(1)
            return f"【INLINE_CODE】{code}【/INLINE_CODE】"

        return re.sub(r'`([^`]+)`', replace_inline_code, text)

    def _process_headers(self, text: str) -> str:
        """处理标题"""
        # H1: # 标题
        text = re.sub(r'^# (.+)$', r'【H1】\1【/H1】', text, flags=re.MULTILINE)
        # H2: ## 标题
        text = re.sub(r'^## (.+)$', r'【H2】\1【/H2】', text, flags=re.MULTILINE)
        # H3-H6: ### 标题
        text = re.sub(r'^###+ (.+)$', r'【H3】\1【/H3】', text, flags=re.MULTILINE)
        return text

    def _process_emphasis(self, text: str) -> str:
        """处理粗体和斜体"""
        # 粗体: **text** 或 __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'【BOLD】\1【/BOLD】', text)
        text = re.sub(r'__(.+?)__', r'【BOLD】\1【/BOLD】', text)
        # 斜体: *text* 或 _text_
        text = re.sub(r'\*(.+?)\*', r'【ITALIC】\1【/ITALIC】', text)
        text = re.sub(r'_(.+?)_', r'【ITALIC】\1【/ITALIC】', text)
        return text

    def _process_strikethrough(self, text: str) -> str:
        """处理删除线"""
        return re.sub(r'~~(.+?)~~', r'【STRIKE】\1【/STRIKE】', text)

    def _process_links(self, text: str) -> str:
        """处理链接"""
        # [text](url)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'【LINK:\2】\1【/LINK】', text)

    def _process_lists(self, text: str) -> str:
        """处理列表"""
        # 无序列表
        text = re.sub(r'^\s*[-*+] (.+)$', r'• \1', text, flags=re.MULTILINE)
        # 有序列表
        text = re.sub(r'^\s*\d+\. (.+)$', r'\1', text, flags=re.MULTILINE)
        return text

    def _process_quotes(self, text: str) -> str:
        """处理引用"""
        lines = text.split('\n')
        result = []
        for line in lines:
            if line.strip().startswith('>'):
                content = line.strip()[1:].strip()
                result.append(f'【QUOTE】{content}【/QUOTE】')
            else:
                result.append(line)
        return '\n'.join(result)

    def _process_hr(self, text: str) -> str:
        """处理分隔线"""
        return re.sub(r'^\s*[-*_]{3,}\s*$', '【HR】', text, flags=re.MULTILINE)

    def _text_to_segments(self, text: str) -> List[Dict[str, Any]]:
        """将处理后的文本转换为 QQ 富文本段"""
        segments = []
        current_text = ""
        i = 0

        while i < len(text):
            # 查找标记
            marker_match = re.search(r'【(\w+)(?::([^】]+))?】', text[i:])

            if not marker_match or marker_match.start() + i > len(text):
                # 没有更多标记，添加剩余文本
                if i < len(text):
                    remaining = text[i:]
                    if remaining:
                        segments.append({"type": "text", "text": remaining})
                break

            # 添加标记前的文本
            if marker_match.start() > 0:
                current_text = text[i:i + marker_match.start()]
                if current_text:
                    segments.append({"type": "text", "text": current_text})

            # 处理标记
            marker_type = marker_match.group(1)
            marker_value = marker_match.group(2) if marker_match.group(2) else None

            # 找到结束标记
            end_marker = f'【/{marker_type}】'
            end_pos = text.find(end_marker, i + marker_match.end())

            if end_pos == -1:
                # 没有结束标记，作为普通文本
                segments.append({"type": "text", "text": marker_match.group(0)})
                i += marker_match.end()
                continue

            # 提取内容
            content = text[i + marker_match.end():end_pos]

            # 根据标记类型创建段
            segment = self._create_segment(marker_type, content, marker_value)
            if segment:
                segments.append(segment)

            i = end_pos + len(end_marker)

        # 合并连续的纯文本段
        segments = self._merge_text_segments(segments)

        return segments

    def _create_segment(self, marker_type: str, content: str, value: str = None) -> Optional[Dict[str, Any]]:
        """根据标记类型创建富文本段"""
        if marker_type == "BOLD":
            return {
                "type": "text",
                "text": content,
                "styles": {"bold": True}
            }
        elif marker_type == "ITALIC":
            return {
                "type": "text",
                "text": content,
                "styles": {"italic": True}
            }
        elif marker_type == "STRIKE":
            return {
                "type": "text",
                "text": content,
                "styles": {"strikethrough": True}
            }
        elif marker_type in ["H1", "H2", "H3"]:
            # 标题：加粗 + 换行
            return {
                "type": "text",
                "text": content + "\n",
                "styles": {"bold": True}
            }
        elif marker_type == "CODE_BLOCK":
            # 代码块
            return {
                "type": "text",
                "text": f"```\n{content}\n```\n"
            }
        elif marker_type == "INLINE_CODE":
            # 行内代码
            return {
                "type": "text",
                "text": content,
                "styles": {"bold": True}
            }
        elif marker_type == "LINK":
            # 链接
            return {
                "type": "text",
                "text": content,
                "styles": {"underline": True}
            }
        elif marker_type == "QUOTE":
            # 引用
            return {
                "type": "text",
                "text": f"> {content}\n"
            }
        elif marker_type == "HR":
            # 分隔线
            return {
                "type": "text",
                "text": "\n─────────\n"
            }
        else:
            # 未知标记，作为普通文本
            return {"type": "text", "text": content}

    def _merge_text_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并连续的纯文本段"""
        if not segments:
            return segments

        merged = []
        current = segments[0].copy()

        for segment in segments[1:]:
            if (current.get("type") == "text" and
                segment.get("type") == "text" and
                current.get("styles") == segment.get("styles")):
                # 合并文本
                current["text"] = current.get("text", "") + segment.get("text", "")
            else:
                merged.append(current)
                current = segment.copy()

        merged.append(current)
        return merged


class PlainTextRenderer(MarkdownRenderer):
    """
    纯文本渲染器

    将 Markdown 转换为纯文本，移除所有格式标记。
    用于不支持富文本的平台。
    """

    def render(self, markdown_text: str) -> RenderedMessage:
        """将 Markdown 转换为纯文本"""
        if not markdown_text:
            return RenderedMessage("", "text")

        text = markdown_text

        # 移除代码块标记
        text = re.sub(r'```\w*\n(.*?)```', r'\1', text, flags=re.DOTALL)

        # 移除行内代码标记
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # 移除标题标记
        text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)

        # 移除粗体和斜体标记
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # 移除删除线标记
        text = re.sub(r'~~(.+?)~~', r'\1', text)

        # 转换链接为文本 (url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)

        # 移除引用标记
        text = re.sub(r'^> ', '', text, flags=re.MULTILINE)

        # 移除分隔线
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        return RenderedMessage(text.strip(), "text")


class QQPlainTextRenderer(MarkdownRenderer):
    """
    QQ 纯文本优化渲染器

    将 Markdown 转换为适合 QQ 显示的纯文本格式。
    保留一些视觉格式（如标题加粗符号、代码块标记等），提升可读性。
    """

    def render(self, markdown_text: str) -> RenderedMessage:
        """将 Markdown 转换为 QQ 优化的纯文本"""
        if not markdown_text:
            return RenderedMessage("", "text")

        text = markdown_text

        # 处理标题 - 保留并加粗
        text = re.sub(r'^# (.+)$', r'【\1】', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'【\1】', text, flags=re.MULTILINE)
        text = re.sub(r'^###+ (.+)$', r'【\1】', text, flags=re.MULTILINE)

        # 处理粗体 - 保留内容，用【】包裹表示强调
        text = re.sub(r'\*\*(.+?)\*\*', r'【\1】', text)
        text = re.sub(r'__(.+?)__', r'【\1】', text)

        # 处理斜体 - 保留内容
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # 处理删除线 - 保留内容，加删除标记
        text = re.sub(r'~~(.+?)~~', r'[删除:\1]', text)

        # 处理代码块 - 保留格式
        def format_code_block(match):
            language = match.group(1) or '代码'
            code = match.group(2).strip()
            return f"\n━━━━━━【{language}】━━━━━━\n{code}\n━━━━━━━━━━━━━━━━\n"

        text = re.sub(r'```(\w+)?\n(.*?)```', format_code_block, text, flags=re.DOTALL)

        # 处理行内代码 - 用【】包裹
        text = re.sub(r'`([^`]+)`', r'【\1】', text)

        # 处理链接 - 显示为 文本(链接)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1(\2)', text)

        # 处理无序列表 - 使用 • 符号
        text = re.sub(r'^\s*[-*+] (.+)$', r'• \1', text, flags=re.MULTILINE)

        # 处理有序列表 - 保留数字
        text = re.sub(r'^\s*(\d+)\. (.+)$', r'\1. \2', text, flags=re.MULTILINE)

        # 处理引用 - 保留 > 符号
        text = re.sub(r'^> (.+)$', r'> \1', text, flags=re.MULTILINE)

        # 处理分隔线
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '\n━━━━━━━━━━━━\n', text, flags=re.MULTILINE)

        # 清理多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        return RenderedMessage(text.strip(), "text")


def get_renderer(platform: str) -> MarkdownRenderer:
    """
    获取指定平台的渲染器

    Args:
        platform: 平台名称 (qq, feishu, dingtalk, plain)

    Returns:
        MarkdownRenderer: 对应平台的渲染器
    """
    renderers = {
        "qq": QQPlainTextRenderer(),  # QQ 使用纯文本优化版
        "feishu": PlainTextRenderer(),  # 飞书可以后续实现卡片消息
        "dingtalk": PlainTextRenderer(),  # 钉钉可以后续实现 Markdown 消息
        "plain": PlainTextRenderer(),
    }

    return renderers.get(platform.lower(), PlainTextRenderer())
