"""
WordAligner 轻量级实现
基于 LangExtract 的 WordAligner 算法移植

提供将提取的实体精确映射回原文位置的能力，支持：
- 精确匹配 (Exact Match)
- 部分匹配 (Lesser Match)  
- 模糊匹配 (Fuzzy Match)
"""

import difflib
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class AlignmentStatus(Enum):
    """对齐状态"""
    MATCH_EXACT = "exact"      # 精确匹配
    MATCH_FUZZY = "fuzzy"      # 模糊匹配
    MATCH_LESSER = "lesser"    # 部分匹配
    UNMATCHED = "unmatched"    # 未匹配


@dataclass
class AlignedEntity:
    """对齐后的实体"""
    name: str                       # 实体名称
    entity_type: str                # 实体类型
    start: int                      # 全文绝对起始位置
    end: int                        # 全文绝对结束位置
    status: AlignmentStatus         # 对齐状态
    similarity: float = 1.0         # 相似度（模糊匹配时）
    chunk_index: int = 0            # 所属块索引
    chunk_relative_start: int = 0   # 块内相对位置
    chunk_relative_end: int = 0     # 块内相对结束位置
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "status": self.status.value,
            "similarity": self.similarity,
            "chunk_index": self.chunk_index,
            "chunk_relative_start": self.chunk_relative_start,
            "chunk_relative_end": self.chunk_relative_end,
        }
    
    def get_status_icon(self) -> str:
        """获取状态图标"""
        icons = {
            AlignmentStatus.MATCH_EXACT: "✓",
            AlignmentStatus.MATCH_LESSER: "~",
            AlignmentStatus.MATCH_FUZZY: "≈",
            AlignmentStatus.UNMATCHED: "✗",
        }
        return icons.get(self.status, "?")
    
    def get_status_desc(self) -> str:
        """获取状态描述"""
        descs = {
            AlignmentStatus.MATCH_EXACT: "精确匹配",
            AlignmentStatus.MATCH_LESSER: "部分匹配",
            AlignmentStatus.MATCH_FUZZY: f"模糊匹配 ({self.similarity:.2f})",
            AlignmentStatus.UNMATCHED: "未匹配",
        }
        return descs.get(self.status, "未知")


class WordAligner:
    """
    文本对齐器 - 将提取的实体精确映射回原文位置
    
    移植自 LangExtract 的 WordAligner，适配 GraphRAG 需求
    
    核心算法:
    1. 精确匹配: 使用 difflib.SequenceMatcher 找完全匹配
    2. 部分匹配: 归一化后匹配（忽略大小写、标点）
    3. 模糊匹配: 滑动窗口 + 相似度阈值兜底
    """
    
    def __init__(self, fuzzy_threshold: float = 0.75):
        """
        初始化对齐器
        
        Args:
            fuzzy_threshold: 模糊匹配阈值（默认0.75）
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.matcher = difflib.SequenceMatcher(autojunk=False)
    
    def align_entities(
        self,
        entities: List[Dict[str, Any]],  # 提取的实体列表
        source_text: str,                # 原始全文
        chunks: List[str],               # 分块后的文本
        chunk_offsets: List[int]         # 每块在全文中的起始位置
    ) -> List[AlignedEntity]:
        """
        将实体对齐到原文位置
        
        Args:
            entities: 实体列表，每个实体包含 name, type, chunk_index, start, end
            source_text: 原始全文
            chunks: 分块列表
            chunk_offsets: 每块在全文中的偏移量
        
        Returns:
            对齐后的实体列表（带全文绝对位置）
        """
        aligned = []
        
        for entity in entities:
            chunk_idx = entity.get('chunk_index', 0)
            
            if chunk_idx >= len(chunks):
                continue
                
            chunk_text = chunks[chunk_idx]
            chunk_offset = chunk_offsets[chunk_idx] if chunk_idx < len(chunk_offsets) else 0
            
            # 提取实体文本
            rel_start = entity.get('start', 0)
            rel_end = entity.get('end', len(chunk_text))
            entity_text = chunk_text[rel_start:rel_end]
            
            if not entity_text.strip():
                continue
            
            # 在原文中查找位置
            result = self._find_in_source(
                entity_text, 
                source_text,
                chunk_offset,
                chunk_text
            )
            
            if result:
                abs_start, abs_end, status, similarity = result
                aligned.append(AlignedEntity(
                    name=entity.get('name', entity_text),
                    entity_type=entity.get('type', 'Unknown'),
                    start=abs_start,
                    end=abs_end,
                    status=status,
                    similarity=similarity,
                    chunk_index=chunk_idx,
                    chunk_relative_start=rel_start,
                    chunk_relative_end=rel_end
                ))
        
        # 去重：相同位置的实体保留置信度高的
        aligned = self._deduplicate(aligned)
        
        return aligned
    
    def _find_in_source(
        self, 
        extraction_text: str, 
        source_text: str,
        search_start: int = 0,
        chunk_text: str = ""
    ) -> Optional[Tuple[int, int, AlignmentStatus, float]]:
        """
        在原文中查找提取文本的位置
        
        匹配策略（按优先级）:
        1. 精确匹配
        2. 归一化匹配（忽略大小写、标点、空格）
        3. 模糊匹配（滑动窗口 + 相似度）
        
        Returns:
            (start, end, status, similarity) 或 None
        """
        # 1. 精确匹配
        exact_pos = source_text.find(extraction_text, search_start)
        if exact_pos != -1:
            return (exact_pos, exact_pos + len(extraction_text), 
                   AlignmentStatus.MATCH_EXACT, 1.0)
        
        # 2. 归一化后匹配（忽略大小写、标点、多余空格）
        norm_result = self._normalized_match(extraction_text, source_text, search_start)
        if norm_result:
            return (*norm_result, AlignmentStatus.MATCH_LESSER, 0.95)
        
        # 3. 模糊匹配（滑动窗口）
        fuzzy_result = self._fuzzy_match(extraction_text, source_text, search_start)
        if fuzzy_result:
            start, end, similarity = fuzzy_result
            return (start, end, AlignmentStatus.MATCH_FUZZY, similarity)
        
        return None
    
    def _normalized_match(
        self,
        extraction_text: str,
        source_text: str,
        search_start: int
    ) -> Optional[Tuple[int, int]]:
        """
        归一化后匹配
        
        处理情况:
        - 大小写差异: "张三" vs "张三"
        - 标点差异: "太阳病，" vs "太阳病"
        - 空格差异: "桂 枝" vs "桂枝"
        """
        norm_extraction = self._normalize_for_match(extraction_text)
        search_text = source_text[search_start:]
        norm_source = self._normalize_for_match(search_text)
        
        norm_pos = norm_source.find(norm_extraction)
        if norm_pos == -1:
            return None
        
        # 将归一化位置映射回原始位置
        abs_pos = search_start + norm_pos
        
        # 尝试在原始文本中找到对应位置
        # 策略：在 abs_pos 附近查找包含 extraction_text 的区域
        window_start = max(0, abs_pos - 5)
        window_end = min(len(source_text), abs_pos + len(norm_extraction) + 5)
        
        # 在这个窗口内找最佳匹配
        best_start = abs_pos
        best_end = min(abs_pos + len(extraction_text), len(source_text))
        
        # 如果实体文本在窗口内，使用实际位置
        for i in range(window_start, window_end - len(extraction_text) + 1):
            substr = source_text[i:i + len(extraction_text)]
            if self._normalize_for_match(substr) == norm_extraction:
                return (i, i + len(extraction_text))
        
        # 如果没找到完全匹配，使用估算位置
        return (best_start, best_end)
    
    def _fuzzy_match(
        self,
        extraction_text: str,
        source_text: str,
        search_start: int
    ) -> Optional[Tuple[int, int, float]]:
        """
        模糊匹配 - 使用滑动窗口找最佳匹配
        
        算法:
        1. 将文本分词
        2. 滑动窗口遍历所有可能位置
        3. 计算每个窗口与提取文本的相似度
        4. 返回超过阈值的最佳匹配
        """
        search_text = source_text[search_start:]
        
        extraction_tokens = self._tokenize(extraction_text)
        source_tokens = self._tokenize(search_text)
        
        if not extraction_tokens or not source_tokens:
            return None
        
        ext_len = len(extraction_tokens)
        max_window = min(ext_len + 3, len(source_tokens))  # 允许一定长度差异
        
        best_ratio = 0.0
        best_start_token = 0
        best_end_token = 0
        
        # 滑动窗口
        for window_size in range(ext_len, max_window + 1):
            for start_token in range(len(source_tokens) - window_size + 1):
                window_tokens = source_tokens[start_token:start_token + window_size]
                
                # 使用 SequenceMatcher 计算相似度
                self.matcher.set_seqs(extraction_tokens, window_tokens)
                ratio = self.matcher.ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start_token = start_token
                    best_end_token = start_token + window_size
        
        if best_ratio >= self.fuzzy_threshold:
            # 将 token 位置映射回字符位置
            char_start = self._token_to_char(best_start_token, source_tokens, search_text)
            char_end = self._token_to_char(best_end_token, source_tokens, search_text)
            
            abs_start = search_start + char_start
            abs_end = search_start + char_end
            
            return (abs_start, abs_end, best_ratio)
        
        return None
    
    def _normalize_for_match(self, text: str) -> str:
        """
        文本归一化用于匹配
        
        处理:
        - 转为小写
        - 移除标点符号
        - 压缩多余空格
        """
        # 转为小写
        text = text.lower()
        # 移除标点符号（保留中英文基本字符）
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text)
        # 压缩多余空格
        text = re.sub(r'\s+', '', text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词
        
        中文：按字符
        英文：按单词
        """
        tokens = []
        # 匹配中文字符或英文单词
        for match in re.finditer(r'[\u4e00-\u9fff]|\w+', text):
            tokens.append(match.group())
        return tokens
    
    def _token_to_char(self, token_idx: int, tokens: List[str], text: str) -> int:
        """将 token 索引转换为字符位置"""
        if token_idx >= len(tokens):
            return len(text)
        
        if token_idx <= 0:
            return 0
        
        # 找到第 token_idx 个 token 的起始位置
        pattern = r'[\u4e00-\u9fff]|\w+'
        matches = list(re.finditer(pattern, text))
        
        if token_idx < len(matches):
            return matches[token_idx].start()
        
        return len(text)
    
    def _deduplicate(self, entities: List[AlignedEntity]) -> List[AlignedEntity]:
        """
        去重：相同位置的实体保留置信度高的
        """
        seen: Dict[Tuple[int, int], AlignedEntity] = {}
        
        for entity in entities:
            key = (entity.start, entity.end)
            
            if key not in seen:
                seen[key] = entity
            else:
                # 保留置信度更高的
                existing = seen[key]
                if entity.similarity > existing.similarity:
                    seen[key] = entity
        
        # 按位置排序
        result = list(seen.values())
        result.sort(key=lambda e: e.start)
        
        return result
    
    def verify_extraction(
        self,
        aligned_entity: AlignedEntity,
        source_text: str
    ) -> Dict[str, Any]:
        """
        验证提取结果
        
        Returns:
            验证报告
        """
        extracted_text = aligned_entity.name
        actual_text = source_text[aligned_entity.start:aligned_entity.end]
        
        # 计算实际相似度
        self.matcher.set_seqs(
            self._tokenize(extracted_text),
            self._tokenize(actual_text)
        )
        actual_similarity = self.matcher.ratio()
        
        return {
            "entity_name": aligned_entity.name,
            "entity_type": aligned_entity.entity_type,
            "position": (aligned_entity.start, aligned_entity.end),
            "extracted_text": extracted_text,
            "actual_text": actual_text,
            "match": extracted_text == actual_text,
            "similarity": actual_similarity,
            "status": aligned_entity.status.value,
            "verification_passed": (
                aligned_entity.status == AlignmentStatus.MATCH_EXACT or
                (aligned_entity.status == AlignmentStatus.MATCH_FUZZY and actual_similarity >= 0.8)
            )
        }
    
    def get_extraction_context(
        self,
        aligned_entity: AlignedEntity,
        source_text: str,
        context_chars: int = 30
    ) -> str:
        """
        获取实体的上下文文本
        
        Args:
            aligned_entity: 对齐后的实体
            source_text: 原文
            context_chars: 上下文字符数
        
        Returns:
            带高亮的上下文文本
        """
        start = max(0, aligned_entity.start - context_chars)
        end = min(len(source_text), aligned_entity.end + context_chars)
        
        before = source_text[start:aligned_entity.start]
        entity = source_text[aligned_entity.start:aligned_entity.end]
        after = source_text[aligned_entity.end:end]
        
        return f"{before}[[{entity}]]{after}"


def format_alignment_report(
    aligned_entities: List[AlignedEntity],
    source_text: str,
    max_display: int = 10
) -> str:
    """
    格式化对齐结果报告
    
    Args:
        aligned_entities: 对齐后的实体列表
        source_text: 原文
        max_display: 最多显示的实体数
    
    Returns:
        格式化的报告字符串
    """
    if not aligned_entities:
        return "  未找到对齐的实体"
    
    lines = []
    lines.append("  ┌" + "─" * 65 + "┐")
    lines.append("  │ {:^63} │".format("📝 实体对齐结果"))
    lines.append("  ├" + "─" * 65 + "┤")
    
    # 统计信息
    exact_count = sum(1 for e in aligned_entities if e.status == AlignmentStatus.MATCH_EXACT)
    fuzzy_count = sum(1 for e in aligned_entities if e.status == AlignmentStatus.MATCH_FUZZY)
    lesser_count = sum(1 for e in aligned_entities if e.status == AlignmentStatus.MATCH_LESSER)
    
    lines.append("  │ {:<61} │".format(
        f"总计: {len(aligned_entities)} | ✓精确: {exact_count} | ~部分: {lesser_count} | ≈模糊: {fuzzy_count}"
    ))
    lines.append("  ├" + "─" * 65 + "┤")
    
    # 实体详情
    for i, entity in enumerate(aligned_entities[:max_display]):
        icon = entity.get_status_icon()
        status_desc = entity.get_status_desc()
        
        # 获取原文片段
        context = source_text[entity.start:entity.end]
        if len(context) > 20:
            context = context[:20] + "..."
        
        lines.append("  │ {:<61} │".format(f"{icon} {entity.name} ({entity.entity_type})"))
        lines.append("  │ {:<61} │".format(f"   位置: 字符 {entity.start}-{entity.end} [{status_desc}]"))
        lines.append("  │ {:<61} │".format(f"   原文: '{context}'"))
        
        if i < min(len(aligned_entities), max_display) - 1:
            lines.append("  │" + " " * 65 + "│")
    
    if len(aligned_entities) > max_display:
        lines.append("  │ {:<61} │".format(f"... 还有 {len(aligned_entities) - max_display} 个实体 ..."))
    
    lines.append("  └" + "─" * 65 + "┘")
    
    return "\n".join(lines)


def calculate_chunk_offsets(full_text: str, chunks: List[str]) -> List[int]:
    """
    计算每块在全文中的起始位置
    
    注意：此函数假设 chunks 是通过对 full_text 分块得到的，
    因此每个 chunk 都应该能在 full_text 中找到。
    
    Args:
        full_text: 全文
        chunks: 分块列表
    
    Returns:
        每块的起始位置列表
    """
    offsets = []
    current_pos = 0
    
    for chunk in chunks:
        if not chunk:
            offsets.append(current_pos)
            continue
        
        # 在全文当前位置后查找块的起始位置
        # 使用块的前30个字符进行定位（避免重叠部分干扰）
        search_key = chunk[:min(30, len(chunk))]
        pos = full_text.find(search_key, current_pos)
        
        if pos == -1:
            # 如果找不到，尝试在整个文本中查找
            pos = full_text.find(search_key)
        
        if pos == -1:
            # 如果还是找不到，使用当前位置
            pos = current_pos
        
        offsets.append(pos)
        # 更新当前位置为这块的结束位置
        current_pos = pos + len(chunk)
    
    return offsets
