"""
Skill 分块处理器

针对超大 Skill（4万字+）的解决方案：
1. 将 Skill 内容分块索引
2. 根据用户请求检索最相关的块
3. 只加载相关部分给模型

类似 RAG，但针对 Skill 执行优化
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillChunk:
    """Skill 内容块"""
    id: str
    content: str
    section: str  # 所属章节
    index: int    # 块序号
    keywords: List[str]  # 提取的关键词
    char_count: int


@dataclass
class ChunkMatch:
    """匹配结果"""
    chunk: SkillChunk
    score: float  # 匹配分数
    matched_keywords: List[str]


class SkillChunker:
    """
    Skill 分块管理器
    
    将超大 Skill 分割成小块，支持：
    - 智能分块（按章节、语义）
    - 关键词索引
    - 相关性检索
    """
    
    # 小模型上下文限制
    MAX_CHUNK_SIZE = 1500  # 每个块最大字符数
    OVERLAP_SIZE = 200     # 块之间重叠字符数（保持上下文）
    
    def __init__(self):
        self._chunks: Dict[str, List[SkillChunk]] = {}  # skill_name -> chunks
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> [chunk_ids]
    
    def process_skill(self, skill_name: str, content: str) -> List[SkillChunk]:
        """
        处理 Skill 内容，分块并建立索引
        
        Args:
            skill_name: Skill 名称
            content: 完整 Skill 内容
            
        Returns:
            分块列表
        """
        # 1. 按章节分割
        sections = self._split_by_sections(content)
        
        # 2. 每个章节再分块
        chunks = []
        chunk_index = 0
        
        for section_title, section_content in sections:
            section_chunks = self._chunk_section(
                section_title, 
                section_content, 
                chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # 3. 建立索引
        self._chunks[skill_name] = chunks
        self._build_index(skill_name, chunks)
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """按二级标题分割成章节"""
        # 匹配 ## 标题
        pattern = r'\n##\s+(.+?)\n'
        matches = list(re.finditer(pattern, content))
        
        if not matches:
            # 没有二级标题，整体作为一个章节
            return [("概述", content)]
        
        sections = []
        
        # 第一个标题之前的内容作为概述
        if matches[0].start() > 0:
            intro = content[:matches[0].start()].strip()
            if intro:
                sections.append(("概述", intro))
        
        # 提取每个章节
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            
            if section_content:
                sections.append((title, section_content))
        
        return sections
    
    def _chunk_section(
        self, 
        section_title: str, 
        section_content: str,
        start_index: int
    ) -> List[SkillChunk]:
        """将章节内容分块"""
        chunks = []
        
        # 如果内容较短，直接作为一个块
        if len(section_content) <= self.MAX_CHUNK_SIZE:
            chunk_id = self._generate_chunk_id(section_title, 0)
            keywords = self._extract_keywords(section_content)
            
            chunk = SkillChunk(
                id=chunk_id,
                content=section_content,
                section=section_title,
                index=start_index,
                keywords=keywords,
                char_count=len(section_content)
            )
            chunks.append(chunk)
        else:
            # 需要分块
            pos = 0
            chunk_idx = 0
            
            while pos < len(section_content):
                # 计算块结束位置
                end_pos = pos + self.MAX_CHUNK_SIZE
                
                if end_pos >= len(section_content):
                    # 最后一块
                    chunk_content = section_content[pos:]
                else:
                    # 找最后一个句号或换行，避免截断句子
                    chunk_content = section_content[pos:end_pos]
                    
                    # 尝试找更好的截断点
                    for delimiter in ['.\n', '。\n', '\n\n', '. ', '。 ', '\n']:
                        last_delim = chunk_content.rfind(delimiter, self.MAX_CHUNK_SIZE - 200)
                        if last_delim > 0:
                            chunk_content = chunk_content[:last_delim + len(delimiter)]
                            break
                
                chunk_id = self._generate_chunk_id(section_title, chunk_idx)
                keywords = self._extract_keywords(chunk_content)
                
                chunk = SkillChunk(
                    id=chunk_id,
                    content=chunk_content,
                    section=section_title,
                    index=start_index + chunk_idx,
                    keywords=keywords,
                    char_count=len(chunk_content)
                )
                chunks.append(chunk)
                
                # 移动位置（考虑重叠）
                pos += len(chunk_content) - self.OVERLAP_SIZE
                chunk_idx += 1
                
                # 防止无限循环
                if len(chunk_content) < self.OVERLAP_SIZE * 2:
                    break
        
        return chunks
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词（简单实现）"""
        keywords = []
        
        # 提取加粗文本
        bold_pattern = r'\*\*(.+?)\*\*'
        bold_matches = re.findall(bold_pattern, content)
        keywords.extend(bold_matches)
        
        # 提取代码块中的函数名
        code_pattern = r'`(.+?)`'
        code_matches = re.findall(code_pattern, content)
        keywords.extend(code_matches)
        
        # 提取标题
        header_pattern = r'###?\s+(.+)'
        header_matches = re.findall(header_pattern, content)
        keywords.extend(header_matches)
        
        # 提取列表项的第一个词（通常是关键词）
        list_pattern = r'^[-*]\s*(\w+)'
        list_matches = re.findall(list_pattern, content, re.MULTILINE)
        keywords.extend(list_matches)
        
        # 去重并限制数量
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower not in seen and len(kw_lower) > 1:
                seen.add(kw_lower)
                unique_keywords.append(kw.strip())
        
        return unique_keywords[:20]  # 最多20个关键词
    
    def _generate_chunk_id(self, section: str, index: int) -> str:
        """生成块 ID"""
        content = f"{section}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _build_index(self, skill_name: str, chunks: List[SkillChunk]):
        """建立关键词索引"""
        for chunk in chunks:
            for keyword in chunk.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self._keyword_index:
                    self._keyword_index[keyword_lower] = []
                self._keyword_index[keyword_lower].append(f"{skill_name}:{chunk.id}")
    
    def retrieve_relevant_chunks(
        self, 
        skill_name: str, 
        query: str,
        top_k: int = 3
    ) -> List[ChunkMatch]:
        """
        检索与查询相关的块
        
        Args:
            skill_name: Skill 名称
            query: 用户查询
            top_k: 返回最相关的 K 个块
            
        Returns:
            匹配的块列表
        """
        if skill_name not in self._chunks:
            return []
        
        chunks = self._chunks[skill_name]
        
        # 提取查询关键词
        query_keywords = self._extract_query_keywords(query)
        
        # 计算每个块的匹配分数
        matches = []
        for chunk in chunks:
            score, matched = self._calculate_match_score(chunk, query_keywords)
            if score > 0:
                matches.append(ChunkMatch(chunk, score, matched))
        
        # 按分数排序，返回 top_k
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        # 简单分词（可以改进为更复杂的 NLP）
        # 移除停用词
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
                     '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                     '你', '会', '着', '没有', '看', '好', '自己', '这', '那'}
        
        # 提取中文词汇（2-6字）
        words = []
        for length in range(6, 1, -1):
            for i in range(len(query) - length + 1):
                word = query[i:i+length]
                if word not in stopwords:
                    words.append(word)
        
        # 同时保留原始词
        words.extend(query.lower().split())
        
        return list(set(words))
    
    def _calculate_match_score(
        self, 
        chunk: SkillChunk, 
        query_keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """计算匹配分数"""
        score = 0.0
        matched = []
        
        chunk_keywords_lower = [k.lower() for k in chunk.keywords]
        chunk_content_lower = chunk.content.lower()
        
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            
            # 关键词匹配
            if keyword_lower in chunk_keywords_lower:
                score += 3.0  # 关键词匹配权重高
                matched.append(keyword)
            
            # 内容匹配
            elif keyword_lower in chunk_content_lower:
                score += 1.0
                matched.append(keyword)
        
        # 归一化分数
        if query_keywords:
            score = score / len(query_keywords)
        
        return score, matched
    
    def get_chunk_summary(self, skill_name: str) -> Dict[str, Any]:
        """获取 Skill 分块摘要"""
        if skill_name not in self._chunks:
            return {"error": "Skill not found"}
        
        chunks = self._chunks[skill_name]
        total_chars = sum(c.char_count for c in chunks)
        
        return {
            "skill_name": skill_name,
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "sections": list(set(c.section for c in chunks)),
        }
    
    def format_chunks_for_model(
        self, 
        matches: List[ChunkMatch],
        include_metadata: bool = True
    ) -> str:
        """
        将检索到的块格式化为模型提示词
        
        Args:
            matches: 匹配的块
            include_metadata: 是否包含元数据
            
        Returns:
            格式化后的字符串
        """
        if not matches:
            return ""
        
        sections = ["=" * 50, "📚 相关 Skill 指南", "=" * 50]
        
        for i, match in enumerate(matches, 1):
            chunk = match.chunk
            
            if include_metadata:
                sections.append(f"\n【片段 {i}】章节: {chunk.section}")
                sections.append(f"关键词: {', '.join(chunk.keywords[:5])}")
                sections.append("-" * 40)
            
            sections.append(chunk.content)
        
        sections.append("\n" + "=" * 50)
        
        return "\n".join(sections)


class AdaptiveSkillLoader:
    """
    自适应 Skill 加载器
    
    根据 Skill 大小和用户请求智能选择加载策略：
    - 小 Skill (< 2000字): 完整加载
    - 中 Skill (2000-8000字): 分层加载
    - 大 Skill (> 8000字): 检索加载（RAG）
    """
    
    def __init__(self):
        self.chunker = SkillChunker()
        self._skill_cache: Dict[str, str] = {}
        self._skill_sizes: Dict[str, int] = {}
    
    def load(
        self, 
        skill_name: str, 
        content: str, 
        query: str,
        strategy: Optional[str] = None
    ) -> str:
        """
        自适应加载 Skill
        
        Args:
            skill_name: Skill 名称
            content: 完整内容
            query: 用户查询（用于检索）
            strategy: 强制指定策略 (full/layered/retrieval)
            
        Returns:
            加载的内容
        """
        content_length = len(content)
        
        # 确定策略
        if strategy is None:
            if content_length < 2000:
                strategy = "full"
            elif content_length < 8000:
                strategy = "layered"
            else:
                strategy = "retrieval"
        
        # 执行加载
        if strategy == "full":
            return self._load_full(content)
        
        elif strategy == "layered":
            return self._load_layered(content)
        
        elif strategy == "retrieval":
            return self._load_retrieval(skill_name, content, query)
        
        else:
            return content
    
    def _load_full(self, content: str) -> str:
        """完整加载"""
        return content
    
    def _load_layered(self, content: str) -> str:
        """分层加载（简化版）"""
        lines = content.split('\n')
        result = []
        h2_count = 0
        
        for line in lines:
            if line.startswith('## '):
                h2_count += 1
                if h2_count > 3:  # 只取前3个章节
                    break
            result.append(line)
        
        return '\n'.join(result)
    
    def _load_retrieval(
        self, 
        skill_name: str, 
        content: str, 
        query: str
    ) -> str:
        """检索加载（RAG）"""
        # 确保已分块
        if skill_name not in self.chunker._chunks:
            self.chunker.process_skill(skill_name, content)
        
        # 检索相关块
        matches = self.chunker.retrieve_relevant_chunks(skill_name, query, top_k=3)
        
        if not matches:
            # 如果没有匹配，返回前两个章节
            return self._load_layered(content)
        
        # 格式化检索结果
        return self.chunker.format_chunks_for_model(matches)
    
    def get_loading_info(self, skill_name: str, content: str) -> Dict[str, Any]:
        """获取加载信息（使用 tiktoken 精确计算 tokens）"""
        content_length = len(content)
        
        # 使用 tiktoken 精确计算 tokens
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4/GPT-3.5 编码
            estimated_tokens = len(encoder.encode(content))
        except ImportError:
            # 如果 tiktoken 未安装，使用估算
            estimated_tokens = content_length // 4
        except Exception:
            estimated_tokens = content_length // 4
        
        if content_length < 2000:
            strategy = "full"
        elif content_length < 8000:
            strategy = "layered"
        else:
            strategy = "retrieval"
        
        return {
            "skill_name": skill_name,
            "total_chars": content_length,
            "strategy": strategy,
            "estimated_tokens": estimated_tokens,
            "suitable_for_small_model": estimated_tokens < 4000,
        }
