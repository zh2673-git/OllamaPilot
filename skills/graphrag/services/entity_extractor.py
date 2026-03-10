"""
轻量级关键词提取器

基于 MiroFish 和 LangExtract 架构简化
适合本地 Ollama 小模型环境
"""

from typing import List, Dict, Optional, Set, Tuple
import re
import jieba
from dataclasses import dataclass
from collections import Counter


@dataclass
class ExtractedKeyword:
    """提取的关键词"""
    name: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


class LightweightEntityExtractor:
    """
    轻量级关键词提取器

    不追求完整的知识图谱实体抽取，而是提取关键词用于增强检索
    适合本地小模型环境
    """

    def __init__(self):
        # 停用词
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '之', '与', '及', '等', '或', '但', '而', '因为', '所以',
            '如果', '虽然', '但是', '然后', '而且', '或者', '还是',
            '可以', '可能', '应该', '需要', '进行', '通过', '根据',
            '由于', '因此', '其中', '其他', '已经', '正在', '曾经',
            '现在', '当时', '这里', '那里', '什么', '怎么', '为什么',
            '如何', '谁', '哪', '个', '为', '以', '及', '而', '或',
            '但', '是', '在', '有', '被', '把', '让', '向', '到',
            '从', '将', '对', '关于', '由于', '根据', '按照',
        ])

        # 专业术语词典（可扩展）
        self.domain_terms = {
            # 中医相关
            '中医病名': [
                '太阳病', '阳明病', '少阳病', '太阴病', '少阴病', '厥阴病',
                '中风', '伤寒', '温病', '风温', '湿温', '暑温',
                '表证', '里证', '半表半里', '寒证', '热证', '虚证', '实证'
            ],
            '中医方剂': [
                '桂枝汤', '麻黄汤', '葛根汤', '小柴胡汤', '大柴胡汤',
                '白虎汤', '承气汤', '四逆汤', '理中汤', '真武汤',
                '五苓散', '小建中汤', '炙甘草汤', '四逆散'
            ],
            '中药名': [
                '桂枝', '麻黄', '柴胡', '黄芩', '人参', '甘草', '大枣', '生姜',
                '芍药', '半夏', '茯苓', '白术', '附子', '干姜', '细辛',
                '黄连', '阿胶', '地黄', '麦冬', '五味子'
            ],
            '中医术语': [
                '脉浮', '脉紧', '脉缓', '脉细', '脉沉', '脉数',
                '发热', '恶寒', '汗出', '无汗', '头痛', '身痛',
                '呕吐', '下利', '腹满', '腹痛', '心悸', '烦躁',
                '口苦', '咽干', '目眩', '耳聋', '胸胁苦满'
            ],
            '古代医家': [
                '张仲景', '华佗', '扁鹊', '孙思邈', '李时珍'
            ],
        }

        # 合并所有专业术语
        self.all_terms = set()
        for terms in self.domain_terms.values():
            self.all_terms.update(terms)

    def extract(
        self,
        text: str,
        top_k: int = 10
    ) -> List[ExtractedKeyword]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本
            top_k: 返回前k个关键词

        Returns:
            关键词列表
        """
        keywords = []

        # 1. 专业术语匹配（高优先级）
        for term_type, terms in self.domain_terms.items():
            for term in terms:
                for match in re.finditer(re.escape(term), text):
                    keywords.append(ExtractedKeyword(
                        name=term,
                        type=term_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    ))

        # 2. 使用 jieba 分词提取关键词
        try:
            # 添加自定义词典
            for term in self.all_terms:
                jieba.add_word(term, freq=1000)

            # 分词
            words = jieba.lcut(text)

            # 过滤停用词和单字
            filtered_words = [
                w for w in words
                if len(w) > 1 and w not in self.stopwords and not w.isdigit()
            ]

            # 统计词频
            word_counts = Counter(filtered_words)

            # 选择高频词（排除已在专业术语中的）
            existing_terms = {k.name for k in keywords}
            for word, count in word_counts.most_common(top_k * 2):
                if word not in existing_terms and len(keywords) < top_k * 2:
                    # 找到词在文本中的位置
                    for match in re.finditer(re.escape(word), text):
                        keywords.append(ExtractedKeyword(
                            name=word,
                            type="关键词",
                            start=match.start(),
                            end=match.end(),
                            confidence=min(0.5 + count * 0.05, 0.8)
                        ))
                        break  # 只取第一次出现的位置

        except Exception as e:
            # jieba 失败时，使用简单的空格分词
            words = text.split()
            for word in words:
                if len(word) > 1 and word not in self.stopwords:
                    for match in re.finditer(re.escape(word), text):
                        keywords.append(ExtractedKeyword(
                            name=word,
                            type="关键词",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.5
                        ))
                        break

        # 去重和排序
        keywords = self._deduplicate(keywords)
        keywords.sort(key=lambda k: (k.confidence, len(k.name)), reverse=True)

        # 返回前 top_k 个
        return keywords[:top_k]

    def extract_from_query(self, text: str) -> List[Dict[str, str]]:
        """
        从查询中提取关键词（简化格式）

        Args:
            text: 查询文本

        Returns:
            关键词字典列表
        """
        keywords = self.extract(text, top_k=5)
        return [{"name": k.name, "type": k.type} for k in keywords]

    def _deduplicate(self, keywords: List[ExtractedKeyword]) -> List[ExtractedKeyword]:
        """去重：优先保留置信度高的"""
        seen: Dict[str, ExtractedKeyword] = {}
        for k in keywords:
            if k.name not in seen or k.confidence > seen[k.name].confidence:
                seen[k.name] = k
        return list(seen.values())

    def add_domain_terms(self, term_type: str, terms: List[str]):
        """添加领域术语"""
        if term_type not in self.domain_terms:
            self.domain_terms[term_type] = []
        self.domain_terms[term_type].extend(terms)
        self.domain_terms[term_type] = list(set(self.domain_terms[term_type]))
        self.all_terms.update(terms)

    def load_domain_dictionary(self, dictionary: Dict[str, List[str]]):
        """加载领域词典"""
        for term_type, terms in dictionary.items():
            self.add_domain_terms(term_type, terms)
