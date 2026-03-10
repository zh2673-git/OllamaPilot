"""
实体抽取器 - 基于规则 + 模式匹配

适合本地 Ollama 小模型环境
能够建立实体关系和知识图谱
"""

from typing import List, Dict, Optional, Set, Tuple
import re
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


class LightweightEntityExtractor:
    """
    轻量级实体抽取器

    基于规则 + 模式匹配抽取实体
    能够识别专业术语并建立关系
    """

    def __init__(self):
        # 停用词 - 过滤常见虚词
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

        # 专业术语词典 - 用于精确匹配
        self.entity_dict = {
            # 中医病名
            "病名": [
                "太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病",
                "中风", "伤寒", "温病", "风温", "湿温", "暑温",
                "表证", "里证", "半表半里", "寒证", "热证", "虚证", "实证",
                "太阳中风", "太阳伤寒", "阳明经证", "阳明腑证",
            ],
            # 中医方剂
            "方剂": [
                "桂枝汤", "麻黄汤", "葛根汤", "小柴胡汤", "大柴胡汤",
                "白虎汤", "承气汤", "四逆汤", "理中汤", "真武汤",
                "五苓散", "小建中汤", "炙甘草汤", "四逆散", "当归四逆汤",
                "桂枝加葛根汤", "桂枝加厚朴杏子汤", "麻黄杏仁甘草石膏汤",
            ],
            # 中药名
            "中药": [
                "桂枝", "麻黄", "柴胡", "黄芩", "人参", "甘草", "大枣", "生姜",
                "芍药", "半夏", "茯苓", "白术", "附子", "干姜", "细辛",
                "黄连", "阿胶", "地黄", "麦冬", "五味子", "当归", "川芎",
                "杏仁", "石膏", "知母", "粳米", "厚朴", "枳实", "大黄",
                "芒硝", "葛根", "天花粉", "牡蛎", "龙骨",
            ],
            # 中医症状/体征
            "症状": [
                "脉浮", "脉紧", "脉缓", "脉细", "脉沉", "脉数", "脉弱",
                "发热", "恶寒", "汗出", "无汗", "头痛", "身痛", "腰痛",
                "呕吐", "下利", "腹满", "腹痛", "心悸", "烦躁", "失眠",
                "口苦", "咽干", "目眩", "耳聋", "胸胁苦满", "往来寒热",
                "项背强几几", "鼻鸣", "干呕", "喘", "咳", "渴",
            ],
            # 中医治法
            "治法": [
                "发汗", "解表", "清热", "泻下", "温里", "补益", "和解",
                "散寒", "生津", "止渴", "止呕", "止痛", "安神",
            ],
            # 古代医家
            "医家": [
                "张仲景", "华佗", "扁鹊", "孙思邈", "李时珍", "王叔和",
            ],
            # 中医经典
            "经典": [
                "伤寒论", "金匮要略", "黄帝内经", "神农本草经", "难经",
            ],
        }

        # 合并所有术语用于快速查找
        self.all_terms = {}
        for entity_type, terms in self.entity_dict.items():
            for term in terms:
                self.all_terms[term] = entity_type

        # 模式匹配规则 - 用于识别未在词典中的实体
        self.patterns = {
            # 方剂模式：XX汤、XX散、XX丸
            "方剂": r"[\u4e00-\u9fa5]{2,6}(?:汤|散|丸|膏|丹)",
            # 症状模式：XX痛、XX满、XX呕
            "症状": r"[\u4e00-\u9fa5]{1,4}(?:痛|满|呕|吐|利|渴|汗|热|寒|烦)",
            # 脉象模式：脉XX
            "症状": r"脉[\u4e00-\u9fa5]{1,3}",
        }

    def extract(self, text: str, top_k: int = 20) -> List[ExtractedEntity]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            top_k: 最多返回的实体数

        Returns:
            实体列表
        """
        entities = []
        found_positions = set()  # 记录已找到的位置，避免重叠

        # 1. 词典精确匹配（高优先级）
        for term, entity_type in self.all_terms.items():
            for match in re.finditer(re.escape(term), text):
                start, end = match.start(), match.end()
                # 检查是否与已找到的实体重叠
                if not self._is_overlapping(start, end, found_positions):
                    entities.append(ExtractedEntity(
                        name=term,
                        type=entity_type,
                        start=start,
                        end=end,
                        confidence=0.95
                    ))
                    # 记录位置
                    for i in range(start, end):
                        found_positions.add(i)

        # 2. 模式匹配（补充识别）
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                term = text[start:end]
                # 检查是否已存在或重叠
                if term not in self.all_terms and not self._is_overlapping(start, end, found_positions):
                    # 过滤停用词和太短的词
                    if term not in self.stopwords and len(term) >= 2:
                        entities.append(ExtractedEntity(
                            name=term,
                            type=entity_type,
                            start=start,
                            end=end,
                            confidence=0.7
                        ))
                        for i in range(start, end):
                            found_positions.add(i)

        # 3. 去重和排序
        entities = self._deduplicate(entities)
        entities.sort(key=lambda e: (e.confidence, len(e.name)), reverse=True)

        return entities[:top_k]

    def extract_with_context(self, text: str, window_size: int = 50) -> List[Dict]:
        """
        抽取实体并附带上下文

        Args:
            text: 输入文本
            window_size: 上下文窗口大小

        Returns:
            带上下文的实体列表
        """
        entities = self.extract(text)
        result = []

        for entity in entities:
            # 提取上下文
            start = max(0, entity.start - window_size)
            end = min(len(text), entity.end + window_size)
            context = text[start:end]

            result.append({
                "name": entity.name,
                "type": entity.type,
                "position": (entity.start, entity.end),
                "confidence": entity.confidence,
                "context": context
            })

        return result

    def _is_overlapping(self, start: int, end: int, found_positions: Set[int]) -> bool:
        """检查位置是否与已找到的实体重叠"""
        for i in range(start, end):
            if i in found_positions:
                return True
        return False

    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """去重：优先保留置信度高的"""
        seen = {}
        for e in entities:
            if e.name not in seen or e.confidence > seen[e.name].confidence:
                seen[e.name] = e
        return list(seen.values())

    def add_entity_type(self, entity_type: str, terms: List[str]):
        """添加新的实体类型"""
        if entity_type not in self.entity_dict:
            self.entity_dict[entity_type] = []
        self.entity_dict[entity_type].extend(terms)
        self.entity_dict[entity_type] = list(set(self.entity_dict[entity_type]))
        for term in terms:
            self.all_terms[term] = entity_type

    def load_dictionary(self, dictionary: Dict[str, List[str]]):
        """加载词典"""
        for entity_type, terms in dictionary.items():
            self.add_entity_type(entity_type, terms)
