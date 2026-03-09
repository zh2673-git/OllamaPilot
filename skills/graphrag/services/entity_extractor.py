"""
轻量级实体抽取器

使用规则 + 词典匹配，适合小模型环境
"""

from typing import List, Dict, Optional, Set
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

    使用规则 + 词典匹配，适合小模型环境
    """

    def __init__(self):
        # 内置词典（可扩展）
        self.entity_dict: Dict[str, List[str]] = {
            "Person": ["张三", "李四", "王五", "赵六", "小明", "小红", "老王"],
            "Organization": ["腾讯", "阿里巴巴", "百度", "字节跳动", "华为", "小米", "京东"],
            "Location": ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "西安"],
            "Product": ["iPhone", "iPad", "MacBook", "Windows", "Android", "微信", "支付宝"],
        }

        # 规则模式
        self.patterns: Dict[str, str] = {
            "Person": r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|博士|教授|经理|老师)?",
            "Organization": r"[\u4e00-\u9fa5]{2,10}(?:公司|集团|大学|学院|研究所|银行|医院)?",
            "Location": r"[\u4e00-\u9fa5]{2,8}(?:省|市|区|县|镇|村|路|街)?",
        }

    def extract(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本
            entity_types: 指定抽取的实体类型（None=全部）

        Returns:
            实体列表
        """
        entities = []
        types_to_extract = entity_types or list(self.entity_dict.keys())

        # 1. 词典匹配
        for entity_type in types_to_extract:
            if entity_type in self.entity_dict:
                for entity_name in self.entity_dict[entity_type]:
                    for match in re.finditer(re.escape(entity_name), text):
                        entities.append(ExtractedEntity(
                            name=entity_name,
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.9
                        ))

        # 2. 规则匹配
        for entity_type in types_to_extract:
            if entity_type in self.patterns:
                pattern = self.patterns[entity_type]
                for match in re.finditer(pattern, text):
                    # 避免重复
                    if not any(e.start == match.start() for e in entities):
                        entities.append(ExtractedEntity(
                            name=match.group(),
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7
                        ))

        # 去重和排序
        entities = self._deduplicate(entities)
        entities.sort(key=lambda e: e.start)

        return entities

    def extract_from_query(self, text: str) -> List[Dict[str, str]]:
        """
        从查询中提取实体（简化格式）

        Args:
            text: 查询文本

        Returns:
            实体字典列表
        """
        entities = self.extract(text)
        return [{"name": e.name, "type": e.type} for e in entities]

    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """去重：优先保留置信度高的"""
        seen: Dict[tuple, ExtractedEntity] = {}
        for e in entities:
            key = (e.name, e.start)
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e
        return list(seen.values())

    def add_to_dictionary(self, entity_type: str, entity_names: List[str]):
        """向词典添加实体"""
        if entity_type not in self.entity_dict:
            self.entity_dict[entity_type] = []
        self.entity_dict[entity_type].extend(entity_names)
        self.entity_dict[entity_type] = list(set(self.entity_dict[entity_type]))

    def load_dictionary(self, dictionary: Dict[str, List[str]]):
        """加载词典"""
        for entity_type, names in dictionary.items():
            self.add_to_dictionary(entity_type, names)
