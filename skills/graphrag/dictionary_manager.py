"""
词典管理器

管理全局预设词典和文档私有词典的加载、合并和继承
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class DictionaryConfig:
    """词典配置"""
    name: str
    version: str
    domain: str
    description: str
    entities: Dict[str, List[str]] = field(default_factory=dict)


class DictionaryManager:
    """
    词典管理器

    功能：
    1. 加载全局预设词典（config/dictionaries/）
    2. 加载文档私有词典（data/graphrag/{doc_id}/）
    3. 合并词典（全局 + 文档私有）
    4. 支持多领域词典组合
    """

    def __init__(self, config_dir: str = "config/dictionaries"):
        """
        初始化词典管理器

        Args:
            config_dir: 全局预设词典目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 缓存已加载的全局词典
        self._global_dictionaries: Dict[str, DictionaryConfig] = {}

        # 加载所有全局词典
        self._load_global_dictionaries()

    def _load_global_dictionaries(self):
        """加载所有全局预设词典"""
        if not self.config_dir.exists():
            return

        loaded_count = 0
        for dict_file in self.config_dir.glob("*.json"):
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                config = DictionaryConfig(
                    name=metadata.get("name", dict_file.stem),
                    version=metadata.get("version", "1.0"),
                    domain=metadata.get("domain", "general"),
                    description=metadata.get("description", ""),
                    entities=data.get("entities", {})
                )

                self._global_dictionaries[dict_file.stem] = config
                loaded_count += 1

            except Exception as e:
                print(f"⚠️ 加载词典失败 {dict_file}: {e}")

        if loaded_count > 0:
            print(f"📚 已加载 {loaded_count} 个领域词典")

    def get_global_dictionary(self, name: str) -> Optional[DictionaryConfig]:
        """
        获取指定全局词典

        Args:
            name: 词典名称（如 "global", "tcm", "custom"）

        Returns:
            词典配置
        """
        return self._global_dictionaries.get(name)

    def list_global_dictionaries(self) -> List[str]:
        """
        列出所有全局词典名称

        Returns:
            词典名称列表
        """
        return list(self._global_dictionaries.keys())

    def get_merged_dictionary(
        self,
        doc_dictionary_path: Optional[Path] = None,
        selected_globals: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        获取合并后的词典

        合并顺序：
        1. 全局预设词典（按selected_globals顺序）
        2. 文档私有词典（如果存在）

        Args:
            doc_dictionary_path: 文档私有词典路径
            selected_globals: 选择的全局词典名称列表，None表示全部

        Returns:
            合并后的实体词典 {类型: [实体列表]}
        """
        merged: Dict[str, Set[str]] = {}

        # 1. 加载选中的全局词典
        if selected_globals is None:
            selected_globals = list(self._global_dictionaries.keys())

        for dict_name in selected_globals:
            config = self._global_dictionaries.get(dict_name)
            if config:
                for entity_type, entities in config.entities.items():
                    if entity_type not in merged:
                        merged[entity_type] = set()
                    merged[entity_type].update(entities)

        # 2. 加载文档私有词典
        if doc_dictionary_path and doc_dictionary_path.exists():
            try:
                with open(doc_dictionary_path, 'r', encoding='utf-8') as f:
                    doc_dict = json.load(f)

                for entity_type, entities in doc_dict.get("entities", {}).items():
                    if entity_type not in merged:
                        merged[entity_type] = set()
                    merged[entity_type].update(entities)

            except Exception as e:
                print(f"⚠️ 加载文档词典失败: {e}")

        # 转换为列表
        return {k: list(v) for k, v in merged.items()}

    def get_all_terms(self, doc_dictionary_path: Optional[Path] = None) -> Dict[str, str]:
        """
        获取所有词条（用于快速匹配）

        Args:
            doc_dictionary_path: 文档私有词典路径

        Returns:
            {词条: 类型} 字典
        """
        merged = self.get_merged_dictionary(doc_dictionary_path)
        all_terms = {}

        for entity_type, entities in merged.items():
            for entity in entities:
                all_terms[entity] = entity_type

        return all_terms

    def save_document_dictionary(
        self,
        doc_id: str,
        entities: Dict[str, List[str]],
        persist_dir: str
    ):
        """
        保存文档私有词典

        Args:
            doc_id: 文档ID
            entities: 实体词典
            persist_dir: 文档存储目录
        """
        doc_dir = Path(persist_dir) / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        dict_path = doc_dir / "dictionary.json"

        data = {
            "metadata": {
                "doc_id": doc_id,
                "version": "1.0",
                "description": "文档私有词典（LLM动态学习）"
            },
            "entities": entities
        }

        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_document_dictionary(
        self,
        doc_id: str,
        new_entities: Dict[str, List[str]],
        persist_dir: str
    ):
        """
        更新文档私有词典（增量更新）

        Args:
            doc_id: 文档ID
            new_entities: 新发现的实体
            persist_dir: 文档存储目录
        """
        doc_dir = Path(persist_dir) / doc_id
        dict_path = doc_dir / "dictionary.json"

        # 加载现有词典
        existing = {}
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing = data.get("entities", {})
            except Exception:
                pass

        # 合并新实体
        for entity_type, entities in new_entities.items():
            if entity_type not in existing:
                existing[entity_type] = []
            for entity in entities:
                if entity not in existing[entity_type]:
                    existing[entity_type].append(entity)

        # 保存
        self.save_document_dictionary(doc_id, existing, persist_dir)

    def get_dictionary_stats(self) -> Dict[str, int]:
        """
        获取词典统计信息

        Returns:
            各词典实体数量统计
        """
        stats = {}
        for name, config in self._global_dictionaries.items():
            total = sum(len(entities) for entities in config.entities.values())
            stats[name] = total
        return stats


# 全局词典管理器实例
_global_dict_manager: Optional[DictionaryManager] = None


def get_dictionary_manager(config_dir: str = "config/dictionaries") -> DictionaryManager:
    """
    获取全局词典管理器实例（单例模式）

    Args:
        config_dir: 词典配置目录

    Returns:
        DictionaryManager 实例
    """
    global _global_dict_manager
    if _global_dict_manager is None:
        _global_dict_manager = DictionaryManager(config_dir)
    return _global_dict_manager
