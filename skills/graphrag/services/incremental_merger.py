"""
增量实体-关系合并器

基于 LightRAG 的增量更新思想，实现：
- 同名实体自动检测和合并
- 支持简单规则合并（零 LLM 成本，默认）
- 支持 LLM 智能合并（高质量，可选）
- 保留所有来源信息

适合小模型场景：默认使用简单规则合并，避免额外的 LLM 调用。
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from skills.graphrag.services.triple_vector_store import TripleVectorStore, EntityInfo, RelationInfo
from skills.graphrag.services.gleaning_extractor import simple_merge_descriptions


@dataclass
class MergeConfig:
    """合并配置"""
    use_llm_merge: bool = False  # 默认使用简单规则，避免 LLM 调用
    similarity_threshold: float = 0.9  # 实体相似度阈值
    keep_all_sources: bool = True  # 保留所有来源信息


class IncrementalMerger:
    """
    增量实体-关系合并器

    借鉴 LightRAG 的增量更新算法：
    1. 检测同名实体
    2. 支持配置化合并策略
    3. 保留来源信息

    小模型场景建议：
    - use_llm_merge=False：使用简单规则合并，零 LLM 成本
    - use_llm_merge=True：使用 LLM 智能合并，更高质量
    """

    def __init__(
        self,
        triple_store: TripleVectorStore,
        embedding_fn: Any,
        llm_client: Any = None,
        config: Optional[MergeConfig] = None
    ):
        """
        初始化增量合并器

        Args:
            triple_store: 三重向量存储
            embedding_fn: Embedding 函数
            llm_client: LLM 客户端（可选，用于智能合并）
            config: 合并配置
        """
        self.triple_store = triple_store
        self.embedding_fn = embedding_fn
        self.llm_client = llm_client
        self.config = config or MergeConfig()

        # 实体缓存：name -> EntityInfo
        self._entity_cache: Dict[str, EntityInfo] = {}
        self._load_entity_cache()

    def _load_entity_cache(self):
        """加载实体缓存"""
        # 从 triple_store 的实体描述缓存加载
        # 实体缓存由 triple_store 维护
        pass

    def merge_entity(self, new_entity: EntityInfo) -> EntityInfo:
        """
        增量合并实体

        如果实体已存在，合并描述；
        如果不存在，直接添加。

        Args:
            new_entity: 新实体

        Returns:
            合并后的实体
        """
        existing = self._find_existing_entity(new_entity.name)

        if existing:
            # 实体已存在，合并描述
            merged_desc = self._merge_descriptions(
                existing.description,
                new_entity.description
            )

            # 合并来源
            merged_source_ids = list(set(
                existing.source_ids + new_entity.source_ids
            ))

            # 更新实体
            existing.description = merged_desc
            existing.source_ids = merged_source_ids

            # 更新向量索引
            self._update_entity_vector(existing)

            return existing
        else:
            # 新实体，添加到存储
            self._add_new_entity(new_entity)
            return new_entity

    def merge_relation(self, new_relation: RelationInfo) -> RelationInfo:
        """
        增量合并关系

        如果关系已存在，更新置信度和来源；
        如果不存在，直接添加。

        Args:
            new_relation: 新关系

        Returns:
            合并后的关系
        """
        existing = self._find_existing_relation(
            new_relation.source,
            new_relation.relation,
            new_relation.target
        )

        if existing:
            # 关系已存在，更新置信度（取最高）
            existing.confidence = max(existing.confidence, new_relation.confidence)

            # 合并来源
            existing.source_ids = list(set(
                existing.source_ids + new_relation.source_ids
            ))

            # 如果新描述更详细，更新描述
            if len(new_relation.description) > len(existing.description):
                existing.description = new_relation.description

            return existing
        else:
            # 新关系，添加到存储
            self._add_new_relation(new_relation)
            return new_relation

    def process_new_document(
        self,
        entities: List[EntityInfo],
        relations: List[RelationInfo]
    ) -> Tuple[List[EntityInfo], List[RelationInfo]]:
        """
        处理新文档的实体和关系

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            (合并后的实体列表, 合并后的关系列表)
        """
        merged_entities = []
        merged_relations = []

        # 先处理实体
        for entity in entities:
            merged = self.merge_entity(entity)
            merged_entities.append(merged)

        # 再处理关系
        for relation in relations:
            merged = self.merge_relation(relation)
            merged_relations.append(merged)

        return merged_entities, merged_relations

    def _find_existing_entity(self, entity_name: str) -> Optional[EntityInfo]:
        """
        查找已存在的实体

        Args:
            entity_name: 实体名称

        Returns:
            实体信息或 None
        """
        # 从缓存中查找
        if entity_name in self._entity_cache:
            return self._entity_cache[entity_name]

        # 从向量存储中查找
        description = self.triple_store.get_entity_description(entity_name)
        if description:
            # 构建 EntityInfo
            entity_info = EntityInfo(
                name=entity_name,
                entity_type="未知",  # 需要从存储中获取
                description=description,
                source_ids=[]
            )
            self._entity_cache[entity_name] = entity_info
            return entity_info

        return None

    def _find_existing_relation(
        self,
        source: str,
        relation: str,
        target: str
    ) -> Optional[RelationInfo]:
        """
        查找已存在的关系

        Args:
            source: 源实体
            relation: 关系类型
            target: 目标实体

        Returns:
            关系信息或 None
        """
        # 关系 ID 格式：source_relation_target
        relation_id = f"{source}_{relation}_{target}"

        # 这里简化处理，实际可以从关系存储中查询
        # 由于关系存储在 triple_store 中，且没有直接的按 ID 查询接口
        # 这里返回 None 表示新关系
        return None

    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        """
        合并描述

        Args:
            desc1: 已有描述
            desc2: 新描述

        Returns:
            合并后的描述
        """
        if self.config.use_llm_merge and self.llm_client:
            return self._llm_merge_descriptions(desc1, desc2)
        else:
            return simple_merge_descriptions(desc1, desc2)

    def _llm_merge_descriptions(self, desc1: str, desc2: str) -> str:
        """
        使用 LLM 智能合并描述

        Args:
            desc1: 已有描述
            desc2: 新描述

        Returns:
            合并后的描述
        """
        try:
            prompt = f"""请将以下两个关于同一实体的描述合并为一个综合描述：

描述1：
{desc1}

描述2：
{desc2}

要求：
1. 保留所有关键信息
2. 去除重复内容
3. 语言简洁流畅
4. 长度控制在200字以内

合并后的描述："""

            merged = self.llm_client.generate(prompt, timeout=30, silent=True)
            return merged.strip() if merged else simple_merge_descriptions(desc1, desc2)

        except Exception:
            # LLM 失败时回退到简单合并
            return simple_merge_descriptions(desc1, desc2)

    def _add_new_entity(self, entity: EntityInfo):
        """
        添加新实体到存储

        Args:
            entity: 实体信息
        """
        try:
            # 生成 embedding
            entity_text = f"{entity.name} {entity.entity_type} {entity.description}"
            embedding = self.embedding_fn([entity_text])[0]

            # 添加到向量存储
            self.triple_store.add_entity(entity, embedding)

            # 更新缓存
            self._entity_cache[entity.name] = entity

        except Exception as e:
            print(f"⚠️ 添加实体失败: {e}")

    def _add_new_relation(self, relation: RelationInfo):
        """
        添加新关系到存储

        Args:
            relation: 关系信息
        """
        try:
            # 生成 embedding
            relation_text = relation.description if relation.description else \
                f"{relation.source} {relation.relation} {relation.target}"
            embedding = self.embedding_fn([relation_text])[0]

            # 添加到向量存储
            self.triple_store.add_relation(relation, embedding)

        except Exception as e:
            print(f"⚠️ 添加关系失败: {e}")

    def _update_entity_vector(self, entity: EntityInfo):
        """
        更新实体向量

        Args:
            entity: 实体信息
        """
        try:
            # 重新生成 embedding
            entity_text = f"{entity.name} {entity.entity_type} {entity.description}"
            embedding = self.embedding_fn([entity_text])[0]

            # 更新向量存储
            self.triple_store.add_entity(entity, embedding)

        except Exception as e:
            print(f"⚠️ 更新实体向量失败: {e}")

    def get_entity_history(self, entity_name: str) -> Dict:
        """
        获取实体历史

        Args:
            entity_name: 实体名称

        Returns:
            实体历史信息
        """
        entity = self._find_existing_entity(entity_name)
        if entity:
            return {
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
                "source_count": len(entity.source_ids),
                "sources": entity.source_ids
            }
        return {}

    def get_merge_stats(self) -> Dict:
        """
        获取合并统计

        Returns:
            统计信息
        """
        return {
            "cached_entities": len(self._entity_cache),
            "use_llm_merge": self.config.use_llm_merge
        }
