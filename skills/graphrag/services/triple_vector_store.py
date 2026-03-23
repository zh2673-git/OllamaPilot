"""
三重向量存储 (Triple Vector Store)

基于 LightRAG 的核心创新，实现三重向量存储：
- 实体向量存储 (Entity Vector Store)
- 关系向量存储 (Relation Vector Store) - 核心新增
- 文本块向量存储 (Chunk Vector Store) - 已有

提供统一的接口管理实体、关系和文本块的向量存储。
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass

from skills.graphrag.services.simple_vector_store import SimpleVectorStore


@dataclass
class EntityInfo:
    """实体信息"""
    name: str
    entity_type: str
    description: str
    source_ids: List[str]


@dataclass
class RelationInfo:
    """关系信息"""
    source: str
    target: str
    relation: str
    description: str
    confidence: float
    source_ids: List[str]


class TripleVectorStore:
    """
    三重向量存储：实体 + 关系 + 文本块

    借鉴 LightRAG 的双层检索思想，将关系也进行向量化存储，
    支持基于关系的语义检索（Global 检索）。
    """

    def __init__(self, persist_dir: str, collection_name: str):
        """
        初始化三重向量存储

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # 文本块向量存储（已有功能）
        self.chunk_store = SimpleVectorStore(
            persist_dir=persist_dir,
            collection_name=f"{collection_name}_chunks"
        )

        # 实体向量存储（新增）
        self.entity_store = SimpleVectorStore(
            persist_dir=persist_dir,
            collection_name=f"{collection_name}_entities"
        )

        # 关系向量存储（核心新增）
        self.relation_store = SimpleVectorStore(
            persist_dir=persist_dir,
            collection_name=f"{collection_name}_relations"
        )

        # 实体描述缓存（用于增量合并）
        self._entity_descriptions: Dict[str, str] = {}
        self._load_entity_descriptions()

    def _load_entity_descriptions(self):
        """加载实体描述缓存"""
        desc_path = self.persist_dir / f"entity_descriptions_{self.collection_name}.json"
        if desc_path.exists():
            try:
                with open(desc_path, 'r', encoding='utf-8') as f:
                    self._entity_descriptions = json.load(f)
            except Exception as e:
                print(f"⚠️ 加载实体描述失败: {e}")
                self._entity_descriptions = {}

    def _save_entity_descriptions(self):
        """保存实体描述缓存"""
        desc_path = self.persist_dir / f"entity_descriptions_{self.collection_name}.json"
        try:
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(self._entity_descriptions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存实体描述失败: {e}")

    def add_chunk(self, chunk_id: str, text: str, embedding: List[float], metadata: Dict = None):
        """
        添加文本块到向量存储

        Args:
            chunk_id: 块ID
            text: 文本内容
            embedding: 向量
            metadata: 元数据
        """
        self.chunk_store.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )

    def add_entity(self, entity: EntityInfo, embedding: List[float]):
        """
        添加实体到向量存储

        Args:
            entity: 实体信息
            embedding: 向量
        """
        entity_id = f"{entity.name}_{entity.entity_type}"

        # 检查是否已存在（增量合并）
        if entity_id in self._entity_descriptions:
            # 合并描述
            existing_desc = self._entity_descriptions[entity_id]
            merged_desc = self._merge_descriptions(existing_desc, entity.description)
            entity_text = f"{entity.name} {entity.entity_type} {merged_desc}"
            self._entity_descriptions[entity_id] = merged_desc
        else:
            entity_text = f"{entity.name} {entity.entity_type} {entity.description}"
            self._entity_descriptions[entity_id] = entity.description

        self.entity_store.add(
            ids=[entity_id],
            documents=[entity_text],
            embeddings=[embedding],
            metadatas=[{
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
                "source_ids": entity.source_ids
            }]
        )

        # 保存描述缓存
        self._save_entity_descriptions()

    def add_relation(self, relation: RelationInfo, embedding: List[float]):
        """
        添加关系到向量存储

        Args:
            relation: 关系信息
            embedding: 向量
        """
        relation_id = f"{relation.source}_{relation.relation}_{relation.target}"

        # 关系文本：用于向量化的描述
        relation_text = relation.description if relation.description else \
            f"{relation.source} {relation.relation} {relation.target}"

        self.relation_store.add(
            ids=[relation_id],
            documents=[relation_text],
            embeddings=[embedding],
            metadatas=[{
                "source": relation.source,
                "target": relation.target,
                "relation": relation.relation,
                "description": relation.description,
                "confidence": relation.confidence,
                "source_ids": relation.source_ids
            }]
        )

    def search_chunks(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """
        检索文本块

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        results = self.chunk_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format_results(results)

    def search_entities(self, query_embedding: List[float], top_k: int = 30) -> List[Dict]:
        """
        检索实体（Local 检索）

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            实体列表
        """
        results = self.entity_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format_results(results)

    def search_relations(self, query_embedding: List[float], top_k: int = 30) -> List[Dict]:
        """
        检索关系（Global 检索）

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            关系列表
        """
        results = self.relation_store.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> List[Dict]:
        """格式化检索结果"""
        formatted = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append({
                "id": doc_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
                "score": 1 - (results["distances"][0][i] if results["distances"] else 0.0)
            })

        return formatted

    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        """
        简单合并描述（避免 LLM 调用）

        Args:
            desc1: 已有描述
            desc2: 新描述

        Returns:
            合并后的描述
        """
        if desc1 == desc2:
            return desc1
        if not desc1:
            return desc2
        if not desc2:
            return desc1
        if desc2 in desc1:
            return desc1
        if desc1 in desc2:
            return desc2
        return f"{desc1}；{desc2}"

    def get_entity_description(self, entity_name: str) -> Optional[str]:
        """获取实体描述"""
        for key, desc in self._entity_descriptions.items():
            if key.startswith(f"{entity_name}_"):
                return desc
        return None

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "chunks": self.chunk_store.count(),
            "entities": self.entity_store.count(),
            "relations": self.relation_store.count()
        }

    def migrate_from_legacy(
        self,
        entity_index: Dict[str, Dict],
        relations: List[Any],
        embedding_fn
    ) -> bool:
        """
        从旧版数据迁移

        Args:
            entity_index: 旧版实体索引
            relations: 旧版关系列表
            embedding_fn: Embedding 函数

        Returns:
            是否成功
        """
        import time
        print("🔄 开始从旧版数据迁移...")

        migrated_count = 0
        failed_count = 0

        # 1. 迁移实体
        total_entities = len(entity_index)
        print(f"  共 {total_entities} 个实体需要迁移")

        for i, (entity_name, entity_data) in enumerate(entity_index.items(), 1):
            try:
                entity_type = entity_data.get("type", "未知")
                entity_text = f"{entity_name} {entity_type}"
                embedding = embedding_fn([entity_text])[0]

                entity_info = EntityInfo(
                    name=entity_name,
                    entity_type=entity_type,
                    description=entity_text,
                    source_ids=list(entity_data.get("doc_ids", set()))
                )

                self.add_entity(entity_info, embedding)
                migrated_count += 1

                # 每 10 个实体添加一个小延迟，避免请求过快
                if i % 10 == 0:
                    time.sleep(0.1)

                if migrated_count % 50 == 0:
                    print(f"  已迁移 {migrated_count}/{total_entities} 个实体...")

            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # 只显示前 3 个错误
                    print(f"  ⚠️ 实体 {entity_name} 迁移失败: {e}")
                elif failed_count == 4:
                    print(f"  ⚠️ 更多实体迁移错误已省略...")

        print(f"✅ 实体迁移完成: {migrated_count}/{total_entities} 个成功")

        # 2. 迁移关系
        relation_count = 0
        failed_relations = 0
        total_relations = len(relations)
        print(f"  共 {total_relations} 个关系需要迁移")

        for i, rel in enumerate(relations, 1):
            try:
                # 处理 Relation 对象或字典
                if hasattr(rel, 'source'):
                    source = rel.source
                    target = rel.target
                    relation = rel.relation
                    confidence = rel.confidence
                    doc_id = rel.doc_id
                else:
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    relation = rel.get("relation", "")
                    confidence = rel.get("confidence", 0.5)
                    doc_id = rel.get("doc_id", "")

                relation_desc = f"{source} {relation} {target}"
                embedding = embedding_fn([relation_desc])[0]

                relation_info = RelationInfo(
                    source=source,
                    target=target,
                    relation=relation,
                    description=relation_desc,
                    confidence=confidence,
                    source_ids=[doc_id] if doc_id else []
                )

                self.add_relation(relation_info, embedding)
                relation_count += 1

                # 每 10 个关系添加一个小延迟
                if i % 10 == 0:
                    time.sleep(0.1)

            except Exception as e:
                failed_relations += 1
                if failed_relations <= 3:
                    print(f"  ⚠️ 关系迁移失败: {e}")

        print(f"✅ 关系迁移完成: {relation_count}/{total_relations} 个成功")

        return migrated_count > 0 or relation_count > 0
