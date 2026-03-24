"""
GraphRAG 服务层 - 混合存储实现（LightRAG 增强版）

提供混合存储能力：
- 向量存储：SimpleVectorStore（SQLite + 内存向量）
- 实体索引：内存中的实体-文档映射
- 关系索引：内存中的关系存储
- 三重向量存储：实体 + 关系 + 文本块（LightRAG 增强）
- 双层检索：Local + Global（LightRAG 增强）
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
import json
import hashlib
import math
import os
from dataclasses import dataclass, asdict

from skills.graphrag.services.embedding_function import OllamaEmbeddingFunction, SafeEmbeddingFunction
from skills.graphrag.services.simple_embedding import HashEmbeddingFunction
from skills.graphrag.services.simple_vector_store import SimpleVectorStore
from skills.graphrag.services.triple_vector_store import TripleVectorStore, EntityInfo, RelationInfo
from skills.graphrag.services.dual_retrieval import DualRetrievalService, RetrievalResult, ContextFusionConfig
from skills.graphrag.services.incremental_merger import IncrementalMerger, MergeConfig


@dataclass
class Entity:
    """实体定义"""
    name: str
    type: str
    positions: List[Tuple[int, int]]  # 在文档中的位置 (start, end)
    alignment_status: Optional[str] = None  # 对齐状态: exact/fuzzy/lesser/unmatched
    similarity: float = 1.0  # 相似度（模糊匹配时）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type,
            "positions": self.positions,
            "alignment_status": self.alignment_status,
            "similarity": self.similarity,
        }


@dataclass
class Relation:
    """关系定义"""
    source: str
    target: str
    relation: str
    confidence: float
    doc_id: str


class GraphRAGService:
    """
    GraphRAG 服务（LightRAG 增强版）

    提供混合存储能力：
    - 向量存储：SimpleVectorStore（SQLite + 内存向量）
    - 实体索引：内存中的实体-文档映射
    - 关系索引：内存中的关系存储
    - 三重向量存储：实体 + 关系 + 文本块（LightRAG 新增）
    - 双层检索：Local + Global（LightRAG 新增）
    """

    def __init__(
        self,
        persist_dir: str = "./data/graphrag",
        embedding_model: Optional[str] = None,
        collection_name: str = "graphrag_default",
        enable_relation_vector: bool = True,
        enable_dual_retrieval: bool = True,
        use_llm_merge: bool = False
    ):
        """
        初始化 GraphRAG 服务

        Args:
            persist_dir: 持久化目录
            embedding_model: Ollama Embedding 模型名称（如 "qwen3-embedding:4b"）
            collection_name: 集合名称（保留参数，实际不使用）
            enable_relation_vector: 是否启用关系向量化（默认 True）
            enable_dual_retrieval: 是否启用双层检索（默认 True）
            use_llm_merge: 是否使用 LLM 智能合并（默认 False，小模型建议关闭）
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model

        # 功能开关
        self.enable_relation_vector = enable_relation_vector
        self.enable_dual_retrieval = enable_dual_retrieval

        # 根据 embedding 模型名称生成集合名称（用于隔离不同模型的向量）
        if embedding_model:
            # 清理模型名称，使其适合作为文件名/表名
            safe_name = "".join(c if c.isalnum() else "_" for c in embedding_model)
            self.collection_name = safe_name
        else:
            self.collection_name = "default"

        # 初始化向量存储（使用 SimpleVectorStore 替代 ChromaDB，按模型隔离）
        self.collection = SimpleVectorStore(
            persist_dir=persist_dir,
            collection_name=self.collection_name
        )

        # 初始化自定义 embedding 函数
        self._embedding_fn = self._get_embedding_function()

        # 实体索引：entity_name -> {doc_ids, entity_type}
        self.entity_index: Dict[str, Dict] = {}

        # 关系索引：list of Relation
        self.relations: List[Relation] = []

        # 本体定义
        self.ontology: Optional[Dict] = None

        # ========== LightRAG 增强功能 ==========

        # 三重向量存储（实体 + 关系 + 文本块）
        self.triple_store: Optional[TripleVectorStore] = None
        if enable_relation_vector:
            self.triple_store = TripleVectorStore(
                persist_dir=persist_dir,
                collection_name=self.collection_name
            )

        # 双层检索服务
        self.retrieval_service: Optional[DualRetrievalService] = None
        if enable_dual_retrieval and self.triple_store:
            self.retrieval_service = DualRetrievalService(
                triple_store=self.triple_store,
                embedding_fn=self._embedding_fn,
                config={
                    "local_top_k": 30,
                    "global_top_k": 30,
                    "chunk_top_k": 10
                }
            )

        # 增量合并服务
        self.incremental_merger: Optional[IncrementalMerger] = None
        if self.triple_store:
            self.incremental_merger = IncrementalMerger(
                triple_store=self.triple_store,
                embedding_fn=self._embedding_fn,
                llm_client=None,  # 可选，需要时传入
                config=MergeConfig(use_llm_merge=use_llm_merge)
            )

        # 加载已有数据
        self._load_index()

        # 注意：自动迁移已禁用，使用 /index 命令手动触发迁移
        # self._check_and_migrate()

    def _get_embedding_function(self):
        """获取 Embedding 函数"""
        if self.embedding_model_name:
            try:
                # 使用自定义的 Embedding 函数
                # 不再每次测试模型，让实际调用时处理错误
                embedding_fn = OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name=self.embedding_model_name,
                    timeout=120,
                    max_retries=2
                )

                # 包装为安全函数
                return SafeEmbeddingFunction(embedding_fn)

            except Exception as e:
                print(f"⚠️ Embedding 函数初始化失败: {e}，使用简单 embedding")
                return HashEmbeddingFunction(dim=384)
        else:
            # 没有指定 embedding 模型，使用简单 embedding
            return HashEmbeddingFunction(dim=384)

    def _check_and_migrate(self):
        """检查并迁移旧数据（内部方法，初始化时调用）"""
        if not self.triple_store:
            return

        # 检查是否有旧数据需要迁移
        migration_flag = self.persist_dir / f"migrated_{self.collection_name}.flag"

        if migration_flag.exists():
            return  # 已经迁移过

        if not self.entity_index and not self.relations:
            return  # 没有旧数据

        # 检查 triple_store 是否为空
        stats = self.triple_store.get_stats()
        if stats["entities"] == 0 and stats["relations"] == 0:
            print("🔄 检测到旧版数据，开始迁移到三重向量存储...")

            try:
                # 迁移时使用 raise_on_error=True，确保错误能被捕获并重试
                from .embedding_function import OllamaEmbeddingFunction, SafeEmbeddingFunction
                embedding_fn = OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name=self.embedding_model_name,
                    timeout=120,
                    max_retries=2
                )
                # 包装为安全函数，但迁移时启用错误抛出
                safe_embedding_fn = SafeEmbeddingFunction(embedding_fn, raise_on_error=True)

                self.triple_store.migrate_from_legacy(
                    entity_index=self.entity_index,
                    relations=self.relations,
                    embedding_fn=safe_embedding_fn,
                    doc_id=self.collection_name
                )

                # 标记已迁移
                with open(migration_flag, 'w') as f:
                    f.write("migrated")

                print("✅ 数据迁移完成")

            except Exception as e:
                print(f"⚠️ 数据迁移失败: {e}")

    def check_and_migrate(self) -> bool:
        """
        手动触发迁移检查（供外部调用）

        Returns:
            是否进行了迁移
        """
        if not self.triple_store:
            return False

        migration_flag = self.persist_dir / f"migrated_{self.collection_name}.flag"
        if migration_flag.exists():
            return False  # 已迁移

        if not self.entity_index and not self.relations:
            return False  # 没有旧数据

        stats = self.triple_store.get_stats()
        if stats["entities"] > 0 or stats["relations"] > 0:
            return False  # 已有数据，无需迁移

        try:
            # 迁移时使用 raise_on_error=True，确保错误能被捕获并重试
            from .embedding_function import OllamaEmbeddingFunction, SafeEmbeddingFunction
            embedding_fn = OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name=self.embedding_model_name,
                timeout=120,
                max_retries=2
            )
            # 包装为安全函数，但迁移时启用错误抛出
            safe_embedding_fn = SafeEmbeddingFunction(embedding_fn, raise_on_error=True)

            self.triple_store.migrate_from_legacy(
                entity_index=self.entity_index,
                relations=self.relations,
                embedding_fn=safe_embedding_fn,
                doc_id=self.collection_name
            )

            # 标记已迁移
            with open(migration_flag, 'w') as f:
                f.write("migrated")

            return True

        except Exception as e:
            print(f"⚠️ 数据迁移失败: {e}")
            return False

    def set_ontology(self, ontology: Dict):
        """设置本体定义"""
        self.ontology = ontology
        self._save_ontology()

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        entities: Optional[List[Entity]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        添加文档到图谱

        Args:
            text: 文档文本
            doc_id: 文档ID（可选，自动生成）
            metadata: 元数据
            entities: 预抽取的实体列表
            embedding: 预生成的embedding向量（可选，如不提供则自动生成）

        Returns:
            文档ID
        """
        import time

        if doc_id is None:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]

        # 1. 准备元数据
        doc_metadata = metadata or {}
        doc_metadata["doc_id"] = doc_id

        if entities:
            doc_metadata["entities"] = json.dumps([
                {"name": e.name, "type": e.type} for e in entities
            ])

        # 2. 生成 embedding（如果未提供）
        start_time = time.time()
        if embedding is None:
            try:
                if self._embedding_fn:
                    embeddings = self._embedding_fn([text])
                    embedding = embeddings[0] if embeddings else None
                else:
                    embedding = None
            except Exception as e:
                print(f"      ⚠️ Embedding 生成失败: {e}")
                embedding = None

        embed_time = time.time() - start_time

        # 3. 存储到向量存储
        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[doc_metadata],
                embeddings=[embedding] if embedding else None
            )

            # 同时添加到 triple_store 的 chunk 存储
            if self.triple_store and embedding:
                self.triple_store.add_chunk(
                    chunk_id=doc_id,
                    text=text,
                    embedding=embedding,
                    metadata=doc_metadata
                )

            elapsed = time.time() - start_time
            if elapsed > 5:
                print(f"      ⚠️ 存储耗时较长: {elapsed:.1f}s (embedding: {embed_time:.1f}s)")
        except Exception as e:
            print(f"      ❌ 存储失败: {e}")
            raise

        # 4. 更新实体索引
        if entities:
            for entity in entities:
                self._index_entity(entity, doc_id)

                # 添加到 triple_store 的实体存储
                if self.triple_store:
                    self._add_entity_to_triple_store(entity, doc_id)

        # 5. 推断关系
        if entities and len(entities) > 1:
            self._infer_relations(entities, doc_id)

        return doc_id

    def add_documents_batch(
        self,
        documents: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> int:
        """
        批量添加文档（优化版，避免逐个调用 embedding API）

        Args:
            documents: 文档列表，每个文档包含:
                - text: 文档文本
                - doc_id: 文档ID
                - metadata: 元数据
                - entities: 实体列表
                - embedding: 预生成的 embedding
                - relation_texts: 预生成的关系描述列表
            progress_callback: 进度回调

        Returns:
            成功添加的文档数
        """
        total = len(documents)
        success_count = 0

        for idx, doc in enumerate(documents):
            try:
                entities = doc.get("entities", [])
                relation_texts = doc.get("relation_texts", [])

                self.collection.add(
                    ids=[doc["doc_id"]],
                    documents=[doc["text"]],
                    metadatas=[doc.get("metadata", {})],
                    embeddings=[doc["embedding"]] if doc.get("embedding") else None
                )

                if self.triple_store and doc.get("embedding"):
                    self.triple_store.add_chunk(
                        chunk_id=doc["doc_id"],
                        text=doc["text"],
                        embedding=doc["embedding"],
                        metadata=doc.get("metadata", {})
                    )

                for entity, embedding in zip(entities, doc.get("entity_embeddings", [])):
                    self._index_entity(entity, doc["doc_id"])
                    if self.triple_store:
                        entity_text = f"{entity.name} {entity.type}"
                        entity_info = EntityInfo(
                            name=entity.name,
                            entity_type=entity.type,
                            description=entity_text,
                            source_ids=[doc["doc_id"]]
                        )
                        self.triple_store.add_entity(entity_info, embedding)

                for relation_text, embedding in zip(relation_texts, doc.get("relation_embeddings", [])):
                    relation = Relation(
                        source=doc["doc_id"],
                        target="",
                        relation="CO_OCCUR",
                        confidence=0.5,
                        doc_id=doc["doc_id"]
                    )
                    relation_info = RelationInfo(
                        source=relation.source,
                        target=relation.target,
                        relation=relation.relation,
                        description=relation_text,
                        confidence=relation.confidence,
                        source_ids=[doc["doc_id"]]
                    )
                    self.triple_store.add_relation(relation_info, embedding)
                    self.relations.append(relation)

                success_count += 1

                if progress_callback and (idx + 1) % 10 == 0:
                    progress_callback(idx + 1, total, f"保存 {idx + 1}/{total}")

            except Exception as e:
                print(f"      ❌ 保存文档 {doc.get('doc_id', 'unknown')} 失败: {e}")

        return success_count

    def _index_entity(self, entity: Entity, doc_id: str):
        """索引实体"""
        if entity.name not in self.entity_index:
            self.entity_index[entity.name] = {
                "type": entity.type,
                "doc_ids": set()
            }
        self.entity_index[entity.name]["doc_ids"].add(doc_id)

    def _add_entity_to_triple_store(self, entity: Entity, doc_id: str):
        """添加实体到三重向量存储"""
        if not self.triple_store:
            return

        try:
            entity_text = f"{entity.name} {entity.type}"
            embedding = self._embedding_fn([entity_text])[0]

            entity_info = EntityInfo(
                name=entity.name,
                entity_type=entity.type,
                description=entity_text,
                source_ids=[doc_id]
            )

            self.triple_store.add_entity(entity_info, embedding)

        except Exception as e:
            print(f"⚠️ 实体向量化失败: {e}")

    def _infer_relations(self, entities: List[Entity], doc_id: str):
        """推断实体间关系（基于共现）"""
        entity_names = [e.name for e in entities]

        # 简单的共现关系
        for i, source in enumerate(entity_names):
            for target in entity_names[i+1:]:
                relation = Relation(
                    source=source,
                    target=target,
                    relation="CO_OCCUR",  # 共现关系
                    confidence=0.5,  # 基础置信度
                    doc_id=doc_id
                )
                self.relations.append(relation)

                # 添加到 triple_store 的关系存储
                if self.triple_store:
                    self._add_relation_to_triple_store(relation, doc_id)

    def _add_relation_to_triple_store(self, relation: Relation, doc_id: str):
        """添加关系到三重向量存储"""
        if not self.triple_store:
            return

        try:
            relation_desc = f"{relation.source} 与 {relation.target} 相关"
            embedding = self._embedding_fn([relation_desc])[0]

            relation_info = RelationInfo(
                source=relation.source,
                target=relation.target,
                relation=relation.relation,
                description=relation_desc,
                confidence=relation.confidence,
                source_ids=[doc_id]
            )

            self.triple_store.add_relation(relation_info, embedding)

        except Exception as e:
            print(f"⚠️ 关系向量化失败: {e}")

    def get_entity_documents(self, entity_name: str) -> List[str]:
        """获取包含实体的文档ID列表"""
        if entity_name in self.entity_index:
            return list(self.entity_index[entity_name]["doc_ids"])
        return []

    def get_relations(
        self,
        entity_name: str,
        hops: int = 1
    ) -> List[Dict]:
        """
        获取实体的关系

        Args:
            entity_name: 实体名
            hops: 遍历跳数（目前只支持1跳）

        Returns:
            关系列表
        """
        results = []

        for rel in self.relations:
            if rel.source == entity_name:
                results.append({
                    "source": rel.source,
                    "target": rel.target,
                    "relation": rel.relation,
                    "confidence": rel.confidence
                })
            elif rel.target == entity_name:
                # 反向关系
                results.append({
                    "source": rel.target,
                    "target": rel.source,
                    "relation": rel.relation,
                    "confidence": rel.confidence
                })

        return results

    def rerank_documents(
        self,
        query: str,
        doc_ids: List[str],
        n_results: int = 5
    ) -> List[Dict]:
        """
        对候选文档进行向量重排序

        Args:
            query: 查询文本
            doc_ids: 候选文档ID列表
            n_results: 返回结果数量

        Returns:
            排序后的文档列表
        """
        if not doc_ids:
            return []

        # 获取候选文档的内容
        try:
            candidates = self.collection.get(ids=doc_ids, include=["embeddings"])
        except Exception:
            return []

        if not candidates or not candidates["ids"]:
            return []

        # 手动计算相似度进行重排序
        try:
            # 获取 query embedding
            if self._embedding_fn:
                query_embeddings = self._embedding_fn([query])
                query_embedding = query_embeddings[0] if query_embeddings else None
            else:
                query_embedding = None

            if query_embedding:
                ranked_docs = []
                for i, doc_id in enumerate(candidates["ids"]):
                    doc_embedding = candidates["embeddings"][i] if candidates["embeddings"] else None

                    if doc_embedding:
                        # 计算余弦相似度
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        score = similarity
                    else:
                        score = 0.0

                    ranked_docs.append({
                        "id": doc_id,
                        "content": candidates["documents"][i],
                        "metadata": candidates["metadatas"][i],
                        "score": score
                    })

                # 按相似度排序
                ranked_docs.sort(key=lambda x: x["score"], reverse=True)
                return ranked_docs[:n_results]
            else:
                # 没有 embedding 函数，返回原始顺序
                return [
                    {
                        "id": candidates["ids"][i],
                        "content": candidates["documents"][i],
                        "metadata": candidates["metadatas"][i],
                        "score": 1.0
                    }
                    for i in range(len(candidates["ids"]))
                ][:n_results]

        except Exception as e:
            print(f"⚠️ 重排序失败: {e}")
            # 失败时返回原始顺序
            return [
                {
                    "id": candidates["ids"][i],
                    "content": candidates["documents"][i],
                    "metadata": candidates["metadatas"][i],
                    "score": 1.0
                }
                for i in range(len(candidates["ids"]))
            ][:n_results]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def vector_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """纯向量检索（回退用）"""
        try:
            # 生成 query embedding
            if self._embedding_fn:
                query_embeddings = self._embedding_fn([query])
                query_embedding = query_embeddings[0] if query_embeddings else None
            else:
                query_embedding = None

            if query_embedding:
                # 使用 embedding 查询
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                # 没有 embedding，返回空结果
                return []

            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]
                })

            return docs
        except Exception as e:
            print(f"⚠️ 向量检索失败: {e}")
            return []

    def dual_retrieval_search(
        self,
        query: str,
        mode: str = "mix",
        local_top_k: int = 30,
        global_top_k: int = 30
    ) -> RetrievalResult:
        """
        双层检索（LightRAG 增强）

        Args:
            query: 查询文本
            mode: 检索模式 ("local" | "global" | "mix")
            local_top_k: Local 检索返回数量
            global_top_k: Global 检索返回数量

        Returns:
            检索结果
        """
        if not self.retrieval_service:
            print("⚠️ 双层检索服务未启用，返回空结果")
            return RetrievalResult()

        try:
            return self.retrieval_service.hybrid_retrieval(
                query=query,
                mode=mode,
                local_top_k=local_top_k,
                global_top_k=global_top_k
            )
        except Exception as e:
            print(f"⚠️ 双层检索失败: {e}")
            return RetrievalResult()

    def _save_index(self):
        """保存索引到文件（按模型隔离）"""
        index_path = self.persist_dir / f"index_{self.collection_name}.json"
        index_data = {
            "entity_index": {
                name: {
                    "type": info["type"],
                    "doc_ids": list(info["doc_ids"])
                }
                for name, info in self.entity_index.items()
            },
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "confidence": r.confidence,
                    "doc_id": r.doc_id
                }
                for r in self.relations
            ]
        }
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        """从文件加载索引（按模型隔离）"""
        index_path = self.persist_dir / f"index_{self.collection_name}.json"
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.entity_index = {
                    name: {
                        "type": info["type"],
                        "doc_ids": set(info["doc_ids"])
                    }
                    for name, info in data.get("entity_index", {}).items()
                }

                self.relations = [
                    Relation(**r) for r in data.get("relations", [])
                ]
            except Exception as e:
                print(f"⚠️ 索引加载失败: {e}")

    def _save_ontology(self):
        """保存本体定义"""
        if self.ontology:
            ontology_path = self.persist_dir / "ontology.json"
            with open(ontology_path, 'w', encoding='utf-8') as f:
                json.dump(self.ontology, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_documents": self.collection.count(),
            "total_entities": len(self.entity_index),
            "total_relations": len(self.relations),
            "entity_types": list(set(
                info["type"] for info in self.entity_index.values()
            )) if self.entity_index else [],
            "features": {
                "relation_vector": self.enable_relation_vector,
                "dual_retrieval": self.enable_dual_retrieval
            }
        }

        # 添加 triple_store 统计
        if self.triple_store:
            triple_stats = self.triple_store.get_stats()
            stats["triple_store"] = triple_stats

        return stats

    def enhanced_search(
        self,
        query: str,
        query_entities: List[Dict],
        n_results: int = 5,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        实体-关系增强检索

        Args:
            query: 原始查询
            query_entities: 查询中提取的实体列表
            n_results: 返回结果数量
            max_hops: 最大关系遍历跳数

        Returns:
            检索结果字典
        """
        # 步骤 1: 实体索引查询
        candidate_docs = set()
        entity_doc_map = {}

        for entity in query_entities:
            entity_name = entity["name"]
            doc_ids = self.get_entity_documents(entity_name)
            candidate_docs.update(doc_ids)
            entity_doc_map[entity_name] = doc_ids

        # 步骤 2: 关系遍历（扩展候选集）
        related_entities = set()
        traversed_relations = []

        for entity in query_entities:
            entity_name = entity["name"]
            # 查找 1 跳关系
            relations = self.get_relations(entity_name, hops=1)

            for rel in relations:
                related_entities.add(rel["target"])
                traversed_relations.append(rel)

                # 找到相关实体的文档
                related_docs = self.get_entity_documents(rel["target"])
                candidate_docs.update(related_docs)

        # 步骤 3: 向量重排序
        ranked_docs = self.rerank_documents(
            query,
            list(candidate_docs),
            n_results=n_results
        )

        return {
            "query_entities": query_entities,
            "related_entities": list(related_entities),
            "relations": traversed_relations,
            "documents": ranked_docs
        }
