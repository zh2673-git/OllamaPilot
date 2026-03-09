"""
GraphRAG 服务层 - 混合存储实现

提供混合存储能力：
- 向量存储：SimpleVectorStore（SQLite + 内存向量）
- 实体索引：内存中的实体-文档映射
- 关系索引：内存中的关系存储
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
import json
import hashlib
import math
from dataclasses import dataclass, asdict

from skills.graphrag.services.embedding_function import OllamaEmbeddingFunction, SafeEmbeddingFunction
from skills.graphrag.services.simple_embedding import HashEmbeddingFunction
from skills.graphrag.services.simple_vector_store import SimpleVectorStore


@dataclass
class Entity:
    """实体定义"""
    name: str
    type: str
    positions: List[Tuple[int, int]]  # 在文档中的位置


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
    GraphRAG 服务

    提供混合存储能力：
    - 向量存储：SimpleVectorStore（SQLite + 内存向量）
    - 实体索引：内存中的实体-文档映射
    - 关系索引：内存中的关系存储
    """

    def __init__(
        self,
        persist_dir: str = "./data/graphrag",
        embedding_model: Optional[str] = None,
        collection_name: str = "graphrag_default"
    ):
        """
        初始化 GraphRAG 服务

        Args:
            persist_dir: 持久化目录
            embedding_model: Ollama Embedding 模型名称（如 "qwen3-embedding:4b"）
            collection_name: 集合名称（保留参数，实际不使用）
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model

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

        # 加载已有数据
        self._load_index()

    def _get_embedding_function(self):
        """获取 Embedding 函数"""
        if self.embedding_model_name:
            try:
                import requests

                # 先测试 Ollama 服务是否可用
                print(f"🔄 测试 Embedding 模型 {self.embedding_model_name}...")
                try:
                    response = requests.post(
                        "http://localhost:11434/api/embeddings",
                        json={"model": self.embedding_model_name, "prompt": "test"},
                        timeout=60
                    )
                    if response.status_code == 200:
                        print(f"✅ Embedding 模型 {self.embedding_model_name} 已就绪")
                    else:
                        print(f"⚠️ Embedding 模型测试失败: {response.status_code}")
                except requests.exceptions.Timeout:
                    print(f"⚠️ Embedding 模型加载超时（60秒），模型可能太大或 Ollama 服务繁忙")
                    print(f"   建议：手动运行 'ollama run {self.embedding_model_name}' 预热模型")
                except Exception as e:
                    print(f"⚠️ Embedding 模型测试失败: {e}")

                # 使用自定义的 Embedding 函数
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
            print("⚠️ 未指定 Embedding 模型，使用简单哈希 embedding（语义检索能力受限）")
            return HashEmbeddingFunction(dim=384)

    def set_ontology(self, ontology: Dict):
        """设置本体定义"""
        self.ontology = ontology
        self._save_ontology()

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        entities: Optional[List[Entity]] = None
    ) -> str:
        """
        添加文档到图谱

        Args:
            text: 文档文本
            doc_id: 文档ID（可选，自动生成）
            metadata: 元数据
            entities: 预抽取的实体列表

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

        # 2. 生成 embedding
        start_time = time.time()
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

        # 5. 推断关系
        if entities and len(entities) > 1:
            self._infer_relations(entities, doc_id)

        return doc_id

    def _index_entity(self, entity: Entity, doc_id: str):
        """索引实体"""
        if entity.name not in self.entity_index:
            self.entity_index[entity.name] = {
                "type": entity.type,
                "doc_ids": set()
            }
        self.entity_index[entity.name]["doc_ids"].add(doc_id)

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
        return {
            "total_documents": self.collection.count(),
            "total_entities": len(self.entity_index),
            "total_relations": len(self.relations),
            "entity_types": list(set(
                info["type"] for info in self.entity_index.values()
            )) if self.entity_index else []
        }

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
