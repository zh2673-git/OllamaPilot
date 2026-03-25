"""
简单向量存储

纯 Python 实现，不依赖 ChromaDB，避免 Windows 上的兼容性问题。
使用 SQLite 存储元数据，内存中存储向量。
"""

import sqlite3
import json
import hashlib
import math
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Document:
    """文档"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SimpleVectorStore:
    """
    简单向量存储

    使用 SQLite 存储文档元数据，内存中存储向量。
    支持按 Embedding 模型隔离存储。
    不依赖 ChromaDB，避免 Windows 上的兼容性问题。
    """

    def __init__(self, persist_dir: str = "./data/graphrag", collection_name: str = "default"):
        """
        初始化

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称（用于区分不同的 Embedding 模型）
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # 初始化 SQLite（按集合分表）
        self.db_path = self.persist_dir / "vectors.db"
        self._init_db()

        # 内存中的向量存储
        self.vectors: Dict[str, List[float]] = {}

        # 加载已有数据
        self._load_vectors()

        # 批量模式：禁用自动保存以提升性能
        self._auto_persist = True

    def set_auto_persist(self, enabled: bool):
        """设置是否自动持久化向量"""
        self._auto_persist = enabled

    def persist(self):
        """手动持久化向量到文件"""
        self._save_vectors()

    def _init_db(self):
        """初始化数据库（按集合创建表）"""
        with sqlite3.connect(self.db_path) as conn:
            # 为每个集合创建独立的表
            table_name = self._get_table_name()
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            # 创建集合元数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_meta (
                    name TEXT PRIMARY KEY,
                    doc_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_table_name(self) -> str:
        """获取当前集合的表名"""
        # 清理集合名称，确保是有效的 SQL 标识符
        safe_name = "".join(c if c.isalnum() else "_" for c in self.collection_name)
        return f"docs_{safe_name}"

    def _load_vectors(self):
        """加载向量（向量存储在单独的 JSON 文件中，按集合隔离）"""
        vectors_path = self.persist_dir / f"vectors_{self.collection_name}.json"
        if vectors_path.exists():
            try:
                with open(vectors_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.vectors = {k: v for k, v in data.items()}
            except Exception as e:
                print(f"⚠️ 加载向量失败: {e}")
                self.vectors = {}

    def _save_vectors(self):
        """保存向量"""
        vectors_path = self.persist_dir / f"vectors_{self.collection_name}.json"
        try:
            with open(vectors_path, 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)
        except Exception as e:
            print(f"⚠️ 保存向量失败: {e}")

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        添加文档

        Args:
            ids: 文档ID列表
            documents: 文档文本列表
            metadatas: 元数据列表
            embeddings: embedding 向量列表
        """
        table_name = self._get_table_name()
        with sqlite3.connect(self.db_path) as conn:
            for i, doc_id in enumerate(ids):
                text = documents[i]
                metadata = metadatas[i]

                # 插入或更新文档
                conn.execute(
                    f"INSERT OR REPLACE INTO {table_name} (id, text, metadata) VALUES (?, ?, ?)",
                    (doc_id, text, json.dumps(metadata, ensure_ascii=False))
                )

                # 存储向量
                if embeddings and i < len(embeddings):
                    self.vectors[doc_id] = embeddings[i]

            conn.commit()

        # 保存向量（仅在自动保存开启时）
        if self._auto_persist:
            self._save_vectors()

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> Dict:
        """
        获取文档

        Args:
            ids: 文档ID列表
            include: 包含的字段

        Returns:
            文档数据字典
        """
        result = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": []
        }

        include_embeddings = include is None or "embeddings" in include
        table_name = self._get_table_name()

        with sqlite3.connect(self.db_path) as conn:
            for doc_id in ids:
                row = conn.execute(
                    f"SELECT text, metadata FROM {table_name} WHERE id = ?",
                    (doc_id,)
                ).fetchone()

                if row:
                    result["ids"].append(doc_id)
                    result["documents"].append(row[0])
                    result["metadatas"].append(json.loads(row[1]))

                    if include_embeddings:
                        result["embeddings"].append(self.vectors.get(doc_id))

        return result

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5
    ) -> Dict:
        """
        查询文档

        Args:
            query_embeddings: 查询向量
            query_texts: 查询文本（不使用，仅保持接口一致）
            n_results: 返回结果数量

        Returns:
            查询结果
        """
        if not query_embeddings or not self.vectors:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query_vec = query_embeddings[0]

        # 计算余弦相似度
        similarities = []
        for doc_id, vec in self.vectors.items():
            similarity = self._cosine_similarity(query_vec, vec)
            similarities.append((doc_id, similarity))

        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:n_results]

        # 获取文档内容
        result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        table_name = self._get_table_name()
        with sqlite3.connect(self.db_path) as conn:
            for doc_id, similarity in top_k:
                row = conn.execute(
                    f"SELECT text, metadata FROM {table_name} WHERE id = ?",
                    (doc_id,)
                ).fetchone()

                if row:
                    result["ids"][0].append(doc_id)
                    result["documents"][0].append(row[0])
                    result["metadatas"][0].append(json.loads(row[1]))
                    result["distances"][0].append(1 - similarity)  # 转换为距离

        return result

    def count(self) -> int:
        """获取文档数量"""
        table_name = self._get_table_name()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return row[0] if row else 0

    def get_all_ids(self) -> List[str]:
        """获取所有文档 ID"""
        table_name = self._get_table_name()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"SELECT id FROM {table_name}").fetchall()
            return [row[0] for row in rows]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
