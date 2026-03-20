"""
MemoryIndexer - 记忆向量索引器

提供向量语义检索能力，支持 FAISS/Annoy/Simple 三种索引类型。
"""

import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OllamaEmbeddingWrapper:
    """
    Ollama Embedding 包装器

    封装 Ollama API 调用，提供 embed_query() 方法。
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self._url = f"{base_url}/api/embeddings"

    def embed_query(self, text: str) -> List[float]:
        """生成单个文本的 embedding 向量"""
        try:
            response = requests.post(
                self._url,
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", [])
        except Exception:
            pass
        return [0.0] * 1024

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成多个文本的 embedding 向量"""
        return [self.embed_query(text) for text in texts]


class MemoryIndexer:
    """
    记忆向量索引器

    职责：
    1. 将记忆内容编码为向量
    2. 构建和管理向量索引
    3. 支持增量更新
    4. 持久化索引到磁盘
    """

    def __init__(
        self,
        storage_dir: Path,
        embedding_model: Any,
        index_type: str = "simple",
    ):
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        self.index_type = index_type

        self.index_file = self.storage_dir / ".index" / f"memory_{index_type}.idx"
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        self._index: Optional[Any] = None
        self._id_to_entry: Dict[int, str] = {}

        self._load_or_build()

    def _load_or_build(self):
        if self.index_file.exists():
            try:
                self._load_index()
                return
            except Exception:
                pass
        self._build_index()

    def _build_index(self):
        from ollamapilot.memory.markdown_storage import MarkdownMemoryStorage

        storage = MarkdownMemoryStorage(str(self.storage_dir))
        entries = storage.load_entries()

        if not entries:
            self._index = None
            return

        vectors = []
        self._id_to_entry = {}

        for i, entry in enumerate(entries):
            try:
                vector = self.embedding_model.embed_query(entry.content)
                vectors.append(vector)
                self._id_to_entry[i] = entry.id
            except Exception:
                continue

        if not vectors:
            self._index = None
            return

        if self.index_type == "faiss":
            self._build_faiss_index(vectors)
        elif self.index_type == "annoy":
            self._build_annoy_index(vectors)
        else:
            self._build_simple_index(vectors)

        self._save_index()

    def _build_faiss_index(self, vectors: List[List[float]]):
        try:
            import faiss

            dim = len(vectors[0])
            index = faiss.IndexFlatIP(dim)

            vectors_np = np.array(vectors).astype('float32')
            faiss.normalize_L2(vectors_np)

            index.add(vectors_np)
            self._index = index
        except ImportError:
            self._build_simple_index(vectors)

    def _build_annoy_index(self, vectors: List[List[float]]):
        try:
            from annoy import AnnoyIndex

            dim = len(vectors[0])
            index = AnnoyIndex(dim, 'angular')

            for i, vec in enumerate(vectors):
                index.add_item(i, vec)

            index.build(10)
            self._index = index
        except ImportError:
            self._build_simple_index(vectors)

    def _build_simple_index(self, vectors: List[List[float]]):
        self._index = np.array(vectors) if vectors else np.array([])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._index is None:
            return []

        try:
            query_vector = self.embedding_model.embed_query(query)
        except Exception:
            return []

        query_np = np.array(query_vector).astype('float32')

        if self.index_type == "faiss":
            return self._search_faiss(query_np, top_k)
        elif self.index_type == "annoy":
            return self._search_annoy(query_np, top_k)
        else:
            return self._search_simple(query_np, top_k)

    def _search_simple(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._index is None or len(self._index) == 0:
            return []

        query_vector = query_vector / np.linalg.norm(query_vector)
        index_norm = self._index / np.linalg.norm(self._index, axis=1, keepdims=True)

        similarities = np.dot(index_norm, query_vector)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (self._id_to_entry[int(i)], float(similarities[i]))
            for i in top_indices
            if int(i) in self._id_to_entry
        ]

    def _search_faiss(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        try:
            import faiss

            query_np = query_vector.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_np)

            distances, indices = self._index.search(query_np, top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and int(idx) in self._id_to_entry:
                    score = float(1.0 / (1.0 + distances[0][i]))
                    results.append((self._id_to_entry[int(idx)], score))

            return results
        except Exception:
            return self._search_simple(query_vector, top_k)

    def _search_annoy(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        try:
            indices = self._index.get_nns_by_vector(query_vector.tolist(), top_k)

            results = []
            for idx in indices:
                if idx in self._id_to_entry:
                    results.append((self._id_to_entry[idx], 1.0))

            return results
        except Exception:
            return []

    def add_entry(self, entry: Any):
        self._build_index()

    def _save_index(self):
        if self._index is None:
            return

        data = {
            "id_to_entry": self._id_to_entry,
            "index_type": self.index_type,
        }

        if self.index_type == "simple" and self._index is not None and len(self._index) > 0:
            data["vectors"] = self._index.tolist()

        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _load_index(self):
        with open(self.index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._id_to_entry = {int(k): v for k, v in data["id_to_entry"].items()}

        if data["index_type"] == "simple" and "vectors" in data:
            self._index = np.array(data["vectors"])
        else:
            self._build_index()