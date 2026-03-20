"""
MemoryManager - 记忆管理器

被 Context 统管的记忆系统，提供向量语义检索和文件加载能力。

V0.5.0 重构：简化架构，直接使用文件系统存储
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ollamapilot.memory.indexer import MemoryIndexer, OllamaEmbeddingWrapper
from ollamapilot.memory.types import MemoryType, MemoryEntry


class MemoryManager:
    """
    记忆管理器 - 被 Context 统管

    职责：
    1. 记忆的语义检索
    2. 记忆内容的加载
    3. 向 Context 提供记忆片段

    注意：这不是独立系统，而是 Context 的一部分
    """

    def __init__(
        self,
        workspace_dir: Path,
        embedding_model: Any = None,
        enable_vector_search: bool = True,
        similarity_threshold: float = 0.7,
    ):
        self.workspace = Path(workspace_dir)
        self.enable_vector_search = enable_vector_search
        self.similarity_threshold = similarity_threshold
        self._embedding_model = embedding_model

        self._indexer: Optional[MemoryIndexer] = None
        if enable_vector_search and embedding_model:
            self._init_indexer()

        self._memory_cache: Optional[List[MemoryEntry]] = None

    def _init_indexer(self):
        try:
            memory_dir = self.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)

            embedding = self._embedding_model
            if isinstance(embedding, str):
                embedding = OllamaEmbeddingWrapper(model_name=embedding)

            self._indexer = MemoryIndexer(
                storage_dir=memory_dir,
                embedding_model=embedding,
                index_type="simple"
            )
        except Exception:
            self._indexer = None

    def recall(self, query: str, top_k: int = 5) -> List[str]:
        results = self.search(query, top_k=top_k)
        return [r.content for r in results]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Any]:
        results: List[Any] = []

        if self._indexer and self.enable_vector_search:
            try:
                vector_results = self._indexer.search(query, top_k=top_k)
                for memory_id, score in vector_results:
                    content = self._load_memory_content(memory_id)
                    if content:
                        results.append(SearchResult(
                            content=content,
                            score=score,
                            source="vector"
                        ))
            except Exception:
                pass

        keyword_results = self._keyword_search(query, top_k=top_k)
        for result in keyword_results:
            if not any(r.content == result.content for r in results):
                results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> List[Any]:
        entries = self._load_all_entries()
        query_words = set(query.lower().split())
        scored = []

        for entry in entries:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append(SearchResult(
                    content=entry.content,
                    score=score * entry.importance,
                    source="keyword"
                ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _load_memory_content(self, memory_id: str) -> str:
        if memory_id == "main":
            path = self.workspace / "MEMORY.md"
        else:
            path = self.workspace / "memory" / f"{memory_id}.md"

        if path.exists():
            return path.read_text(encoding='utf-8')
        return ""

    def _load_all_entries(self) -> List[MemoryEntry]:
        if self._memory_cache is not None:
            return self._memory_cache

        entries = []
        memory_md = self.workspace / "MEMORY.md"
        if memory_md.exists():
            try:
                content = memory_md.read_text(encoding='utf-8')
                entries.extend(self._parse_memory_file(content, "main"))
            except Exception:
                pass

        memory_dir = self.workspace / "memory"
        if memory_dir.exists():
            for md_file in memory_dir.glob("*.md"):
                if md_file.stem == "projects":
                    continue
                try:
                    content = md_file.read_text(encoding='utf-8')
                    entries.extend(self._parse_memory_file(content, md_file.stem))
                except Exception:
                    continue

        self._memory_cache = entries
        return entries

    def _parse_memory_file(self, content: str, entry_id: str) -> List[MemoryEntry]:
        """解析记忆文件内容"""
        entries = []

        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            body = frontmatter_match.group(2)

            metadata = {}
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

            entry_type = MemoryType.SEMANTIC
            if 'type' in metadata:
                try:
                    entry_type = MemoryType(metadata['type'])
                except Exception:
                    pass

            entry = MemoryEntry(
                id=entry_id,
                type=entry_type,
                content=body.strip(),
                metadata=metadata,
                importance=float(metadata.get('importance', 0.5)),
            )
            entries.append(entry)
        else:
            entry = MemoryEntry(
                id=entry_id,
                type=MemoryType.SEMANTIC,
                content=content.strip(),
                importance=0.5,
            )
            entries.append(entry)

        return entries

    def invalidate_cache(self):
        self._memory_cache = None

    def _save_entry(self, entry: MemoryEntry):
        """保存记忆条目到文件"""
        try:
            memory_dir = self.workspace / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)

            file_path = memory_dir / f"{entry.id}.md"

            frontmatter = f"""---
id: {entry.id}
type: {entry.type.value}
timestamp: {datetime.now().isoformat()}
importance: {entry.importance}
---

{entry.content}"""

            file_path.write_text(frontmatter, encoding='utf-8')
            self.invalidate_cache()
        except Exception:
            pass

    def record_skill_usage(self, skill_name: str, context: str = ""):
        """记录 Skill 使用"""
        entry_id = f"skill_usage_{skill_name}"

        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.PROCEDURAL,
            content=f"Skill '{skill_name}' 使用记录",
            metadata={
                "skill_name": skill_name,
                "context": context,
            },
            importance=0.5,
        )

        self._save_entry(entry)


class SearchResult:
    def __init__(self, content: str, score: float, source: str):
        self.content = content
        self.score = score
        self.source = source