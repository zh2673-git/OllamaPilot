"""
SystemMemory - 统一版系统记忆

基于 LangChain 1.0 的向量存储实现，支持可选的语义检索。
结合关键词匹配和语义检索，提供混合搜索能力。

特性：
- 可选向量检索：可以启用或禁用
- 混合检索：关键词匹配 + 向量语义检索
- 多维度评分：相关性、重要性、时效性、访问频率
- Markdown 持久化：人类可读的存储格式
- 向后兼容：完全兼容旧版 SystemMemory 接口
"""

import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ollamapilot.memory.types import MemoryType, MemoryEntry


class SystemMemory:
    """
    系统记忆 - 跨会话的长期记忆
    
    统一版本，支持可选的向量语义检索：
    - 禁用向量时：轻量级，快速启动，仅关键词检索
    - 启用向量时：智能语义检索，混合搜索
    
    存储结构（Markdown + JSON Cache）：
    - memories.md: 主记忆文件（人类可读的 Markdown 格式）
    - memories.md.json: 自动缓存（性能优化）
    """

    def __init__(
        self,
        storage_dir: str = "./data/memories",
        use_markdown: bool = True,
        embedding_model: Optional[Any] = None,
        enable_vector_search: bool = True,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        verbose: bool = False,
    ):
        """
        初始化系统记忆

        Args:
            storage_dir: 存储目录
            use_markdown: 是否使用 Markdown 格式存储
            embedding_model: 嵌入模型（用于向量检索，可选）
            enable_vector_search: 是否启用向量检索
            vector_weight: 向量检索结果权重
            keyword_weight: 关键词检索结果权重
            verbose: 是否显示详细日志
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_markdown = use_markdown
        self.verbose = verbose

        # 向量检索配置
        self.enable_vector_search = enable_vector_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._embedding_model = embedding_model
        self._vector_store_type: Optional[str] = None
        self._memory_access_count: Dict[str, int] = {}

        # 初始化存储
        self._init_storage()

        # 缓存
        self._cache: Dict[MemoryType, List[MemoryEntry]] = {}
        self._load_cache()

        # 如果启用了向量搜索，检测可用的向量存储
        if self.enable_vector_search:
            self._detect_vector_store()

    def _init_storage(self):
        """初始化存储文件"""
        if self.use_markdown:
            self.md_file = self.storage_dir / "memories.md"
            self.cache_file = self.storage_dir / "memories.md.json"
            self._init_markdown_storage()
        else:
            for mem_type in MemoryType:
                file_path = self.storage_dir / f"{mem_type.value}.jsonl"
                if not file_path.exists():
                    file_path.touch()

    def _init_markdown_storage(self):
        """初始化 Markdown 存储文件"""
        if not self.md_file.exists():
            template = self._generate_markdown_template()
            self.md_file.write_text(template, encoding='utf-8')

        if not self.cache_file.exists() and self.md_file.exists():
            self._rebuild_cache()

    def _generate_markdown_template(self) -> str:
        """生成 Markdown 模板文件"""
        return """# System Memory - 系统记忆

> 此文件由 OllamaPilot 自动生成和维护
> 您可以手动编辑此文件来添加、修改或删除记忆
> 格式：YAML frontmatter + Markdown 内容

---

## 记忆条目格式说明

每个记忆条目格式如下：

```markdown
---
id: "唯一标识"
type: "semantic"  # semantic(语义) | procedural(程序) | episodic(情景)
category: "general"  # 分类
timestamp: "2024-01-15T10:30:00"
importance: 0.9  # 0-1 重要性评分
---

记忆内容写在这里，支持 Markdown 格式。
可以是多行文本。
```

---

## 语义记忆 (Semantic)

用户偏好、重要事实

---

## 程序记忆 (Procedural)

Skill 使用模式、操作习惯

---

## 情景记忆 (Episodic)

重要对话摘要、历史事件

---

"""

    def _detect_vector_store(self):
        """检测可用的向量存储类型"""
        if not self.enable_vector_search:
            return

        try:
            from langchain_community.vectorstores import FAISS
            self._vector_store_type = "faiss"
            if self.verbose:
                print(f"🔍 向量存储类型: FAISS")
        except ImportError:
            try:
                from langchain_chroma import Chroma
                self._vector_store_type = "chroma"
                if self.verbose:
                    print(f"🔍 向量存储类型: Chroma")
            except ImportError:
                self._vector_store_type = None
                if self.verbose:
                    print(f"⚠️ 未找到 LangChain 向量存储库，向量检索将不可用")

    def _get_embedding_model(self) -> Optional[Any]:
        """获取嵌入模型"""
        if self._embedding_model is not None:
            return self._embedding_model

        if not self.enable_vector_search:
            return None

        # 尝试创建默认嵌入模型
        try:
            from langchain_ollama import OllamaEmbeddings
            self._embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")
            return self._embedding_model
        except ImportError:
            pass

        try:
            from ollamapilot.infra.embeddings import EmbeddingManager
            manager = EmbeddingManager()
            self._embedding_model = manager.get_embedding_model()
            return self._embedding_model
        except Exception:
            pass

        return None

    def _build_vector_store(self) -> Optional[Any]:
        """构建 LangChain 向量存储"""
        if not self.enable_vector_search or self._vector_store_type is None:
            return None

        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            return None

        try:
            # 收集所有记忆内容
            documents = []
            metadatas = []
            ids = []

            for mem_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC]:
                for entry in self._cache.get(mem_type, []):
                    documents.append(entry.content)
                    metadatas.append({
                        "id": entry.id,
                        "type": entry.type.value,
                        "importance": entry.importance,
                        "timestamp": entry.timestamp.isoformat(),
                    })
                    ids.append(entry.id)

            if not documents:
                return None

            # 使用 LangChain 的 FAISS
            if self._vector_store_type == "faiss":
                from langchain_community.vectorstores import FAISS
                return FAISS.from_texts(
                    texts=documents,
                    embedding=embedding_model,
                    metadatas=metadatas,
                    ids=ids
                )
            elif self._vector_store_type == "chroma":
                from langchain_chroma import Chroma
                return Chroma.from_texts(
                    texts=documents,
                    embedding=embedding_model,
                    metadatas=metadatas,
                    ids=ids
                )

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 构建向量存储失败: {e}")

        return None

    # ========== 存储和加载方法（与原版相同）==========

    def _load_cache(self):
        """加载缓存"""
        for mem_type in MemoryType:
            self._cache[mem_type] = self._load_entries(mem_type)

    def _load_entries(self, mem_type: MemoryType) -> List[MemoryEntry]:
        """加载指定类型的记忆条目"""
        if self.use_markdown:
            return self._load_entries_from_markdown(mem_type)
        else:
            return self._load_entries_from_jsonl(mem_type)

    def _load_entries_from_jsonl(self, mem_type: MemoryType) -> List[MemoryEntry]:
        """从 JSONL 加载条目"""
        file_path = self.storage_dir / f"{mem_type.value}.jsonl"
        entries = []

        if not file_path.exists():
            return entries

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(MemoryEntry.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

        return entries

    def _load_entries_from_markdown(self, mem_type: MemoryType) -> List[MemoryEntry]:
        """从 Markdown 加载条目"""
        all_entries = self._parse_markdown_file()
        return [e for e in all_entries if e.type == mem_type]

    def _parse_markdown_file(self) -> List[MemoryEntry]:
        """解析 Markdown 文件中的所有条目"""
        if not self.md_file.exists():
            return []

        content = self.md_file.read_text(encoding='utf-8')
        entries = []

        # 匹配包含 id 和 type 的 YAML frontmatter 块
        # 要求必须包含 id: 和 type: 字段才认为是有效的记忆条目
        pattern = r'---\s*\n((?:(?!---).)*?id:\s*\S+.*?type:\s*\S+.*?)\n---\s*\n?(.*?)(?=\n---|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            yaml_text = match.group(1)
            entry_content = match.group(2).strip()

            full_text = f"---\n{yaml_text}\n---\n\n{entry_content}"
            entry = self._markdown_to_entry(full_text)
            if entry:
                entries.append(entry)

        return entries

    def _markdown_to_entry(self, markdown_text: str) -> Optional[MemoryEntry]:
        """将 Markdown 转换为记忆条目"""
        try:
            pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)$'
            match = re.match(pattern, markdown_text.strip(), re.DOTALL)

            if not match:
                return None

            yaml_text = match.group(1)
            content = match.group(2).strip()

            frontmatter = {}
            for line in yaml_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')

                    if value.lower() in ('true', 'false'):
                        frontmatter[key] = value.lower() == 'true'
                    elif value.replace('.', '').isdigit():
                        frontmatter[key] = float(value) if '.' in value else int(value)
                    else:
                        frontmatter[key] = value

            if not frontmatter.get('id') or not frontmatter.get('type'):
                return None

            standard_fields = {'id', 'type', 'timestamp', 'importance'}
            metadata = {k: v for k, v in frontmatter.items() if k not in standard_fields}

            return MemoryEntry(
                id=frontmatter['id'],
                type=MemoryType(frontmatter['type']),
                content=content,
                metadata=metadata,
                timestamp=datetime.fromisoformat(frontmatter.get('timestamp', datetime.now().isoformat())),
                importance=frontmatter.get('importance', 1.0),
            )
        except Exception:
            return None

    def _save_entry(self, entry: MemoryEntry):
        """保存单个条目"""
        if self.use_markdown:
            self._save_entry_to_markdown(entry)
        else:
            self._save_entry_to_jsonl(entry)

    def _save_entry_to_jsonl(self, entry: MemoryEntry):
        """保存到 JSONL"""
        file_path = self.storage_dir / f"{entry.type.value}.jsonl"
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')

    def _save_entry_to_markdown(self, entry: MemoryEntry):
        """保存到 Markdown"""
        entries = self._parse_markdown_file()

        existing_idx = None
        for i, e in enumerate(entries):
            if e.id == entry.id:
                existing_idx = i
                break

        if existing_idx is not None:
            entries[existing_idx] = entry
        else:
            entries.append(entry)

        self._save_all_entries_to_markdown(entries)

    def _save_all_entries_to_markdown(self, entries: List[MemoryEntry]):
        """保存所有条目到 Markdown"""
        grouped = {
            MemoryType.SEMANTIC: [],
            MemoryType.PROCEDURAL: [],
            MemoryType.EPISODIC: [],
        }

        for entry in entries:
            grouped[entry.type].append(entry)

        sections = [
            "# System Memory - 系统记忆",
            "",
            "> 此文件由 OllamaPilot 自动生成和维护",
            "> 您可以手动编辑此文件来添加、修改或删除记忆",
            "> 格式：YAML frontmatter + Markdown 内容",
            "",
            "---",
            "",
        ]

        sections.extend([
            "## 语义记忆 (Semantic)",
            "",
            "用户偏好、重要事实",
            "",
        ])
        for entry in grouped[MemoryType.SEMANTIC]:
            sections.append(self._entry_to_markdown(entry))

        sections.extend([
            "## 程序记忆 (Procedural)",
            "",
            "Skill 使用模式、操作习惯",
            "",
        ])
        for entry in grouped[MemoryType.PROCEDURAL]:
            sections.append(self._entry_to_markdown(entry))

        sections.extend([
            "## 情景记忆 (Episodic)",
            "",
            "重要对话摘要、历史事件",
            "",
        ])
        for entry in grouped[MemoryType.EPISODIC]:
            sections.append(self._entry_to_markdown(entry))

        content = "\n".join(sections)
        self.md_file.write_text(content, encoding='utf-8')
        self._rebuild_cache()

    def _entry_to_markdown(self, entry: MemoryEntry) -> str:
        """将记忆条目转换为 Markdown"""
        frontmatter_data = {
            "id": entry.id,
            "type": entry.type.value,
            "timestamp": entry.timestamp.isoformat(),
            "importance": entry.importance,
        }

        for key, value in entry.metadata.items():
            if key not in frontmatter_data:
                frontmatter_data[key] = value

        lines = ["---"]
        for key, value in frontmatter_data.items():
            if isinstance(value, str):
                if any(c in value for c in [':', '#', '\n', '"', "'"]):
                    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                    value = f'"{escaped}"'
                lines.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{key}: {str(value).lower()}")
        lines.append("---")
        frontmatter = "\n".join(lines)

        return f"{frontmatter}\n\n{entry.content}\n\n---\n\n"

    def _rebuild_cache(self):
        """从 Markdown 重建 JSON 缓存"""
        entries = self._parse_markdown_file()
        cache_data = {
            "version": "2.0",
            "last_modified": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in entries]
        }

        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def _generate_id(self, content: str) -> str:
        """生成记忆 ID"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _find_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """根据 ID 查找条目"""
        for entries in self._cache.values():
            for entry in entries:
                if entry.id == entry_id:
                    return entry
        return None

    def _find_entry_by_content(self, content: str) -> Optional[MemoryEntry]:
        """根据内容查找记忆条目"""
        for mem_type in MemoryType:
            for entry in self._cache.get(mem_type, []):
                if entry.content == content:
                    return entry
        return None

    # ========== 语义记忆接口 ==========

    def remember_fact(self, fact: str, category: str = "general", importance: float = 1.0):
        """记住重要事实"""
        entry_id = self._generate_id(f"{category}:{fact}")

        existing = self._find_by_id(entry_id)
        if existing:
            existing.importance = min(1.0, existing.importance + 0.1)
            if self.use_markdown:
                self._save_entry(existing)
            return

        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.SEMANTIC,
            content=fact,
            metadata={"category": category},
            importance=importance,
        )

        self._save_entry(entry)
        if MemoryType.SEMANTIC not in self._cache:
            self._cache[MemoryType.SEMANTIC] = []
        self._cache[MemoryType.SEMANTIC].append(entry)

    def recall_facts(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[str]:
        """检索相关事实"""
        entries = self._cache.get(MemoryType.SEMANTIC, [])

        if category:
            entries = [e for e in entries if e.metadata.get("category") == category]

        query_words = set(query.lower().split())
        scored_entries = []

        for entry in entries:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap * entry.importance
            scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [e.content for _, e in scored_entries[:top_k]]

    # ========== 程序记忆接口 ==========

    def record_skill_usage(self, skill_name: str, context: str = ""):
        """记录 Skill 使用"""
        entry_id = f"skill_usage:{skill_name}"

        existing = self._find_by_id(entry_id)
        if existing:
            count = existing.metadata.get("count", 0) + 1
            existing.metadata["count"] = count
            existing.metadata["last_used"] = datetime.now().isoformat()
            if self.use_markdown:
                self._save_entry(existing)
        else:
            entry = MemoryEntry(
                id=entry_id,
                type=MemoryType.PROCEDURAL,
                content=f"Skill '{skill_name}' 使用记录",
                metadata={
                    "skill_name": skill_name,
                    "count": 1,
                    "first_used": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                    "context": context,
                },
            )
            self._save_entry(entry)
            if MemoryType.PROCEDURAL not in self._cache:
                self._cache[MemoryType.PROCEDURAL] = []
            self._cache[MemoryType.PROCEDURAL].append(entry)

    def get_skill_preferences(self) -> Dict[str, int]:
        """获取 Skill 使用偏好"""
        entries = self._cache.get(MemoryType.PROCEDURAL, [])
        preferences = {}

        for entry in entries:
            if entry.metadata.get("skill_name"):
                skill_name = entry.metadata["skill_name"]
                count = entry.metadata.get("count", 0)
                preferences[skill_name] = count

        return preferences

    def get_favorite_skills(self, top_k: int = 3) -> List[str]:
        """获取最常用的 Skills"""
        preferences = self.get_skill_preferences()
        sorted_skills = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_skills[:top_k]]

    # ========== 情景记忆接口 ==========

    def summarize_conversation(self, thread_id: str, summary: str, topic: str = ""):
        """存储对话摘要"""
        entry_id = f"conversation:{thread_id}"

        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.EPISODIC,
            content=summary,
            metadata={
                "thread_id": thread_id,
                "topic": topic,
                "date": datetime.now().isoformat(),
            },
        )

        self._save_entry(entry)
        if MemoryType.EPISODIC not in self._cache:
            self._cache[MemoryType.EPISODIC] = []
        self._cache[MemoryType.EPISODIC].append(entry)

    def recall_conversations(self, topic: str = "", top_k: int = 3) -> List[str]:
        """检索相关对话"""
        entries = self._cache.get(MemoryType.EPISODIC, [])

        if not topic:
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return [e.content for e in entries[:top_k]]

        topic_words = set(topic.lower().split())
        scored_entries = []

        for entry in entries:
            content_words = set(entry.content.lower().split())
            topic_meta = set(entry.metadata.get("topic", "").lower().split())
            overlap = len(topic_words & (content_words | topic_meta))
            scored_entries.append((overlap, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [e.content for _, e in scored_entries[:top_k]]

    # ========== 增强版检索接口 ==========

    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """
        统一检索接口
        
        如果启用了向量搜索，使用增强版混合检索；
        否则使用基础关键词检索。
        """
        if not self.enable_vector_search:
            # 基础检索
            results = []
            facts = self.recall_facts(query, top_k=top_k)
            results.extend(facts)
            conversations = self.recall_conversations(topic=query, top_k=top_k)
            results.extend(conversations)
            return results[:top_k]

        # 增强版混合检索
        results = self._recall_enhanced(query, top_k=top_k, use_hybrid=True)
        
        # 记录访问
        for content, _ in results:
            entry = self._find_entry_by_content(content)
            if entry:
                self._record_access(entry.id)

        return [content for content, _ in results]

    def _recall_enhanced(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        min_score: float = 0.3
    ) -> List[Tuple[str, float]]:
        """增强版检索接口（内部方法）"""
        if not use_hybrid or not self.enable_vector_search:
            # 回退到基础检索
            results = []
            facts = self.recall_facts(query, top_k=top_k)
            results.extend([(f, 1.0) for f in facts])
            conversations = self.recall_conversations(topic=query, top_k=top_k)
            results.extend([(c, 1.0) for c in conversations])
            return results[:top_k]

        # 1. 关键词检索
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        # 2. 向量语义检索
        vector_results = self._vector_search(query, top_k=top_k * 2)

        # 3. 结果融合（加权）
        combined_scores: Dict[str, float] = {}

        for i, (content, entry) in enumerate(keyword_results):
            rank_score = 1.0 / (i + 1)
            combined_scores[content] = combined_scores.get(content, 0) + rank_score * self.keyword_weight

        for content, score in vector_results:
            combined_scores[content] = combined_scores.get(content, 0) + score * self.vector_weight

        # 4. 多维度评分增强
        final_results = []
        for content, base_score in combined_scores.items():
            entry = self._find_entry_by_content(content)
            if entry:
                final_score = self._calculate_multi_dim_score(base_score, entry, query)
                if final_score >= min_score:
                    final_results.append((content, final_score))

        # 5. 排序并返回 Top-k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, MemoryEntry]]:
        """关键词检索"""
        query_words = set(query.lower().split())
        results = []

        for mem_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC]:
            for entry in self._cache.get(mem_type, []):
                content_words = set(entry.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap * entry.importance
                    results.append((score, entry.content, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [(content, entry) for _, content, entry in results[:top_k]]

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """向量语义检索"""
        try:
            vector_store = self._build_vector_store()
            if vector_store is None:
                return []

            if hasattr(vector_store, 'similarity_search_with_score'):
                results = vector_store.similarity_search_with_score(query, k=top_k)
                return [(doc.page_content, score) for doc, score in results]
            else:
                return []

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 向量检索失败: {e}")
            return []

    def _calculate_multi_dim_score(
        self,
        base_score: float,
        entry: MemoryEntry,
        query: str
    ) -> float:
        """多维度价值评分"""
        # 1. 相关性 (40%)
        relevance_score = base_score

        # 2. 重要性 (30%)
        importance_score = entry.importance

        # 3. 时效性 (20%)
        days_old = (datetime.now() - entry.timestamp).days
        recency_score = math.exp(-days_old / 30)

        # 4. 访问频率 (10%)
        access_count = self._memory_access_count.get(entry.id, 0)
        frequency_score = min(math.log(access_count + 1) / 5, 1.0)

        # 综合评分
        final_score = (
            relevance_score * 0.4 +
            importance_score * 0.3 +
            recency_score * 0.2 +
            frequency_score * 0.1
        )

        return final_score

    def _record_access(self, entry_id: str):
        """记录记忆访问（用于频率统计）"""
        self._memory_access_count[entry_id] = self._memory_access_count.get(entry_id, 0) + 1

    # ========== 统计接口 ==========

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "semantic": len(self._cache.get(MemoryType.SEMANTIC, [])),
            "procedural": len(self._cache.get(MemoryType.PROCEDURAL, [])),
            "episodic": len(self._cache.get(MemoryType.EPISODIC, [])),
            "total": sum(len(entries) for entries in self._cache.values()),
        }
