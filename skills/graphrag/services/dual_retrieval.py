"""
双层检索服务 (Dual Retrieval Service)

基于 LightRAG 的双层检索思想，实现：
- 低层检索 (Local)：基于实体的精确匹配检索
- 高层检索 (Global)：基于关系的语义发现检索
- 混合检索 (Mix)：融合 Local 和 Global 结果

适合小模型场景：零额外 LLM 调用，纯向量检索增强。
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from skills.graphrag.services.triple_vector_store import TripleVectorStore


class RetrievalMode(Enum):
    """检索模式"""
    LOCAL = "local"      # 仅低层检索（实体）
    GLOBAL = "global"    # 仅高层检索（关系）
    MIX = "mix"          # 混合检索


@dataclass
class EntityContext:
    """实体上下文"""
    name: str
    entity_type: str
    description: str
    source_ids: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class RelationContext:
    """关系上下文"""
    source: str
    target: str
    relation: str
    description: str
    confidence: float = 0.5
    source_ids: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class RetrievalResult:
    """检索结果"""
    entities: List[EntityContext] = field(default_factory=list)
    relations: List[RelationContext] = field(default_factory=list)
    chunks: List[Dict] = field(default_factory=list)

    def to_context_string(self, max_entity_tokens: int = 2000,
                         max_relation_tokens: int = 3000) -> str:
        """
        转换为上下文字符串

        Args:
            max_entity_tokens: 实体部分最大 token 数（估算）
            max_relation_tokens: 关系部分最大 token 数（估算）

        Returns:
            格式化的上下文字符串
        """
        parts = []

        # 实体部分
        if self.entities:
            entity_texts = []
            current_tokens = 0

            for entity in self.entities:
                text = f"- {entity.name} ({entity.entity_type}): {entity.description}"
                # 简单估算：中文字符 + 英文单词
                tokens = len(text) // 2  # 粗略估算

                if current_tokens + tokens > max_entity_tokens:
                    break

                entity_texts.append(text)
                current_tokens += tokens

            if entity_texts:
                parts.append("## 实体信息\n" + "\n".join(entity_texts))

        # 关系部分
        if self.relations:
            relation_texts = []
            current_tokens = 0

            for rel in self.relations:
                text = f"- {rel.source} {rel.relation} {rel.target}"
                if rel.description and rel.description != f"{rel.source} {rel.relation} {rel.target}":
                    text += f": {rel.description}"

                tokens = len(text) // 2

                if current_tokens + tokens > max_relation_tokens:
                    break

                relation_texts.append(text)
                current_tokens += tokens

            if relation_texts:
                parts.append("## 关系信息\n" + "\n".join(relation_texts))

        return "\n\n".join(parts)


class DualRetrievalService:
    """
    双层检索服务

    借鉴 LightRAG 的核心创新：
    1. Local 检索：基于实体的精确匹配，适合"XX是什么"类问题
    2. Global 检索：基于关系的语义发现，适合"有哪些趋势"类问题
    3. Mix 模式：融合两者，提供更全面的上下文
    """

    def __init__(
        self,
        triple_store: TripleVectorStore,
        embedding_fn: Any,
        config: Optional[Dict] = None
    ):
        """
        初始化双层检索服务

        Args:
            triple_store: 三重向量存储
            embedding_fn: Embedding 函数
            config: 配置字典
        """
        self.triple_store = triple_store
        self.embedding_fn = embedding_fn
        self.config = config or {}

        # 检索配置
        self.local_top_k = self.config.get("local_top_k", 30)
        self.global_top_k = self.config.get("global_top_k", 30)
        self.chunk_top_k = self.config.get("chunk_top_k", 10)

    def local_retrieval(self, query: str, top_k: Optional[int] = None) -> List[EntityContext]:
        """
        低层检索 (Local Retrieval)

        基于实体的向量匹配，适合回答"XX是什么"类问题。
        直接匹配查询中的实体，获取相关实体信息。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            实体上下文列表
        """
        top_k = top_k or self.local_top_k

        try:
            # 1. 查询向量化
            query_embedding = self.embedding_fn([query])[0]

            # 2. 实体向量检索
            entity_results = self.triple_store.search_entities(
                query_embedding=query_embedding,
                top_k=top_k
            )

            # 3. 构建实体上下文
            contexts = []
            for result in entity_results:
                metadata = result.get("metadata", {})
                context = EntityContext(
                    name=metadata.get("name", ""),
                    entity_type=metadata.get("type", "未知"),
                    description=metadata.get("description", ""),
                    source_ids=metadata.get("source_ids", []),
                    score=result.get("score", 0.0)
                )
                contexts.append(context)

            return contexts

        except Exception as e:
            print(f"⚠️ Local 检索失败: {e}")
            return []

    def global_retrieval(self, query: str, top_k: Optional[int] = None) -> List[RelationContext]:
        """
        高层检索 (Global Retrieval)

        基于关系的向量语义发现，适合回答"有哪些趋势"、"主题是什么"类问题。
        通过关系语义发现与查询相关的主题和模式。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            关系上下文列表
        """
        top_k = top_k or self.global_top_k

        try:
            # 1. 查询向量化
            query_embedding = self.embedding_fn([query])[0]

            # 2. 关系向量检索（核心）
            relation_results = self.triple_store.search_relations(
                query_embedding=query_embedding,
                top_k=top_k
            )

            # 3. 构建关系上下文
            contexts = []
            for result in relation_results:
                metadata = result.get("metadata", {})
                context = RelationContext(
                    source=metadata.get("source", ""),
                    target=metadata.get("target", ""),
                    relation=metadata.get("relation", ""),
                    description=metadata.get("description", ""),
                    confidence=metadata.get("confidence", 0.5),
                    source_ids=metadata.get("source_ids", []),
                    score=result.get("score", 0.0)
                )
                contexts.append(context)

            return contexts

        except Exception as e:
            print(f"⚠️ Global 检索失败: {e}")
            return []

    def hybrid_retrieval(
        self,
        query: str,
        mode: str = "mix",
        local_top_k: Optional[int] = None,
        global_top_k: Optional[int] = None,
        include_chunks: bool = False
    ) -> RetrievalResult:
        """
        混合检索

        根据模式选择检索策略：
        - local: 仅低层检索
        - global: 仅高层检索
        - mix: 融合两者

        Args:
            query: 查询文本
            mode: 检索模式 ("local" | "global" | "mix")
            local_top_k: Local 检索返回数量
            global_top_k: Global 检索返回数量
            include_chunks: 是否包含文本块检索

        Returns:
            检索结果
        """
        local_top_k = local_top_k or self.local_top_k
        global_top_k = global_top_k or self.global_top_k

        result = RetrievalResult()

        if mode == RetrievalMode.LOCAL.value:
            # 仅 Local 检索
            result.entities = self.local_retrieval(query, local_top_k)

        elif mode == RetrievalMode.GLOBAL.value:
            # 仅 Global 检索
            result.relations = self.global_retrieval(query, global_top_k)

        else:  # mix 模式
            # 同时进行 Local 和 Global 检索
            result.entities = self.local_retrieval(query, local_top_k)
            result.relations = self.global_retrieval(query, global_top_k)

        # 可选：文本块检索
        if include_chunks:
            try:
                query_embedding = self.embedding_fn([query])[0]
                result.chunks = self.triple_store.search_chunks(
                    query_embedding=query_embedding,
                    top_k=self.chunk_top_k
                )
            except Exception as e:
                print(f"⚠️ 文本块检索失败: {e}")

        return result

    def get_related_entities_from_relations(
        self,
        relations: List[RelationContext],
        query_entities: List[str]
    ) -> List[str]:
        """
        从关系中获取相关实体

        用于扩展查询实体集合，发现与查询相关的其他实体。

        Args:
            relations: 关系列表
            query_entities: 查询中的实体

        Returns:
            相关实体名称列表
        """
        related = set()
        query_entity_set = set(name.lower() for name in query_entities)

        for rel in relations:
            # 如果关系的 source 在查询实体中，添加 target
            if rel.source.lower() in query_entity_set:
                related.add(rel.target)
            # 如果关系的 target 在查询实体中，添加 source
            if rel.target.lower() in query_entity_set:
                related.add(rel.source)

        # 排除查询实体本身
        return list(related - query_entity_set)

    def deduplicate_results(self, result: RetrievalResult) -> RetrievalResult:
        """
        去重检索结果

        Args:
            result: 原始检索结果

        Returns:
            去重后的结果
        """
        # 实体去重
        seen_entities = set()
        unique_entities = []
        for entity in result.entities:
            key = f"{entity.name}_{entity.entity_type}"
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity)

        # 关系去重
        seen_relations = set()
        unique_relations = []
        for rel in result.relations:
            key = f"{rel.source}_{rel.relation}_{rel.target}"
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(rel)

        return RetrievalResult(
            entities=unique_entities,
            relations=unique_relations,
            chunks=result.chunks
        )


class ContextFusionConfig:
    """上下文融合配置"""

    def __init__(
        self,
        max_entity_tokens: int = 2000,
        max_relation_tokens: int = 3000,
        max_text_tokens: int = 8000,
        priority: str = "balanced"  # "entity" | "relation" | "balanced"
    ):
        self.max_entity_tokens = max_entity_tokens
        self.max_relation_tokens = max_relation_tokens
        self.max_text_tokens = max_text_tokens
        self.priority = priority

    @classmethod
    def from_model_context(cls, context_length: int) -> "ContextFusionConfig":
        """
        根据模型上下文长度自动配置

        Args:
            context_length: 模型上下文长度

        Returns:
            配置对象
        """
        if context_length <= 8192:  # 8K 模型
            return cls(
                max_entity_tokens=1000,
                max_relation_tokens=1500,
                max_text_tokens=4000
            )
        elif context_length <= 32768:  # 32K 模型
            return cls(
                max_entity_tokens=2000,
                max_relation_tokens=3000,
                max_text_tokens=8000
            )
        else:  # 128K+ 模型
            return cls(
                max_entity_tokens=4000,
                max_relation_tokens=6000,
                max_text_tokens=16000
            )


class ContextFusionService:
    """上下文融合服务"""

    def __init__(self, config: Optional[ContextFusionConfig] = None):
        self.config = config or ContextFusionConfig()

    def fuse(self, retrieval_result: RetrievalResult) -> str:
        """
        融合检索结果

        Args:
            retrieval_result: 检索结果

        Returns:
            融合后的上下文字符串
        """
        return retrieval_result.to_context_string(
            max_entity_tokens=self.config.max_entity_tokens,
            max_relation_tokens=self.config.max_relation_tokens
        )
