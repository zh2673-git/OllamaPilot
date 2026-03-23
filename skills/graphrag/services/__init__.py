"""
GraphRAG 服务层

提供知识图谱构建和检索服务
"""

from skills.graphrag.services.graphrag_service import GraphRAGService, Entity, Relation
from skills.graphrag.services.entity_extractor import (
    HybridEntityExtractor,
    ExtractedEntity,
    ExtractedRelation
)
from skills.graphrag.services.triple_vector_store import (
    TripleVectorStore,
    EntityInfo,
    RelationInfo
)
from skills.graphrag.services.dual_retrieval import (
    DualRetrievalService,
    RetrievalResult,
    EntityContext,
    RelationContext,
    RetrievalMode,
    ContextFusionConfig,
    ContextFusionService
)
from skills.graphrag.services.gleaning_extractor import (
    GleaningExtractor,
    GleaningConfig,
    simple_merge_descriptions
)
from skills.graphrag.services.incremental_merger import (
    IncrementalMerger,
    MergeConfig
)

__all__ = [
    # 核心服务
    "GraphRAGService",
    "Entity",
    "Relation",

    # 实体抽取
    "HybridEntityExtractor",
    "ExtractedEntity",
    "ExtractedRelation",

    # 三重向量存储 (LightRAG 增强)
    "TripleVectorStore",
    "EntityInfo",
    "RelationInfo",

    # 双层检索 (LightRAG 增强)
    "DualRetrievalService",
    "RetrievalResult",
    "EntityContext",
    "RelationContext",
    "RetrievalMode",
    "ContextFusionConfig",
    "ContextFusionService",

    # Gleaning 抽取 (LightRAG 增强)
    "GleaningExtractor",
    "GleaningConfig",
    "simple_merge_descriptions",

    # 增量合并 (LightRAG 增强)
    "IncrementalMerger",
    "MergeConfig",
]
