"""
Gleaning 循环实体抽取器

基于 LightRAG 的 Gleaning 机制，通过多轮循环提取提升实体-关系抽取完整度：
- 第一轮：基础 LLM 抽取
- 第二轮：基于首轮结果的补充提取（Gleaning）

适合小模型场景：可配置 Gleaning 轮数，默认关闭以节省 LLM 调用。
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from skills.graphrag.services.entity_extractor import (
    HybridEntityExtractor,
    ExtractedEntity,
    ExtractedRelation
)


@dataclass
class GleaningConfig:
    """Gleaning 配置"""
    rounds: int = 0  # Gleaning 轮数（默认0，即不启用）
    max_entities_per_round: int = 20
    confidence_threshold: float = 0.6


class GleaningExtractor:
    """
    Gleaning 循环实体抽取器

    借鉴 LightRAG 的 Gleaning 机制：
    1. 第一轮：基础提取（词典 + LLM）
    2. 第二轮+：基于首轮结果的补充提取（Gleaning）

    小模型场景建议：
    - rounds=0：关闭 Gleaning，节省 LLM 调用
    - rounds=1：基础提取（默认）
    - rounds=2：一轮 Gleaning，提升完整度
    """

    def __init__(
        self,
        base_extractor: HybridEntityExtractor,
        config: Optional[GleaningConfig] = None
    ):
        """
        初始化 Gleaning 抽取器

        Args:
            base_extractor: 基础抽取器（词典+LLM混合）
            config: Gleaning 配置
        """
        self.base_extractor = base_extractor
        self.config = config or GleaningConfig()

    def extract(
        self,
        text: str,
        llm_client: Any = None,
        use_dictionary: bool = True
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        带 Gleaning 的实体抽取

        Args:
            text: 输入文本
            llm_client: LLM 客户端
            use_dictionary: 是否使用词典

        Returns:
            (实体列表, 关系列表)
        """
        # 第一轮：基础提取
        entities, relations = self.base_extractor.extract(
            text=text,
            use_llm=True,
            llm_client=llm_client,
            top_k=self.config.max_entities_per_round
        )

        # 如果没有启用 Gleaning 或没有 LLM 客户端，直接返回
        if self.config.rounds <= 1 or not llm_client:
            return entities, relations

        # Gleaning 循环（第二轮及以后）
        for round_num in range(1, self.config.rounds):
            # 构建上下文：已提取的实体
            context = self._build_gleaning_context(entities, relations)

            # 补充提取
            supplement_entities, supplement_relations = self._gleaning_extract(
                text, context, llm_client
            )

            # 合并结果（去重）
            entities = self._merge_extractions(entities, supplement_entities)
            relations = self._merge_relations(relations, supplement_relations)

        return entities, relations

    def extract_batch(
        self,
        chunks: List[str],
        llm_client: Any = None,
        progress_callback=None
    ) -> List[Tuple[List[ExtractedEntity], List[ExtractedRelation]]]:
        """
        批量 Gleaning 抽取

        Args:
            chunks: 文本块列表
            llm_client: LLM 客户端
            progress_callback: 进度回调

        Returns:
            每块的 (实体列表, 关系列表)
        """
        results = []

        for i, chunk in enumerate(chunks):
            entities, relations = self.extract(
                text=chunk,
                llm_client=llm_client
            )
            results.append((entities, relations))

            if progress_callback:
                progress_callback(i + 1, len(chunks), len(entities))

        return results

    def _build_gleaning_context(
        self,
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation]
    ) -> str:
        """
        构建 Gleaning 上下文

        Args:
            entities: 已提取的实体
            relations: 已提取的关系

        Returns:
            上下文字符串
        """
        context_parts = []

        if entities:
            context_parts.append("已提取的实体：")
            for e in entities[:15]:  # 限制数量，避免提示词过长
                context_parts.append(f"- {e.name} ({e.type})")

        if relations:
            context_parts.append("\n已提取的关系：")
            for r in relations[:10]:  # 限制数量
                context_parts.append(f"- {r.source} {r.relation} {r.target}")

        context_parts.append("\n请补充遗漏的实体和关系，不要重复已列出的内容。")

        return "\n".join(context_parts)

    def _gleaning_extract(
        self,
        text: str,
        context: str,
        llm_client: Any
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        基于上下文的补充提取

        Args:
            text: 原始文本
            context: 已提取内容的上下文
            llm_client: LLM 客户端

        Returns:
            (补充实体列表, 补充关系列表)
        """
        try:
            # 构造补充提取提示词
            prompt = f"""基于以下文本和已提取的内容，请补充遗漏的实体和关系。

文本内容：
{text[:2000]}  # 限制长度，避免提示词过长

{context}

请只输出新发现的实体（不要重复已列出的实体），格式与之前相同。

按JSON格式输出：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "confidence": 0.8}}
  ],
  "relations": [
    {{"source": "实体1", "target": "实体2", "relation": "关系类型", "confidence": 0.7}}
  ]
}}

如果没有新发现，请返回空列表。"""

            # 调用 LLM 补充提取
            response = llm_client.generate(prompt, timeout=60, silent=True)

            if not response:
                return [], []

            # 解析补充结果
            return self._parse_gleaning_response(response)

        except Exception as e:
            # 静默处理错误，不影响主流程
            return [], []

    def _parse_gleaning_response(
        self,
        response: str
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        解析 Gleaning 响应

        Args:
            response: LLM 响应

        Returns:
            (实体列表, 关系列表)
        """
        import json
        import re

        entities = []
        relations = []

        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # 解析实体
                for entity_data in data.get("entities", []):
                    name = entity_data.get("name", "")
                    if name:
                        entities.append(ExtractedEntity(
                            name=name,
                            type=entity_data.get("type", "未知"),
                            start=0,  # Gleaning 阶段不记录位置
                            end=0,
                            confidence=entity_data.get("confidence", 0.6),
                            source="gleaning"
                        ))

                # 解析关系
                for relation_data in data.get("relations", []):
                    relations.append(ExtractedRelation(
                        source=relation_data.get("source", ""),
                        target=relation_data.get("target", ""),
                        relation=relation_data.get("relation", "RELATED"),
                        confidence=relation_data.get("confidence", 0.6)
                    ))

        except Exception:
            pass

        return entities, relations

    def _merge_extractions(
        self,
        existing: List[ExtractedEntity],
        supplement: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        合并实体提取结果（去重）

        Args:
            existing: 已有实体
            supplement: 补充实体

        Returns:
            合并后的实体列表
        """
        existing_names = {e.name.lower() for e in existing}

        for entity in supplement:
            if entity.name.lower() not in existing_names:
                existing.append(entity)
                existing_names.add(entity.name.lower())

        return existing

    def _merge_relations(
        self,
        existing: List[ExtractedRelation],
        supplement: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """
        合并关系提取结果（去重）

        Args:
            existing: 已有关系
            supplement: 补充关系

        Returns:
            合并后的关系列表
        """
        existing_keys = {
            f"{r.source}_{r.relation}_{r.target}".lower()
            for r in existing
        }

        for relation in supplement:
            key = f"{relation.source}_{relation.relation}_{relation.target}".lower()
            if key not in existing_keys:
                existing.append(relation)
                existing_keys.add(key)

        return existing


def simple_merge_descriptions(desc1: str, desc2: str) -> str:
    """
    简单合并描述（零 LLM 成本）

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
