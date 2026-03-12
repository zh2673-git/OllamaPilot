"""
GraphRAG 中间件

在模型调用前注入实体-关系增强的上下文
"""

from typing import Dict, Any, List, Optional
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, HumanMessage

from skills.graphrag.services import GraphRAGService, HybridEntityExtractor


class GraphRAGMiddleware(AgentMiddleware):
    """
    GraphRAG 检索中间件

    功能：
    1. 在 before_model 阶段从查询中提取实体
    2. 检索相关文档（实体索引 + 向量检索）
    3. 遍历关系找到相关实体（1-2跳）
    4. 将增强的上下文注入到系统提示词
    """

    def __init__(
        self,
        graph_service: GraphRAGService,
        entity_extractor: HybridEntityExtractor,
        n_results: int = 5,
        max_hops: int = 2,
        min_relevance_score: float = 0.3,
        verbose: bool = True
    ):
        """
        初始化中间件

        Args:
            graph_service: 图谱服务实例
            entity_extractor: 实体抽取器
            n_results: 返回结果数量
            max_hops: 最大关系遍历跳数
            min_relevance_score: 最小相关性阈值
            verbose: 是否显示详细日志
        """
        super().__init__()
        self.graph_service = graph_service
        self.entity_extractor = entity_extractor
        self.n_results = n_results
        self.max_hops = max_hops
        self.min_relevance_score = min_relevance_score
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "GraphRAGMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """
        在模型调用前执行检索
        
        注意：默认不自动检索，只在用户明确要求时使用知识库

        Args:
            state: 当前状态，包含 messages 等
            runtime: 运行时上下文

        Returns:
            修改后的 state
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # 获取最后一条用户消息
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return None

        query = str(last_message.content)

        # 检查是否明确要求使用知识库
        # 只有在查询中包含特定关键词时才启用知识库检索
        kg_keywords = ['根据知识库', '查一下知识库', '知识库中', '文档中', '伤寒论', '金匮要略', '搜索文档']
        use_knowledge_base = any(kw in query for kw in kg_keywords)
        
        # 检查是否指定了分类（如"伤寒论知识库"、"XXX分类"）
        # 如果指定了分类，让 LLM 调用 search_in_category 工具，中间件不拦截
        category_keywords = ['知识库', '分类']
        has_category = any(kw in query for kw in category_keywords)

        if not use_knowledge_base:
            # 默认不使用知识库，让模型自行处理
            return None

        if has_category:
            # 用户指定了分类，让 LLM 调用 search_in_category 工具
            if self.verbose:
                print(f"📚 检测到分类查询请求，交由 Skill 工具处理...")
            return None

        if self.verbose:
            print(f"📚 检测到知识库查询请求，开始检索...")

        # 步骤 1: 从查询中提取实体
        query_entities = self.entity_extractor.extract_from_query(query)

        if not query_entities:
            if self.verbose:
                print("📝 未提取到实体，尝试向量检索")
            # 尝试向量检索作为回退
            return self._fallback_vector_search(state, query)

        if self.verbose:
            entity_names = [e["name"] for e in query_entities]
            print(f"🔍 提取到实体: {', '.join(entity_names)}")

        # 步骤 2-4: 实体-关系增强检索
        retrieval_result = self.graph_service.enhanced_search(
            query=query,
            query_entities=query_entities,
            n_results=self.n_results,
            max_hops=self.max_hops
        )

        if not retrieval_result["documents"]:
            if self.verbose:
                print("⚠️ 未找到相关文档")
            return None

        if self.verbose:
            print(f"📚 找到 {len(retrieval_result['documents'])} 个相关文档")
            if retrieval_result["relations"]:
                print(f"🔗 发现 {len(retrieval_result['relations'])} 个关系")

        # 步骤 5: 构建增强上下文
        enhanced_context = self._build_context(retrieval_result)

        # 步骤 6: 注入到系统提示词
        kg_system_prompt = SystemMessage(content=f"""\
基于知识图谱的检索结果：

{enhanced_context}

请基于以上信息回答用户问题。如果信息不足，请明确告知。""")

        # 插入到消息列表
        new_messages = self._insert_context(messages, kg_system_prompt)

        # 返回更新后的状态
        return {
            "messages": new_messages,
            "graphrag_context": retrieval_result
        }

    def _build_context(self, retrieval_result: Dict[str, Any]) -> str:
        """构建增强上下文"""
        context_parts = []

        # 1. 查询实体
        if retrieval_result.get("query_entities"):
            context_parts.append("【查询实体】")
            for entity in retrieval_result["query_entities"]:
                context_parts.append(f"- {entity['name']} ({entity['type']})")

        # 2. 发现的关系
        if retrieval_result.get("relations"):
            context_parts.append("\n【相关关系】")
            for rel in retrieval_result["relations"][:5]:  # 最多显示5个
                context_parts.append(
                    f"- {rel['source']} --{rel['relation']}--> {rel['target']}"
                )

        # 3. 相关实体
        if retrieval_result.get("related_entities"):
            context_parts.append("\n【相关实体】")
            context_parts.append(
                ", ".join(retrieval_result["related_entities"][:10])
            )

        # 4. 文档内容
        if retrieval_result.get("documents"):
            context_parts.append("\n【参考文档】")
            for i, doc in enumerate(retrieval_result["documents"], 1):
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "未知")
                score = doc.get("score", 0)
                content = doc.get("content", "")[:500]

                context_parts.append(f"\n[{i}] 来源: {source}")
                context_parts.append(f"相关度: {score:.2f}")
                context_parts.append(f"内容: {content}...")

        return "\n".join(context_parts)

    def _insert_context(
        self,
        messages: List[Any],
        context_msg: SystemMessage
    ) -> List[Any]:
        """将上下文插入到消息列表"""
        new_messages = []
        has_system = False
        inserted = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                has_system = True
                new_messages.append(msg)
            elif has_system and isinstance(msg, HumanMessage) and not inserted:
                new_messages.append(context_msg)
                new_messages.append(msg)
                inserted = True
                has_system = False
            else:
                new_messages.append(msg)

        if not has_system and not inserted:
            new_messages.insert(0, context_msg)

        return new_messages

    def _fallback_vector_search(
        self,
        state: Dict[str, Any],
        query: str
    ) -> Optional[Dict[str, Any]]:
        """回退到纯向量检索"""
        results = self.graph_service.vector_search(query, n_results=self.n_results)

        if not results:
            return None

        # 构建简化上下文
        context_parts = ["【相关文档】"]
        for i, doc in enumerate(results, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "未知")
            score = doc.get("score", 0)
            content = doc.get("content", "")[:500]

            context_parts.append(f"\n[{i}] 来源: {source}")
            context_parts.append(f"相关度: {score:.2f}")
            context_parts.append(f"内容: {content}...")

        enhanced_context = "\n".join(context_parts)

        kg_system_prompt = SystemMessage(content=f"""\
基于向量检索的相关文档：

{enhanced_context}

请基于以上信息回答用户问题。如果信息不足，请明确告知。""")

        messages = state.get("messages", [])
        new_messages = self._insert_context(messages, kg_system_prompt)

        return {
            "messages": new_messages,
            "graphrag_context": {"documents": results}
        }
