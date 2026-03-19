"""
GraphRAG 工具集

提供知识图谱相关的工具函数。
"""

import logging
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pathlib import Path

from skills.graphrag.services import (
    GraphRAGService,
    HybridEntityExtractor,
    Entity
)
from skills.graphrag.llm_client import SimpleLLMClient
from skills.graphrag.utils import DocumentProcessor

logger = logging.getLogger("ollamapilot.skills.graphrag.tools")


# 全局服务实例（由 Agent 初始化时注入）
_graph_service: Optional[GraphRAGService] = None
_entity_extractor: Optional[HybridEntityExtractor] = None
_llm_client: Optional[SimpleLLMClient] = None
_use_llm: bool = False
_knowledge_base_dir: str = "./data/knowledge_base"
_temp_documents_dir: str = "./data/temp_documents"
_document_manager: Optional[Any] = None
_model_name: Optional[str] = None


def init_graphrag_services(
    service: GraphRAGService,
    ext: Optional[HybridEntityExtractor] = None,
    kb_dir: str = "./data/knowledge_base",
    temp_dir: str = "./data/temp_documents",
    doc_manager: Optional[Any] = None,
    model_name: Optional[str] = None
):
    """初始化服务"""
    global _graph_service, _entity_extractor, _llm_client, _use_llm, _knowledge_base_dir, _temp_documents_dir, _document_manager, _model_name
    _graph_service = service
    _entity_extractor = ext or HybridEntityExtractor(persist_dir=service.persist_dir)
    _llm_client = SimpleLLMClient()
    _use_llm = _llm_client.is_available()
    _knowledge_base_dir = kb_dir
    _temp_documents_dir = temp_dir
    _document_manager = doc_manager
    _model_name = model_name


def _normalize_filename(filename: str) -> str:
    """
    标准化文件名用于模糊匹配
    移除常见标点符号、空格，转为小写
    """
    import re
    # 移除引号、书名号、空格等常见差异字符
    normalized = re.sub(r'[""''《》【】()（）\s_-]+', '', filename)
    return normalized.lower()


def _find_similar_file(file_path: str) -> Optional[str]:
    """
    查找相似文件名（模糊匹配）
    
    当精确匹配失败时，尝试在同目录下查找相似文件名。
    匹配规则：
    1. 相同扩展名
    2. 标准化后的文件名相似度 >= 80%
    
    Args:
        file_path: 原始文件路径
        
    Returns:
        匹配到的文件路径，如果没有找到则返回 None
    """
    try:
        from difflib import SequenceMatcher
        
        path = Path(file_path)
        parent_dir = path.parent
        target_name = path.name
        target_stem = path.stem
        target_ext = path.suffix.lower()
        target_normalized = _normalize_filename(target_name)
        
        if not parent_dir.exists():
            return None
        
        best_match = None
        best_ratio = 0.0
        
        # 遍历目录中的所有文件
        for f in parent_dir.iterdir():
            if not f.is_file():
                continue
            
            # 扩展名必须匹配
            if f.suffix.lower() != target_ext:
                continue
            
            # 计算相似度
            candidate_normalized = _normalize_filename(f.name)
            ratio = SequenceMatcher(None, target_normalized, candidate_normalized).ratio()
            
            # 更新最佳匹配
            if ratio > best_ratio and ratio >= 0.8:  # 80% 相似度阈值
                best_ratio = ratio
                best_match = str(f)
        
        return best_match
        
    except Exception:
        return None


def should_use_full_text(text_length: int, model_name: Optional[str] = None) -> bool:
    """
    判断是否应该直接加载全文而不走向量检索
    
    策略：
    - 获取模型上下文窗口大小
    - 预留 40% 给对话历史、系统提示、工具结果
    - 30% 可用于文档（保守估计）
    - 估算 token 数（中文字符 ≈ 0.6 token）
    
    Args:
        text_length: 文本长度（字符数）
        model_name: 模型名称
        
    Returns:
        True 表示直接加载全文，False 表示走向量检索
    """
    try:
        from ollamapilot.model_context import get_recommended_num_ctx
        
        model = model_name or _model_name
        if not model:
            return text_length < 2000
        
        max_ctx = get_recommended_num_ctx(model)
        threshold = int(max_ctx * 0.3)
        estimated_tokens = text_length * 0.6
        
        return estimated_tokens < threshold
    except Exception:
        return text_length < 2000


def get_full_text_threshold(model_name: Optional[str] = None) -> int:
    """
    获取全文加载的字符数阈值
    
    Args:
        model_name: 模型名称
        
    Returns:
        字符数阈值
    """
    try:
        from ollamapilot.model_context import get_recommended_num_ctx
        
        model = model_name or _model_name
        if not model:
            return 2000
        
        max_ctx = get_recommended_num_ctx(model)
        token_threshold = int(max_ctx * 0.3)
        char_threshold = int(token_threshold / 0.6)
        
        return char_threshold
    except Exception:
        return 2000


def get_graph_service() -> Optional[GraphRAGService]:
    """获取图谱服务实例"""
    return _graph_service


@tool
def upload_document(
    file_path: str,
    save_to_knowledge_base: bool = True
) -> str:
    """
    上传文档到知识库并建立完整索引（长期保存）

    将文档保存到知识库目录，进行分块、实体抽取、关系建立等完整索引流程。
    适用于需要长期保存、多次查询的文档。
    支持 PDF、TXT、MD、DOCX 等格式。

    使用场景：
    - 用户明确要求"添加到知识库"、"建立索引"
    - 需要长期保存的重要文档
    - 会被多次查询引用的文档

    注意：所有文档（无论长短）都会建立完整索引，确保检索质量。

    Args:
        file_path: 文档路径（可以是任意位置的文件）
        save_to_knowledge_base: 是否保存到知识库目录，默认 True

    Returns:
        处理结果
    """
    if not _graph_service or not _entity_extractor:
        return "❌ 服务未初始化"

    try:
        source_path = Path(file_path)
        
        # 文件不存在时尝试模糊匹配
        if not source_path.exists():
            similar_file = _find_similar_file(file_path)
            if similar_file:
                logger.info(f"文件名模糊匹配: '{file_path}' -> '{similar_file}'")
                source_path = Path(similar_file)
                file_path = similar_file
            else:
                return f"文件不存在: {file_path}"

        # 检查文件类型
        supported_ext = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        if source_path.suffix.lower() not in supported_ext:
            return f"不支持的文件类型: {source_path.suffix}，支持的类型: {', '.join(supported_ext)}"

        # 知识库文档保存到知识库目录
        save_dir = Path(_knowledge_base_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存文档
        if save_to_knowledge_base:
            # 目标路径
            dest_path = save_dir / source_path.name

            # 如果文件已存在，添加数字后缀
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                dest_path = save_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            # 复制文件
            import shutil
            shutil.copy2(source_path, dest_path)
            logger.info(f"已保存到知识库: {dest_path}")

            # 使用新路径继续处理
            file_path = str(dest_path)

        # 读取文档
        logger.debug(f"正在读取文档: {Path(file_path).name}")
        try:
            embedding_model = _graph_service.embedding_model if _graph_service else None
            processor = DocumentProcessor.from_model_name(embedding_model) if embedding_model else DocumentProcessor()
        except Exception:
            processor = DocumentProcessor()
        text = processor.read_document(file_path)

        if not text:
            return f"无法读取文档内容: {file_path}"

        logger.debug(f"文档读取完成，长度: {len(text)} 字符")
        logger.info(f"知识库模式：建立完整索引（文档长度: {len(text)} 字符）")

        # 长文档：正常索引流程
        logger.debug("正在分块...")
        chunks = processor.chunk_text(text)
        logger.info(f"分块完成，共 {len(chunks)} 块")

        # 处理每个块
        logger.debug(f"正在处理 {len(chunks)} 个块...")
        total_entities = 0
        total_relations = 0
        for i, chunk in enumerate(chunks):
            try:
                # 使用混合模式抽取实体和关系
                entities, relations = _entity_extractor.extract(
                    chunk,
                    use_llm=_use_llm,
                    llm_client=_llm_client,
                    top_k=20
                )

                # 转换为 Entity 对象
                entity_objects = [
                    Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                    for e in entities
                ]

                # 添加到图谱
                doc_prefix = "temp" if is_temporary else "kb"
                doc_id = f"{doc_prefix}_{hash(file_path) % 10000}_{i}"
                _graph_service.add_document(
                    text=chunk,
                    doc_id=doc_id,
                    metadata={
                        "source": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": Path(file_path).name,
                        "is_short_doc": False,
                        "is_temporary": is_temporary,
                        "full_text": None
                    },
                    entities=entity_objects
                )

                total_entities += len(entities)
                total_relations += len(relations)

                if (i + 1) % 10 == 0:
                    logger.debug(f"已处理 {i + 1}/{len(chunks)} 块...")

            except Exception as e:
                logger.warning(f"处理第 {i+1} 块时出错: {e}")
                continue

        logger.info("所有块处理完成")

        stats = _graph_service.get_stats()
        dest_info = f"已保存到知识库: {Path(file_path).name}\n" if save_to_knowledge_base else ""
        return f"""文档上传完成！
{dest_info}- 文件：{Path(file_path).name}
- 文档长度：{len(text)} 字符
- 分块数：{len(chunks)}
- 新抽取实体：{total_entities}
- 图谱总实体数：{stats['total_entities']}
- 图谱总关系数：{stats['total_relations']}

现在可以直接询问与该文档相关的问题了。"""

    except Exception as e:
        return f"上传失败：{str(e)}"


@tool
def add_document(
    file_path: str,
    collection_name: Optional[str] = "default"
) -> str:
    """
    添加文档到知识图谱（不复制到知识库目录）

    文档会被自动分块、抽取实体、建立索引，但不会保存到知识库目录。
    适用于临时添加文档或文档已在知识库目录中的情况。

    Args:
        file_path: 文档路径
        collection_name: 集合名称

    Returns:
        处理结果
    """
    # 直接调用 upload_document，但不保存到知识库
    return upload_document(file_path, save_to_knowledge_base=False)


@tool
def add_text(
    text: str,
    source: str = "manual_input",
    collection_name: Optional[str] = "default"
) -> str:
    """
    添加文本到知识图谱

    Args:
        text: 文本内容
        source: 来源标识
        collection_name: 集合名称

    Returns:
        处理结果
    """
    if not _graph_service or not _entity_extractor:
        return "❌ 服务未初始化"

    try:
        # 分块（根据当前 embedding 模型动态选择分块大小）
        try:
            embedding_model = _graph_service.embedding_model if _graph_service else None
            processor = DocumentProcessor.from_model_name(embedding_model) if embedding_model else DocumentProcessor()
        except Exception:
            processor = DocumentProcessor()
        chunks = processor.chunk_text(text)

        # 处理每个块
        total_entities = 0
        total_relations = 0
        for i, chunk in enumerate(chunks):
            # 使用混合模式抽取实体和关系
            entities, relations = _entity_extractor.extract(
                chunk,
                use_llm=_use_llm,
                llm_client=_llm_client,
                top_k=20
            )

            # 转换为 Entity 对象
            entity_objects = [
                Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                for e in entities
            ]

            # 添加到图谱
            doc_id = f"{collection_name}_{source}_{i}"
            _graph_service.add_document(
                text=chunk,
                doc_id=doc_id,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "collection": collection_name
                },
                entities=entity_objects
            )

            total_entities += len(entities)
            total_relations += len(relations)

        stats = _graph_service.get_stats()
        return f"""✅ 文本处理完成！
- 来源：{source}
- 分块数：{len(chunks)}
- 新抽取实体：{total_entities}
- 新抽取关系：{total_relations}
- 图谱总实体数：{stats['total_entities']}
- 图谱总关系数：{stats['total_relations']}"""

    except Exception as e:
        return f"❌ 处理失败：{str(e)}"


@tool
def generate_ontology(document_text: str) -> str:
    """
    从文档生成本体定义

    Args:
        document_text: 文档文本（前10000字符）

    Returns:
        本体定义 JSON
    """
    if not _ontology_generator:
        return "❌ 本体生成器未初始化"

    try:
        ontology = _ontology_generator.generate([document_text])

        # 设置到服务
        _graph_service.set_ontology(ontology)

        entity_types = "\n".join([f"- {e['name']}: {e['description']}" for e in ontology['entity_types']])
        relation_types = "\n".join([f"- {r['name']}: {r['description']}" for r in ontology['relation_types']])

        return f"""✅ 本体定义生成完成！

实体类型（{len(ontology['entity_types'])}个）：
{entity_types}

关系类型（{len(ontology['relation_types'])}个）：
{relation_types}

摘要：{ontology.get('analysis_summary', '无')}"""

    except Exception as e:
        return f"❌ 生成失败：{str(e)}"


@tool
def query_graph_stats() -> str:
    """
    查询知识图谱统计信息

    Returns:
        统计信息
    """
    try:
        # 优先使用 DocumentManager 获取全局统计
        if _document_manager:
            stats = _document_manager.get_global_stats()
            entity_types = ', '.join(stats['entity_types']) if stats['entity_types'] else '无'

            return f"""📊 知识图谱统计（所有文档）：
- 文档数：{stats['total_documents']}
- 实体数：{stats['total_entities']}
- 关系数：{stats['total_relations']}
- 实体类型：{entity_types}"""

        # 回退到全局 GraphRAGService
        elif _graph_service:
            stats = _graph_service.get_stats()
            entity_types = ', '.join(stats['entity_types']) if stats['entity_types'] else '无'

            return f"""📊 知识图谱统计：
- 文档数：{stats['total_documents']}
- 实体数：{stats['total_entities']}
- 关系数：{stats['total_relations']}
- 实体类型：{entity_types}"""

        else:
            return "❌ 服务未初始化"

    except Exception as e:
        return f"❌ 获取统计信息失败：{str(e)}"


@tool
def search_all_documents(
    query: str,
    n_results: int = 5
) -> str:
    """
    搜索所有文档（全局搜索）
    
    在整个知识库中搜索，不限定特定分类。
    当用户没有指定具体分类时使用此工具。
    
    对于短文档，会直接返回全文内容，无需向量检索。

    Args:
        query: 查询文本
        n_results: 返回结果数量

    Returns:
        搜索结果
    """
    if not _entity_extractor:
        return "❌ 服务未初始化"

    try:
        # 提取查询实体
        query_entities = _entity_extractor.extract_from_query(query)

        # 优先使用 DocumentManager 搜索所有文档
        if _document_manager:
            # 先检查是否有短文档，直接返回全文
            short_docs = []
            for doc_id, doc_info in _document_manager.documents.items():
                if doc_info.status.name != "COMPLETED":
                    continue
                try:
                    graph_service = _document_manager._get_cached_graph_service(doc_id, doc_info)
                    all_docs = graph_service.collection.get(include=["metadatas"])
                    
                    for i, metadata in enumerate(all_docs.get("metadatas", [])):
                        if metadata.get("is_short_doc") and metadata.get("full_text"):
                            short_docs.append({
                                "document_name": doc_info.name,
                                "full_text": metadata["full_text"],
                                "source": metadata.get("source", "未知")
                            })
                            break
                except Exception:
                    continue
            
            if short_docs:
                output = [f"📄 短文档全文加载 ({len(short_docs)} 个文档)",
                          "=" * 50]
                for i, doc in enumerate(short_docs, 1):
                    output.append(f"\n[{i}] 来源: {doc['document_name']}")
                    output.append(f"{'─' * 40}")
                    output.append(doc['full_text'][:2000])
                    if len(doc['full_text']) > 2000:
                        output.append(f"\n... (共 {len(doc['full_text'])} 字符)")
                return "\n".join(output)
            
            # 正常向量检索
            results = _document_manager.search_all_documents(query, n_results=n_results)

            if query_entities:
                entity_info = f"提取到实体: {', '.join([e['name'] for e in query_entities])}"
            else:
                entity_info = "未提取到实体，使用向量检索"

            if not results:
                return f"🔍 未找到相关文档\n{entity_info}"

            # 格式化结果
            output = [f"🔍 搜索结果 ({len(results)} 条)",
                      f"{entity_info}",
                      "=" * 50]

            # 显示文档
            output.append("\n【相关文档】")
            for i, doc in enumerate(results, 1):
                source = doc.get("document_name", "未知")
                score = doc.get("score", 0)
                content = doc.get("content", "")[:300]

                output.append(f"\n[{i}] 来源: {source} | 相关度: {score:.2f}")
                output.append(f"    {content}...")

            return "\n".join(output)

        # 回退到全局 GraphRAGService 搜索
        elif _graph_service:
            if query_entities:
                results = _graph_service.enhanced_search(
                    query=query,
                    query_entities=query_entities,
                    n_results=n_results
                )
                entity_info = f"提取到实体: {', '.join([e['name'] for e in query_entities])}"
            else:
                results = {
                    "documents": _graph_service.vector_search(query, n_results=n_results),
                    "query_entities": [],
                    "relations": [],
                    "related_entities": []
                }
                entity_info = "未提取到实体，使用向量检索"

            if not results["documents"]:
                return f"🔍 未找到相关文档\n{entity_info}"

            # 格式化结果
            output = [f"🔍 搜索结果 ({len(results['documents'])} 条)",
                      f"{entity_info}",
                      "=" * 50]

            # 显示关系
            if results.get("relations"):
                output.append("\n【相关关系】")
                for rel in results["relations"][:3]:
                    output.append(f"- {rel['source']} --{rel['relation']}--> {rel['target']}")

            # 显示文档
            output.append("\n【相关文档】")
            for i, doc in enumerate(results["documents"], 1):
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "未知")
                score = doc.get("score", 0)
                content = doc.get("content", "")[:300]

                output.append(f"\n[{i}] 来源: {source} | 相关度: {score:.2f}")
                output.append(f"    {content}...")

            return "\n".join(output)

        else:
            return "❌ 搜索服务未初始化"

    except Exception as e:
        return f"❌ 搜索失败：{str(e)}"


@tool
def list_entities(entity_type: Optional[str] = None) -> str:
    """
    列出知识图谱中的实体

    Args:
        entity_type: 实体类型过滤（可选）

    Returns:
        实体列表
    """
    if not _graph_service:
        return "❌ 服务未初始化"

    try:
        entities = _graph_service.entity_index

        if not entities:
            return "📭 知识图谱中暂无实体"

        # 过滤实体类型
        filtered_entities = {}
        for name, info in entities.items():
            if entity_type is None or info["type"] == entity_type:
                filtered_entities[name] = info

        if not filtered_entities:
            return f"📭 未找到类型为 '{entity_type}' 的实体"

        # 按类型分组
        by_type: Dict[str, List[str]] = {}
        for name, info in filtered_entities.items():
            etype = info["type"]
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(name)

        # 格式化输出
        output = [f"📋 实体列表 (共 {len(filtered_entities)} 个)", "=" * 50]

        for etype, names in sorted(by_type.items()):
            output.append(f"\n【{etype}】({len(names)}个)")
            for name in sorted(names)[:20]:  # 每类最多显示20个
                doc_count = len(entities[name]["doc_ids"])
                output.append(f"  - {name} (出现在 {doc_count} 个文档)")

            if len(names) > 20:
                output.append(f"  ... 还有 {len(names) - 20} 个")

        return "\n".join(output)

    except Exception as e:
        return f"❌ 查询失败：{str(e)}"


@tool
def get_entity_relations(entity_name: str) -> str:
    """
    获取实体的关系

    Args:
        entity_name: 实体名称

    Returns:
        关系信息
    """
    if not _graph_service:
        return "❌ 服务未初始化"

    try:
        # 获取实体信息
        if entity_name not in _graph_service.entity_index:
            return f"❌ 未找到实体: {entity_name}"

        entity_info = _graph_service.entity_index[entity_name]
        doc_ids = entity_info["doc_ids"]

        # 获取关系
        relations = _graph_service.get_relations(entity_name)

        # 格式化输出
        output = [
            f"🔍 实体: {entity_name}",
            f"类型: {entity_info['type']}",
            f"出现在 {len(doc_ids)} 个文档中",
            "=" * 50
        ]

        if relations:
            output.append(f"\n【关系】({len(relations)}个)")
            for rel in relations:
                output.append(
                    f"- {rel['source']} --{rel['relation']}--> {rel['target']} "
                    f"(置信度: {rel['confidence']:.2f})"
                )
        else:
            output.append("\n【关系】暂无")

        return "\n".join(output)

    except Exception as e:
        return f"❌ 查询失败：{str(e)}"


@tool
def search_in_category(
    category: str,
    query: str,
    n_results: int = 5
) -> str:
    """
    在指定分类中搜索（分类搜索）
    
    只在用户指定的分类文件夹中搜索，不搜索其他分类。
    当用户明确说了分类名称（如"伤寒论"、"中医经典"）时使用此工具。
    
    分类是 data/graphrag/ 下的文件夹，由用户自定义名称。
    
    Args:
        category: 分类名称（如"伤寒论"、"中医经典"、"中医经典/伤寒论"）
        query: 查询内容
        n_results: 返回结果数量，默认5条
        
    Returns:
        搜索结果
        
    Example:
        search_in_category(category="伤寒论", query="太阳病")
        search_in_category(category="中医经典", query="金匮要略")
    """
    if not _document_manager:
        return "❌ 文档管理器未初始化"
    
    try:
        # 搜索指定分类
        results = _document_manager.search_by_category(category, query, n_results)
        
        if not results:
            # 检查分类是否存在
            categories = _document_manager.list_categories()
            if category not in categories:
                return f"❌ 分类 '{category}' 不存在\n   可用分类: {', '.join(categories) if categories else '无'}\n   提示: 请先在 data/graphrag/ 下创建文件夹并移动文档"
            else:
                return f"🔍 在 '{category}' 分类中未找到相关内容"
        
        # 格式化结果
        output = [
            f"🔍 知识库搜索结果 [{category}] ({len(results)} 条)",
            "=" * 50
        ]
        
        for i, doc in enumerate(results, 1):
            doc_name = doc.get("document_name", "未知")
            score = doc.get("score", 0)
            content = doc.get("content", "")[:400]
            
            output.append(f"\n[{i}] 来源: {doc_name} | 相关度: {score:.2f}")
            output.append(f"    {content}...")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ 搜索失败: {str(e)}"


@tool
def list_knowledge_categories() -> str:
    """
    列出所有可用的知识库分类
    
    显示你在 data/graphrag/ 目录下创建的所有分类文件夹。
    
    Returns:
        分类列表
    """
    if not _document_manager:
        return "❌ 文档管理器未初始化"
    
    try:
        categories = _document_manager.list_categories()
        
        if not categories:
            return "📂 暂无知识库分类\n   提示: 在 data/graphrag/ 下创建文件夹即可建立分类"
        
        output = ["📚 可用知识库分类:", "=" * 30]
        for i, cat in enumerate(categories, 1):
            output.append(f"  {i}. {cat}")
        
        output.append(f"\n💡 使用方式: search_in_category(category='分类名', query='查询内容')")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ 获取分类列表失败: {str(e)}"


# ========== 实时模式工具（新增）==========

@tool
def analyze_document(file_path: str, query: Optional[str] = None) -> str:
    """
    实时分析文档内容（一次性查询，不保存到知识库）

    适用于临时上传文档进行快速分析，无需提前索引：
    - 短文档（<阈值）：直接返回全文内容
    - 中长文档（>=阈值）：建立临时索引 → 检索相关内容

    文档会被保存到临时目录，可定期清理。
    如果只需要文档总结，可以不传 query 参数。

    使用场景：
    - 用户说"分析一下这个文档"
    - 临时上传的PDF、Word等需要快速查看内容
    - 一次性查询，不需要长期保存

    Args:
        file_path: 文档路径
        query: 查询内容（可选，不传则返回文档总结）

    Returns:
        分析结果或检索结果
    """
    if not _graph_service or not _entity_extractor:
        return "❌ 服务未初始化"
    
    try:
        from pathlib import Path
        import shutil
        import time

        source_path = Path(file_path)

        if not source_path.exists():
            similar_file = _find_similar_file(file_path)
            if similar_file:
                logger.info(f"文件名模糊匹配: '{file_path}' -> '{similar_file}'")
                source_path = Path(similar_file)
                file_path = similar_file
            else:
                return f"文件不存在: {file_path}"

        # 保存到临时目录
        temp_dir = Path(_temp_documents_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 生成临时文件名
        timestamp = int(time.time())
        temp_filename = f"temp_{timestamp}_{source_path.name}"
        temp_file_path = temp_dir / temp_filename

        # 复制文件到临时目录
        shutil.copy2(source_path, temp_file_path)
        logger.info(f"已保存到临时文档: {temp_file_path}")

        logger.debug(f"实时分析文档: {source_path.name}")

        # 1. 读取文档
        try:
            embedding_model = _graph_service.embedding_model if _graph_service else None
            processor = DocumentProcessor.from_model_name(embedding_model) if embedding_model else DocumentProcessor()
        except Exception:
            processor = DocumentProcessor()

        text = processor.read_document(file_path)
        if not text:
            return f"无法读取文档内容: {file_path}"

        logger.debug(f"文档读取完成，长度: {len(text)} 字符")

        # 2. 判断是否为短文档，直接返回全文
        if should_use_full_text(len(text)):
            threshold = get_full_text_threshold()
            logger.debug(f"短文档检测（{len(text)} < {threshold} 字符），直接返回全文")

            # 短文档直接返回全文
            output = [
                f"📄 实时文档分析（短文档模式）",
                f"文档: {source_path.name}",
                f"长度: {len(text)} 字符",
                "=" * 50,
            ]

            if query:
                # 如果有查询，在全文内搜索相关内容
                output.append(f"\n【查询】{query}")
                # 简单关键词匹配
                query_keywords = query.lower().split()
                paragraphs = text.split('\n')
                relevant_parts = []
                for para in paragraphs:
                    if any(kw in para.lower() for kw in query_keywords):
                        relevant_parts.append(para)

                if relevant_parts:
                    output.append("\n【相关内容】")
                    output.append('\n'.join(relevant_parts[:10]))
                else:
                    output.append("\n【文档全文】")
                    output.append(text[:3000])
            else:
                # 无查询，返回全文
                output.append("\n【文档全文】")
                output.append(text[:5000])

            if len(text) > 5000:
                output.append(f"\n... (共 {len(text)} 字符，已截断)")

            return "\n".join(output)

        # ===== 中长文档处理 =====
        # 分块
        chunks = processor.chunk_text(text)
        logger.info(f"分块完成，共 {len(chunks)} 块")

        # 如果没有查询，返回文档总结
        if not query:
            output = [
                f"📄 实时文档分析（中长文档模式）",
                f"文档: {source_path.name}",
                f"长度: {len(text)} 字符",
                f"分块: {len(chunks)} 块",
                "=" * 50,
                "\n【文档开头】",
                text[:2000],
                f"\n... (共 {len(text)} 字符)",
                "\n💡 提示：如需查询特定内容，请提供查询关键词"
            ]
            return "\n".join(output)

        # 有查询，建立临时索引并检索
        logger.info(f"建立临时索引并检索: {query}")

        # 临时索引
        temp_entities = []
        temp_docs = []

        for i, chunk in enumerate(chunks):
            try:
                # 抽取实体
                entities, _ = _entity_extractor.extract(
                    chunk,
                    use_llm=_use_llm,
                    llm_client=_llm_client,
                    top_k=10
                )

                temp_entities.extend(entities)
                temp_docs.append({
                    "content": chunk,
                    "index": i,
                    "entities": entities
                })
            except Exception as e:
                logger.warning(f"处理第 {i+1} 块时出错: {e}")
                continue

        logger.info(f"临时索引完成，实体: {len(temp_entities)}")

        # 检索
        query_entities = _entity_extractor.extract_from_query(query)
        query_entity_names = {e['name'] for e in query_entities}

        results = []
        for doc in temp_docs:
            doc_entity_names = {e.name for e in doc['entities']}
            overlap = query_entity_names & doc_entity_names
            if overlap:
                results.append({
                    "content": doc['content'],
                    "score": len(overlap),
                    "source": f"块 {doc['index']}"
                })

        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:5]

        # 格式化结果
        if not results:
            return f"🔍 未找到与 '{query}' 相关的内容\n   文档已处理: {len(chunks)} 块, {len(temp_entities)} 实体"

        output = [
            f"🔍 实时检索结果 ({len(results)} 条)",
            f"文档: {source_path.name}",
            f"查询: {query}",
            f"处理: {len(chunks)} 块, {len(temp_entities)} 实体",
            "=" * 50
        ]

        for i, doc in enumerate(results, 1):
            content = doc.get("content", "")[:500]
            source = doc.get("source", "未知")
            output.append(f"\n[{i}] 来源: {source}")
            output.append(f"    {content}...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ 实时分析失败: {str(e)}"


@tool
def add_text_and_search(text: str, query: str, n_results: int = 5) -> str:
    """
    实时模式：直接粘贴文本并检索
    
    无需提前索引，直接粘贴文本即可检索。
    短文本直接返回全文，长文本走检索流程。
    适用于临时文本分析场景。
    
    Args:
        text: 文本内容
        query: 查询内容
        n_results: 返回结果数量
        
    Returns:
        检索结果
    """
    if not _graph_service or not _entity_extractor:
        return "服务未初始化"

    try:
        import tempfile

        # 1. 判断是否为短文本，直接返回全文
        if should_use_full_text(len(text)):
            threshold = get_full_text_threshold()
            logger.debug(f"短文本检测（{len(text)} < {threshold} 字符），直接返回全文")
            
            output = [
                f"📄 实时文本分析（短文本模式）",
                f"长度: {len(text)} 字符",
                "=" * 50,
                "\n【文本全文】",
                text[:3000]
            ]
            
            if len(text) > 3000:
                output.append(f"\n... (共 {len(text)} 字符)")
            
            return "\n".join(output)
        
        # 2. 长文本：实时处理
        try:
            embedding_model = _graph_service.embedding_model if _graph_service else None
            processor = DocumentProcessor.from_model_name(embedding_model) if embedding_model else DocumentProcessor()
        except Exception:
            processor = DocumentProcessor()

        # 分块
        chunks = processor.chunk_text(text)
        logger.info(f"文本分块完成，共 {len(chunks)} 块")

        # 2. 临时索引
        temp_entities = []
        temp_docs = []

        for i, chunk in enumerate(chunks):
            try:
                entities, _ = _entity_extractor.extract(
                    chunk,
                    use_llm=_use_llm,
                    llm_client=_llm_client,
                    top_k=20
                )

                temp_entities.extend(entities)
                temp_docs.append({
                    "content": chunk,
                    "index": i,
                    "entities": entities
                })
            except Exception as e:
                logger.warning(f"处理第 {i+1} 块时出错: {e}")
                continue
        
        logger.info(f"临时索引完成，实体: {len(temp_entities)}")
        
        # 3. 立即检索
        query_entities = _entity_extractor.extract_from_query(query)
        query_entity_names = {e['name'] for e in query_entities}
        
        results = []
        for doc in temp_docs:
            doc_entity_names = {e.name for e in doc['entities']}
            overlap = query_entity_names & doc_entity_names
            if overlap:
                results.append({
                    "content": doc['content'],
                    "score": len(overlap) / len(query_entity_names) if query_entity_names else 0,
                    "source": f"文本块 {doc['index']}",
                    "match_type": "entity"
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:n_results]
        
        # 4. 格式化结果
        if not results:
            return f"🔍 未找到与 '{query}' 相关的内容\n   文本已处理: {len(chunks)} 块, {len(temp_entities)} 实体"
        
        output = [
            f"🔍 实时文本检索结果 ({len(results)} 条)",
            f"处理: {len(chunks)} 块, {len(temp_entities)} 实体",
            "=" * 50
        ]
        
        for i, doc in enumerate(results, 1):
            score = doc.get("score", 0)
            content = doc.get("content", "")[:400]
            source = doc.get("source", "未知")
            
            output.append(f"\n[{i}] {source} | 匹配度: {score:.2f}")
            output.append(f"    {content}...")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ 实时文本检索失败: {str(e)}"


# 工具列表
graphrag_tools = [
    upload_document,             # 上传文档到知识库（长期保存）
    analyze_document,            # 实时分析文档（一次性查询）
    add_document,
    add_text,
    add_text_and_search,         # 实时模式：添加文本并检索
    generate_ontology,
    query_graph_stats,
    search_all_documents,        # 搜索所有文档
    search_in_category,          # 搜索指定分类
    list_knowledge_categories,   # 列出分类
    list_entities,
    get_entity_relations,
]
