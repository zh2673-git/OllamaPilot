"""
GraphRAG 工具集

提供知识图谱相关的工具函数。
"""

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


# 全局服务实例（由 Agent 初始化时注入）
_graph_service: Optional[GraphRAGService] = None
_entity_extractor: Optional[HybridEntityExtractor] = None
_llm_client: Optional[SimpleLLMClient] = None
_use_llm: bool = False
_knowledge_base_dir: str = "./knowledge_base"
_document_manager: Optional[Any] = None  # DocumentManager 实例


def init_graphrag_services(
    service: GraphRAGService,
    ext: Optional[HybridEntityExtractor] = None,
    kb_dir: str = "./knowledge_base",
    doc_manager: Optional[Any] = None
):
    """初始化服务"""
    global _graph_service, _entity_extractor, _llm_client, _use_llm, _knowledge_base_dir, _document_manager
    _graph_service = service
    _entity_extractor = ext or HybridEntityExtractor(persist_dir=service.persist_dir)
    _llm_client = SimpleLLMClient()
    _use_llm = _llm_client.is_available()
    _knowledge_base_dir = kb_dir
    _document_manager = doc_manager


def get_graph_service() -> Optional[GraphRAGService]:
    """获取图谱服务实例"""
    return _graph_service


@tool
def upload_document(
    file_path: str,
    save_to_knowledge_base: bool = True
) -> str:
    """
    上传文档到知识库并建立索引

    文档会被自动保存到知识库目录，然后分块、抽取实体、建立索引。
    支持 PDF、TXT、MD、DOCX 等格式。

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
        if not source_path.exists():
            return f"❌ 文件不存在: {file_path}"

        # 检查文件类型
        supported_ext = {'.txt', '.md', '.pdf', '.docx', '.doc'}
        if source_path.suffix.lower() not in supported_ext:
            return f"❌ 不支持的文件类型: {source_path.suffix}，支持的类型: {', '.join(supported_ext)}"

        # 保存到知识库目录
        if save_to_knowledge_base:
            kb_path = Path(_knowledge_base_dir)
            kb_path.mkdir(parents=True, exist_ok=True)

            # 目标路径
            dest_path = kb_path / source_path.name

            # 如果文件已存在，添加数字后缀
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                dest_path = kb_path / f"{stem}_{counter}{suffix}"
                counter += 1

            # 复制文件
            import shutil
            shutil.copy2(source_path, dest_path)
            print(f"📁 已保存到知识库: {dest_path}")

            # 使用新路径继续处理
            file_path = str(dest_path)

        # 读取文档
        print(f"📖 正在读取文档: {Path(file_path).name}")
        processor = DocumentProcessor()
        text = processor.read_document(file_path)

        if not text:
            return f"❌ 无法读取文档内容: {file_path}"

        print(f"✅ 文档读取完成，长度: {len(text)} 字符")

        # 分块
        print("🔄 正在分块...")
        chunks = processor.chunk_text(text)
        print(f"✅ 分块完成，共 {len(chunks)} 块")

        # 处理每个块
        print(f"🔄 正在处理 {len(chunks)} 个块...")
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
                doc_id = f"kb_{hash(file_path) % 10000}_{i}"
                _graph_service.add_document(
                    text=chunk,
                    doc_id=doc_id,
                    metadata={
                        "source": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": Path(file_path).name
                    },
                    entities=entity_objects
                )

                total_entities += len(entities)
                total_relations += len(relations)

                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1}/{len(chunks)} 块...")

            except Exception as e:
                print(f"  ⚠️ 处理第 {i+1} 块时出错: {e}")
                continue

        print(f"✅ 所有块处理完成")

        stats = _graph_service.get_stats()
        dest_info = f"📁 已保存到知识库: {Path(file_path).name}\n" if save_to_knowledge_base else ""
        return f"""✅ 文档上传完成！
{dest_info}- 文件：{Path(file_path).name}
- 分块数：{len(chunks)}
- 新抽取实体：{total_entities}
- 图谱总实体数：{stats['total_entities']}
- 图谱总关系数：{stats['total_relations']}

现在可以直接询问与该文档相关的问题了。"""

    except Exception as e:
        return f"❌ 上传失败：{str(e)}"


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
        # 分块
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
def search_knowledge(
    query: str,
    n_results: int = 5
) -> str:
    """
    搜索知识图谱

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
def search_knowledge_base(
    category: str,
    query: str,
    n_results: int = 5
) -> str:
    """
    搜索指定分类的知识库
    
    在指定的知识库分类中搜索相关内容。分类是你手动创建的文件夹，
    名称由用户自定义（如"中医经典"、"现代医学"等）。
    
    支持多级文件夹结构：
    - 一级分类：data/graphrag/中医经典/
    - 二级分类：data/graphrag/中医经典/伤寒论/
    - 可直接搜索一级分类，会自动包含所有子目录
    
    使用前请确保：
    1. 已在 data/graphrag/ 下创建分类文件夹
    2. 已将相关文档的向量存储移动到该文件夹
    
    Args:
        category: 知识库分类名称（用户自定义的文件夹名），支持多级如"中医经典/伤寒论"
        query: 查询内容
        n_results: 返回结果数量，默认5条
        
    Returns:
        搜索结果
        
    Example:
        search_knowledge_base(category="中医经典", query="伤寒论治疗方法")
        search_knowledge_base(category="中医经典/伤寒论", query="太阳病")
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
        
        output.append(f"\n💡 使用方式: search_knowledge_base(category='分类名', query='查询内容')")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ 获取分类列表失败: {str(e)}"


# 工具列表
graphrag_tools = [
    upload_document,  # 新增：上传文档到知识库
    add_document,
    add_text,
    generate_ontology,
    query_graph_stats,
    search_knowledge,        # 搜索所有文档
    search_knowledge_base,   # 搜索指定分类
    list_knowledge_categories,  # 列出分类
    list_entities,
    get_entity_relations,
]
