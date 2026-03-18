"""
GraphRAG Skill - 知识图谱检索增强

基于实体-关系的智能文档问答 Skill。
作为独立的 Python Skill，遵循 USB 即插即用设计理念。
支持从 .env 配置文件读取设置。
"""

from typing import List, Optional, Any
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware
import threading
import time

from ollamapilot.skills.base import Skill
from ollamapilot.config import get_config

# GraphRAG Skill 内部模块
from skills.graphrag.services import (
    GraphRAGService,
    HybridEntityExtractor,
    Entity
)
from skills.graphrag.llm_client import SimpleLLMClient
from skills.graphrag.middleware import GraphRAGMiddleware
from skills.graphrag.utils import DocumentProcessor

from skills.graphrag.tools import (
    upload_document,
    add_document,
    add_text,
    analyze_document,
    add_text_and_search,
    query_graph_stats,
    search_all_documents,
    search_in_category,
    list_knowledge_categories,
    list_entities,
    get_entity_relations,
    init_graphrag_services,
)
from skills.graphrag.knowledge_base import KnowledgeBaseManager

# 获取配置
config = get_config()


class GraphRAGSkill(Skill):
    """
    GraphRAG Skill - 知识图谱检索增强

    提供基于实体-关系的智能文档问答能力。
    包含完整的知识图谱管理工具集和自动检索中间件。

    特点:
    - 混合存储: ChromaDB向量 + 实体索引 + 关系索引
    - 轻量级实体抽取: 规则+词典匹配，适合小模型
    - 自动检索增强: 通过中间件自动注入相关知识
    - 即插即用: 作为独立Skill，不影响其他功能

    使用方法:
        # 1. 添加文档到知识图谱
        add_document(file_path="./document.pdf")

        # 2. 查询时自动触发GraphRAG检索
        # 系统会自动提取查询中的实体，检索相关文档和关系

        # 3. 手动搜索知识图谱
        search_all_documents(query="张三在哪里工作？")

    触发词:
        - 知识图谱、文档问答、知识库
        - 检索、实体、关系
        - 添加文档、搜索知识
    """

    name = "graphrag"
    description = "知识图谱检索增强 - 基于实体-关系的智能文档问答"
    tags = ["知识图谱", "RAG", "文档问答", "检索增强"]
    version = "1.0.0"
    triggers = [
        "知识图谱", "文档问答", "知识库",
        "检索", "实体", "关系",
        "添加文档", "上传文档", "搜索知识",
        ".pdf", ".txt", ".md", ".docx", ".doc"
    ]

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_dir: Optional[str] = None,
        enable_auto_retrieval: bool = True,
        knowledge_base_dir: Optional[str] = None,
        enable_word_aligner: Optional[bool] = None,
        fuzzy_threshold: Optional[float] = None,
        use_config: bool = True,
        model_name: Optional[str] = None
    ):
        """
        初始化 GraphRAG Skill

        Args:
            embedding_model: Embedding模型名称，默认从 .env 读取
            persist_dir: 数据持久化目录，默认从 .env 读取
            enable_auto_retrieval: 是否启用自动检索中间件
            knowledge_base_dir: 知识库目录路径，默认从 .env 读取
            enable_word_aligner: 是否启用 WordAligner，默认从 .env 读取
            fuzzy_threshold: 模糊匹配阈值，默认从 .env 读取
            use_config: 是否使用配置文件
            model_name: 主模型名称（用于动态上下文判断）
        """
        super().__init__()
        
        self.model_name = model_name
        
        # 从配置文件获取默认值
        if use_config:
            self.embedding_model = embedding_model or config.embedding_model
            self.persist_dir = persist_dir or config.graph_rag_persist_dir
            self.knowledge_base_dir = knowledge_base_dir or config.graph_rag_knowledge_base_dir
            self.enable_word_aligner = enable_word_aligner if enable_word_aligner is not None else config.graph_rag_enable_word_aligner
            self.fuzzy_threshold = fuzzy_threshold if fuzzy_threshold is not None else config.graph_rag_fuzzy_threshold
        else:
            self.embedding_model = embedding_model or "qwen3-embedding:0.6b"
            self.persist_dir = persist_dir or "./data/graphrag"
            self.knowledge_base_dir = knowledge_base_dir or "./knowledge_base"
            self.enable_word_aligner = enable_word_aligner if enable_word_aligner is not None else True
            self.fuzzy_threshold = fuzzy_threshold if fuzzy_threshold is not None else 0.75
            
        self.enable_auto_retrieval = enable_auto_retrieval
        self._indexing_thread = None
        self._indexing_status = {"running": False, "total": 0, "completed": 0, "failed": 0}

        # 初始化服务
        self._init_services()

        # 注意：自动索引已禁用，使用 /index 命令手动索引文档
        # self._start_background_indexing()

    def _init_services(self):
        """初始化 GraphRAG 服务"""
        try:
            # 创建服务实例
            self.graph_service = GraphRAGService(
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_model
            )
            # 使用混合实体抽取器（支持词典+LLM）
            self.entity_extractor = HybridEntityExtractor(persist_dir=self.persist_dir)
            # 初始化LLM客户端
            self.llm_client = SimpleLLMClient()
            self.use_llm = self.llm_client.is_available()
            # 根据 embedding 模型动态选择分块大小
            try:
                self.document_processor = DocumentProcessor.from_model_name(self.embedding_model)
            except Exception:
                self.document_processor = DocumentProcessor()

            # 创建 DocumentManager 实例（用于管理所有文档）
            from skills.graphrag.document_manager import DocumentManager
            self.document_manager = DocumentManager(
                base_persist_dir=self.persist_dir,
                embedding_model=self.embedding_model
            )

            # 初始化工具服务（供tools.py使用）
            init_graphrag_services(
                self.graph_service,
                self.entity_extractor,
                kb_dir="./data/knowledge_base",
                temp_dir="./data/temp_documents",
                doc_manager=self.document_manager,
                model_name=self.model_name or self.embedding_model
            )

            # 创建中间件（如果启用自动检索）
            self.middleware: Optional[AgentMiddleware] = None
            if self.enable_auto_retrieval:
                self.middleware = GraphRAGMiddleware(
                    graph_service=self.graph_service,
                    entity_extractor=self.entity_extractor,
                    n_results=5,
                    max_hops=2,
                    verbose=True
                )

            print(f"🧠 GraphRAG: Embedding模型={self.embedding_model or '默认'}")

        except Exception as e:
            print(f"⚠️ GraphRAG Skill 初始化失败: {e}")
            self.graph_service = None
            self.middleware = None
            self.kb_manager = None

    def _start_background_indexing(self):
        """启动后台索引线程"""
        if not self.graph_service:
            return

        kb_path = Path(self.knowledge_base_dir)
        if not kb_path.exists():
            print(f"💡 提示: 创建 {self.knowledge_base_dir}/ 目录并放入文档，启动时会自动索引")
            return

        # 创建知识库管理器（集成 WordAligner）
        self.kb_manager = KnowledgeBaseManager(
            graph_service=self.graph_service,
            entity_extractor=self.entity_extractor,
            enable_word_aligner=self.enable_word_aligner,
            fuzzy_threshold=self.fuzzy_threshold,
            embedding_model=self.embedding_model
        )

        # 检查是否有新文档需要索引
        files = self.kb_manager._scan_directory(kb_path)
        indexed_docs = self.kb_manager._get_indexed_documents()

        new_files = []
        for file_path in files:
            doc_id = self.kb_manager._generate_doc_id(file_path)
            if doc_id not in indexed_docs:
                new_files.append(file_path)

        if not new_files:
            print(f"📊 知识库已是最新: {len(files)} 个文档已索引")
            return

        # 有未索引的文档，启动后台线程
        self._indexing_status["running"] = True
        self._indexing_status["total"] = len(new_files)
        self._indexing_status["completed"] = 0
        self._indexing_status["failed"] = 0

        print(f"📚 发现 {len(new_files)} 个新文档，启动后台索引...")
        print(f"   可以立即开始对话，索引将在后台完成")

        self._indexing_thread = threading.Thread(
            target=self._background_index_worker,
            args=(new_files,),
            daemon=True
        )
        self._indexing_thread.start()

        # 启动进度监控线程
        self._progress_thread = threading.Thread(
            target=self._monitor_indexing_progress,
            daemon=True
        )
        self._progress_thread.start()

    def _monitor_indexing_progress(self):
        """监控索引进度并定期输出"""
        import time

        # 等待几秒让索引开始
        time.sleep(3)

        while self._indexing_status["running"]:
            total = self._indexing_status["total"]
            completed = self._indexing_status["completed"]
            failed = self._indexing_status["failed"]

            if total > 0:
                progress = (completed + failed) / total * 100
                remaining = total - completed - failed

                if remaining > 0:
                    print(f"\n📊 [后台索引] 进度: {completed}/{total} ({progress:.0f}%), 剩余 {remaining} 个文档\n你: ", end="", flush=True)

            # 每 10 秒更新一次
            time.sleep(10)

    def _background_index_worker(self, files_to_index):
        """后台索引工作线程（低优先级，不阻塞主线程）"""
        try:
            # 降低线程优先级（Windows）
            try:
                import sys
                if sys.platform == 'win32':
                    import ctypes
                    ctypes.windll.kernel32.SetThreadPriority(
                        ctypes.windll.kernel32.GetCurrentThread(),
                        1  # THREAD_PRIORITY_LOWEST
                    )
            except Exception:
                pass

            for file_path in files_to_index:
                if not self._indexing_status["running"]:
                    break

                doc_id = self.kb_manager._generate_doc_id(file_path)
                try:
                    # 使用更温和的方式索引，每处理一个块后暂停
                    self.kb_manager._index_document(file_path, doc_id, verbose=False)
                    self._indexing_status["completed"] += 1
                    progress = (self._indexing_status["completed"] + self._indexing_status["failed"]) / len(files_to_index) * 100
                    print(f"\n  ✅ [后台 {progress:.0f}%] 已索引: {file_path.name}")
                    print("你: ", end="", flush=True)
                except Exception as e:
                    self._indexing_status["failed"] += 1
                    print(f"\n  ❌ [后台] 索引失败: {file_path.name} - {e}")
                    print("你: ", end="", flush=True)

                # 每个文档之间暂停，让出时间给主线程
                time.sleep(0.5)

            total = self._indexing_status['total']
            completed = self._indexing_status['completed']
            failed = self._indexing_status['failed']
            print(f"\n📚 后台索引完成: {completed} 成功, {failed} 失败, 总计 {total} 个文档")
            print("你: ", end="", flush=True)

        except Exception as e:
            print(f"\n⚠️ 后台索引线程出错: {e}")
            print("你: ", end="", flush=True)
        finally:
            self._indexing_status["running"] = False

    def get_indexing_status(self) -> dict:
        """获取索引状态"""
        return self._indexing_status.copy()

    def _auto_index_knowledge_base(self):
        """自动扫描并索引知识库（同步版本，已废弃）"""
        # 此方法已废弃，使用 _start_background_indexing 替代
        pass

    def get_tools(self) -> List[BaseTool]:
        """
        返回 GraphRAG 工具列表

        Returns:
            工具列表
        """
        return [
            upload_document,  # 上传文档到知识库（推荐）
            add_document,     # 添加文档到知识图谱（不复制到知识库）
            add_text,
            analyze_document,      # 实时分析文档（一次性查询）
            add_text_and_search,   # 实时模式：添加文本并检索
            query_graph_stats,
            search_all_documents,  # 全局搜索所有文档
            search_in_category,    # 在指定分类中搜索
            list_knowledge_categories,  # 列出所有分类
            list_entities,
            get_entity_relations,
        ]

    def get_system_prompt(self) -> Optional[str]:
        """
        返回系统提示词

        Returns:
            系统提示词
        """
        return """你是知识图谱问答专家，专门负责管理知识库和回答基于知识库的问题。

## 可用工具（只能使用以下工具）

### 知识库管理工具（长期保存）
1. **upload_document(file_path)** - 上传文档到知识库
   - 自动复制文件到 knowledge_base/ 目录
   - 自动分块、抽取实体、建立完整索引
   - 适用于：用户明确要求"添加到知识库"、"建立索引"
   - 所有文档（无论长短）都会建立完整索引

### 实时分析工具（一次性查询）
2. **analyze_document(file_path, query=None)** - 实时分析文档内容
   - 短文档（<阈值）：直接返回全文
   - 中长文档（>=阈值）：建立临时索引并检索
   - 适用于：用户说"分析一下这个文档"、"总结文档内容"
   - 不长期保存，临时处理

### 其他工具
3. **add_document(file_path)** - 添加文档到知识图谱（不复制到知识库）
4. **add_text(text, source)** - 添加文本片段
5. **search_all_documents(query)** - 全局搜索所有文档
6. **search_in_category(category, query)** - 在指定分类中搜索
7. **list_knowledge_categories()** - 列出所有知识库分类
8. **query_graph_stats()** - 查看图谱统计
9. **list_entities(entity_type)** - 列出实体
10. **get_entity_relations(entity_name)** - 查看实体关系

## 核心规则

**根据用户意图选择正确的工具：**

1. **用户说"分析/总结/查看文档"** → 使用 `analyze_document`
   ```
   用户：分析一下 D:\文档\报告.pdf
   → 调用：analyze_document("D:\\文档\\报告.pdf")
   ```

2. **用户说"添加到知识库/建立索引"** → 使用 `upload_document`
   ```
   用户：把 D:\文档\伤寒论.pdf 添加到知识库
   → 调用：upload_document("D:\\文档\\伤寒论.pdf")
   ```

正确示例：
```
用户：D:\文档\伤寒论.pdf
→ 立即调用：upload_document("D:\\文档\\伤寒论.pdf")
→ 等待结果
→ 报告：已保存到知识库，索引完成
```

错误示例：
```
用户：D:\文档\伤寒论.pdf
→ 不要列出目录
→ 不要搜索文件
→ 不要执行 shell 命令
→ 直接调用 upload_document
```

## 工作流程

1. **用户上传文档**：立即使用 upload_document
2. **用户提问**：系统会自动检索知识库并回答
3. **用户探索知识库**：使用 list_entities 或 get_entity_relations

## 注意事项

- 文档添加后会自动分块、抽取实体、建立索引
- 实体抽取使用轻量级规则+词典匹配
- **集成 WordAligner**: 实体精确映射到原文位置，支持溯源验证
- 关系推断基于实体共现
- 支持多跳推理（1-2跳）

## 实体对齐质量说明

系统使用 WordAligner 算法将提取的实体精确对齐到原文：
- ✓ 精确匹配: 实体文本与原文完全一致
- ~ 部分匹配: 实体文本与原文略有差异（如标点）
- ≈ 模糊匹配: 通过相似度算法找到的匹配（相似度≥0.75）
- 对齐结果可用于验证提取准确性，提升知识库可信度
"""

    def get_middleware(self) -> Optional[AgentMiddleware]:
        """
        返回 GraphRAG 中间件（用于自动检索增强）

        Returns:
            AgentMiddleware 实例或 None
        """
        return self.middleware

    def on_activate(self) -> None:
        """Skill 被激活时调用"""
        if self.graph_service:
            stats = self.graph_service.get_stats()
            print(f"📊 GraphRAG 状态: {stats['total_documents']} 文档, "
                  f"{stats['total_entities']} 实体, {stats['total_relations']} 关系")
            
            # 显示 WordAligner 对齐统计
            if self.enable_word_aligner and self.kb_manager:
                alignment_stats = self.kb_manager.get_alignment_stats()
                if alignment_stats['total_entities'] > 0:
                    print(f"   🎯 对齐质量: {alignment_stats['exact_match_pct']} 精确匹配, "
                          f"{alignment_stats['fuzzy_match_pct']} 模糊匹配")

    def on_deactivate(self) -> None:
        """Skill 被停用时调用"""
        pass

    def get_stats(self) -> dict:
        """
        获取 GraphRAG 统计信息

        Returns:
            统计信息字典
        """
        if self.graph_service:
            return self.graph_service.get_stats()
        return {"total_documents": 0, "total_entities": 0, "total_relations": 0}

    # ========== Context 集成（新增）==========

    def to_context(self) -> "SkillContext":
        """
        将 GraphRAG Skill 转换为 Context 片段

        包含知识库描述和统计信息。

        Returns:
            SkillContext
        """
        from ollamapilot.context.types import SkillContext, ToolDefinition, Example

        # 获取知识库概览
        kb_summary = self._get_kb_summary()

        return SkillContext(
            tool_definitions=self.get_tool_definitions(),
            system_prompt=self.get_system_prompt(),
            examples=self._get_search_examples(),
            knowledge=kb_summary,
        )

    def _get_kb_summary(self) -> str:
        """获取知识库概览"""
        if not self.graph_service:
            return "知识库服务未初始化"

        stats = self.graph_service.get_stats()
        return f"""知识库概览:
- 文档数: {stats.get('total_documents', 0)}
- 实体数: {stats.get('total_entities', 0)}
- 关系数: {stats.get('total_relations', 0)}
- 支持模式: 预索引模式 + 实时模式
"""

    def _get_search_examples(self) -> List["Example"]:
        """获取搜索示例"""
        from ollamapilot.context.types import Example

        return [
            Example(
                input="帮我分析这个PDF并回答其中的问题",
                output="使用 add_and_search(file_path='path/to/file.pdf', query='问题内容')",
                description="实时模式：即时添加文档并检索"
            ),
            Example(
                input="搜索伤寒论中关于太阳病的内容",
                output="使用 search_in_category(category='伤寒论', query='太阳病')",
                description="在指定分类中搜索"
            ),
        ]

    # ========== 实时模式（新增）==========

    def add_and_search(self, file_path: str, query: str, n_results: int = 5) -> List[dict]:
        """
        实时模式：即时添加文档并检索

        无需提前索引，即用即走：
        1. 实时处理文档
        2. 临时索引（内存中）
        3. 立即检索
        4. 清理临时索引

        Args:
            file_path: 文档路径
            query: 查询内容
            n_results: 返回结果数量

        Returns:
            检索结果列表
        """
        from pathlib import Path
        import tempfile

        source_path = Path(file_path)
        if not source_path.exists():
            return [{"error": f"文件不存在: {file_path}"}]

        if not self.graph_service or not self.entity_extractor:
            return [{"error": "服务未初始化"}]

        try:
            # 1. 实时处理文档
            processor = self.document_processor
            text = processor.read_document(file_path)

            if not text:
                return [{"error": f"无法读取文档内容: {file_path}"}]

            chunks = processor.chunk_text(text)

            # 2. 临时索引（内存中）
            temp_docs = []
            for i, chunk in enumerate(chunks):
                try:
                    entities, _ = self.entity_extractor.extract(
                        chunk,
                        use_llm=self.use_llm if hasattr(self, 'use_llm') else False,
                        llm_client=self.llm_client if hasattr(self, 'llm_client') else None,
                        top_k=20
                    )
                    temp_docs.append({
                        "content": chunk,
                        "index": i,
                        "entities": entities
                    })
                except Exception:
                    continue

            # 3. 立即检索
            query_entities = self.entity_extractor.extract_from_query(query)
            query_entity_names = {e['name'] for e in query_entities}

            results = []
            for doc in temp_docs:
                doc_entity_names = {e.name for e in doc['entities']}
                overlap = query_entity_names & doc_entity_names
                if overlap:
                    results.append({
                        "content": doc['content'],
                        "score": len(overlap) / len(query_entity_names) if query_entity_names else 0,
                        "source": f"{source_path.name} (块 {doc['index']})",
                        "match_type": "entity"
                    })

            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)

            # 4. 返回结果（临时索引自动清理）
            return results[:n_results]

        except Exception as e:
            return [{"error": f"实时检索失败: {str(e)}"}]

    def add_text_and_search(self, text: str, query: str, n_results: int = 5) -> List[dict]:
        """
        实时模式：直接粘贴文本并检索

        Args:
            text: 文本内容
            query: 查询内容
            n_results: 返回结果数量

        Returns:
            检索结果列表
        """
        if not self.graph_service or not self.entity_extractor:
            return [{"error": "服务未初始化"}]

        try:
            # 1. 实时处理文本
            processor = self.document_processor
            chunks = processor.chunk_text(text)

            # 2. 临时索引
            temp_docs = []
            for i, chunk in enumerate(chunks):
                try:
                    entities, _ = self.entity_extractor.extract(
                        chunk,
                        use_llm=self.use_llm if hasattr(self, 'use_llm') else False,
                        llm_client=self.llm_client if hasattr(self, 'llm_client') else None,
                        top_k=20
                    )
                    temp_docs.append({
                        "content": chunk,
                        "index": i,
                        "entities": entities
                    })
                except Exception:
                    continue

            # 3. 立即检索
            query_entities = self.entity_extractor.extract_from_query(query)
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

            return results[:n_results]

        except Exception as e:
            return [{"error": f"实时文本检索失败: {str(e)}"}]
