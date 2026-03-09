"""
GraphRAG Skill - 知识图谱检索增强

基于实体-关系的智能文档问答 Skill。
作为独立的 Python Skill，遵循 USB 即插即用设计理念。
"""

from typing import List, Optional, Any
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware
import threading
import time

from ollamapilot.skills.base import Skill

# GraphRAG Skill 内部模块
from skills.graphrag.services import (
    GraphRAGService,
    OntologyGenerator,
    LightweightEntityExtractor,
    Entity
)
from skills.graphrag.middleware import GraphRAGMiddleware
from skills.graphrag.utils import DocumentProcessor

from skills.graphrag.tools import (
    upload_document,
    add_document,
    add_text,
    generate_ontology,
    query_graph_stats,
    search_knowledge,
    list_entities,
    get_entity_relations,
    init_graphrag_services,
)
from skills.graphrag.knowledge_base import KnowledgeBaseManager


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
        search_knowledge(query="张三在哪里工作？")

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
        persist_dir: str = "./data/graphrag",
        enable_auto_retrieval: bool = True,
        knowledge_base_dir: str = "./knowledge_base"
    ):
        """
        初始化 GraphRAG Skill

        Args:
            embedding_model: Embedding模型名称（如 "qwen3-embedding:4b"）
            persist_dir: 数据持久化目录
            enable_auto_retrieval: 是否启用自动检索中间件
            knowledge_base_dir: 知识库目录路径
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.enable_auto_retrieval = enable_auto_retrieval
        self.knowledge_base_dir = knowledge_base_dir
        self._indexing_thread = None
        self._indexing_status = {"running": False, "total": 0, "completed": 0, "failed": 0}

        # 初始化服务
        self._init_services()

        # 自动扫描知识库（后台异步）
        self._start_background_indexing()

    def _init_services(self):
        """初始化 GraphRAG 服务"""
        try:
            # 创建服务实例
            self.graph_service = GraphRAGService(
                persist_dir=self.persist_dir,
                embedding_model=self.embedding_model
            )
            self.entity_extractor = LightweightEntityExtractor()
            self.ontology_generator = OntologyGenerator(None)  # 暂时不传入LLM
            self.document_processor = DocumentProcessor()

            # 初始化工具服务（供tools.py使用）
            init_graphrag_services(
                self.graph_service,
                self.ontology_generator,
                self.entity_extractor
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

            print(f"🧠 GraphRAG Skill 已加载 (Embedding: {self.embedding_model or '默认'})")

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

        # 创建知识库管理器
        self.kb_manager = KnowledgeBaseManager(
            graph_service=self.graph_service,
            entity_extractor=self.entity_extractor,
            document_processor=self.document_processor
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
            generate_ontology,
            query_graph_stats,
            search_knowledge,
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

1. **upload_document(file_path)** - 上传文档到知识库（最常用）
   - 自动复制文件到 knowledge_base/ 目录
   - 自动分块、抽取实体、建立索引
   - 当用户提供文件路径时，必须立即使用此工具

2. **add_document(file_path)** - 添加文档到知识图谱（不复制到知识库）
   - 仅建立索引，不保存到知识库目录
   - 适用于临时文档

3. **add_text(text, source)** - 添加文本片段
4. **search_knowledge(query)** - 搜索知识库
5. **query_graph_stats()** - 查看图谱统计
6. **list_entities(entity_type)** - 列出实体
7. **get_entity_relations(entity_name)** - 查看实体关系
8. **generate_ontology(document_text)** - 生成本体定义

## 核心规则

**当用户提供文件路径时，必须立即调用 upload_document(file_path)，不要执行其他操作。**

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
- 关系推断基于实体共现
- 支持多跳推理（1-2跳）
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
