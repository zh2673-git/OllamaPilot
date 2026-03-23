"""
文档管理器 - 支持按模型和文档名隔离存储

提供：
1. 按模型隔离的向量存储（如 伤寒论_qwen3-embedding_0.6b）
2. 手动控制向量化（非自动）
3. 静默后台处理，进度显示（50%和完成）
4. 文档级别的独立管理
"""

from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import os
import time
import threading
from dataclasses import dataclass
from enum import Enum


class IndexingStatus(Enum):
    """索引状态"""
    PENDING = "pending"      # 等待处理
    RUNNING = "running"      # 正在处理
    COMPLETED = "completed"  # 完成
    FAILED = "failed"        # 失败


@dataclass
class DocumentInfo:
    """文档信息"""
    doc_id: str
    name: str
    file_path: str
    status: IndexingStatus
    model_name: Optional[str] = None
    chunks_count: int = 0
    entities_count: int = 0
    progress: float = 0.0
    message: str = ""
    created_at: float = 0.0
    started_at: Optional[float] = None  # 索引开始时间
    completed_at: Optional[float] = None


class DocumentManager:
    """
    文档管理器
    
    管理文档的添加、索引和查询，支持：
    - 按模型隔离存储
    - 手动控制向量化
    - 后台静默处理
    """
    
    def __init__(
        self,
        base_persist_dir: str = "./data/graphrag",
        embedding_model: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        batch_size: int = 20,
        enable_relation_vector: bool = True,
        enable_dual_retrieval: bool = True,
        use_llm_merge: bool = False
    ):
        """
        初始化文档管理器

        Args:
            base_persist_dir: 基础持久化目录
            embedding_model: Embedding模型名称
            progress_callback: 进度回调函数(doc_id, progress, message)
            batch_size: 批量处理大小（默认20，可配置）
            enable_relation_vector: 是否启用关系向量化（默认True，LightRAG增强）
            enable_dual_retrieval: 是否启用双层检索（默认True，LightRAG增强）
            use_llm_merge: 是否使用LLM智能合并（默认False，小模型建议关闭）
        """
        self.base_persist_dir = Path(base_persist_dir)
        self.base_persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.progress_callback = progress_callback
        self.batch_size = batch_size
        self.enable_relation_vector = enable_relation_vector
        self.enable_dual_retrieval = enable_dual_retrieval
        self.use_llm_merge = use_llm_merge

        # 文档信息存储
        self.documents: Dict[str, DocumentInfo] = {}

        # 后台任务
        self._indexing_tasks: Dict[str, threading.Thread] = {}

        # 缓存 GraphRAGService 实例（避免重复创建）
        self._graph_service_cache: Dict[str, Any] = {}

        # 加载已有文档信息
        self._load_document_registry()

        # 恢复卡住的文档状态
        self._recover_stuck_documents()

    def _recover_stuck_documents(self):
        """
        恢复卡住的文档状态

        当程序异常退出时，可能有文档状态卡在 RUNNING
        此方法将这些文档状态重置为 FAILED，允许重新索引
        """
        recovered_count = 0
        for doc_id, doc_info in self.documents.items():
            if doc_info.status == IndexingStatus.RUNNING:
                # 检查是否有对应的活跃线程
                if doc_id not in self._indexing_tasks:
                    # 没有活跃线程，说明是上次异常退出留下的
                    doc_info.status = IndexingStatus.FAILED
                    doc_info.message = "上次索引中断，可尝试断点续传"
                    recovered_count += 1
                    print(f"🔄 恢复卡住文档: {doc_info.name}")

        if recovered_count > 0:
            self._save_document_registry()
            print(f"✅ 已恢复 {recovered_count} 个卡住的文档")

    def _get_model_safe_name(self, model_name: Optional[str]) -> str:
        """获取模型安全名称（用于文件名）"""
        if not model_name:
            return "default"
        # 清理模型名称，使其适合作为文件名
        return "".join(c if c.isalnum() else "_" for c in model_name)
    
    def _get_document_storage_path(self, doc_name: str, model_name: Optional[str] = None) -> Path:
        """
        获取文档存储路径
        
        格式: {base_persist_dir}/{doc_name}_{model_safe_name}/
        例如: ./data/graphrag/伤寒论_qwen3-embedding_0.6b/
        """
        model_safe = self._get_model_safe_name(model_name or self.embedding_model)
        # 清理文档名
        doc_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in doc_name)
        storage_name = f"{doc_safe}_{model_safe}"
        return self.base_persist_dir / storage_name
    
    def register_document(
        self,
        doc_name: str,
        file_path: str,
        auto_index: bool = False
    ) -> str:
        """
        注册文档
        
        Args:
            doc_name: 文档名称（如"伤寒论"）
            file_path: 文件路径
            auto_index: 是否自动索引（默认False，手动控制）
            
        Returns:
            文档ID
        """
        import hashlib
        
        doc_id = hashlib.md5(f"{doc_name}_{file_path}".encode()).hexdigest()[:16]
        
        # 检查是否已存在
        if doc_id in self.documents:
            return doc_id
        
        # 创建文档信息
        doc_info = DocumentInfo(
            doc_id=doc_id,
            name=doc_name,
            file_path=file_path,
            status=IndexingStatus.PENDING,
            model_name=self.embedding_model,
            created_at=time.time()
        )
        
        self.documents[doc_id] = doc_info
        self._save_document_registry()
        
        print(f"📄 文档已注册: {doc_name}")
        print(f"   ID: {doc_id}")
        print(f"   存储路径: {self._get_document_storage_path(doc_name, self.embedding_model)}")
        
        if auto_index:
            self.start_indexing(doc_id)
        else:
            print(f"   状态: 等待手动索引（使用 index_document 命令）")
        
        return doc_id
    
    def start_indexing(self, doc_id: str, silent: bool = True, resume: bool = True) -> bool:
        """
        开始索引文档（后台静默处理）
        
        Args:
            doc_id: 文档ID
            silent: 是否静默模式（只显示50%和100%进度）
            resume: 是否支持断点续传（默认True）
            
        Returns:
            是否成功启动
        """
        if doc_id not in self.documents:
            print(f"❌ 文档不存在: {doc_id}")
            return False
        
        doc_info = self.documents[doc_id]
        
        if doc_info.status == IndexingStatus.RUNNING:
            print(f"⏳ 文档正在索引中: {doc_info.name}")
            return False
        
        if doc_info.status == IndexingStatus.COMPLETED:
            print(f"✅ 文档已索引: {doc_info.name}")
            return False
        
        # 检查是否支持断点续传
        if resume and doc_info.status == IndexingStatus.FAILED:
            print(f"🔄 检测到上次索引失败，尝试断点续传: {doc_info.name}")
            doc_info.status = IndexingStatus.PENDING
            doc_info.progress = 0.0
            doc_info.message = "准备断点续传..."
            self._save_document_registry()
        
        # 启动后台线程
        doc_info.status = IndexingStatus.RUNNING
        doc_info.progress = 0.0
        doc_info.message = "开始索引..."
        doc_info.started_at = time.time()  # 记录开始时间
        
        thread = threading.Thread(
            target=self._index_document_worker,
            args=(doc_id, silent),
            daemon=True
        )
        self._indexing_tasks[doc_id] = thread
        thread.start()
        
        if silent:
            print(f"🔄 开始索引: {doc_info.name}（后台静默处理）")
        
        return True
    
    def _index_document_worker(self, doc_id: str, silent: bool):
        """文档索引工作线程"""
        doc_info = self.documents[doc_id]
        
        try:
            # 更新进度回调
            def update_progress(progress: float, message: str):
                doc_info.progress = progress
                doc_info.message = message
                
                # 静默模式下只显示50%和100%
                if silent:
                    if progress >= 0.5 and doc_info.progress < 0.5:
                        print(f"⏳ {doc_info.name}: 50% 完成")
                    elif progress >= 1.0:
                        print(f"✅ {doc_info.name}: 索引完成")
                else:
                    print(f"  [{progress*100:.0f}%] {message}")
                
                if self.progress_callback:
                    self.progress_callback(doc_id, progress, message)
            
            # 执行索引
            self._do_index_document(doc_id, update_progress)
            
            # 完成
            doc_info.status = IndexingStatus.COMPLETED
            doc_info.progress = 1.0
            doc_info.completed_at = time.time()
            doc_info.message = "索引完成"
            
            if silent:
                print(f"✅ {doc_info.name}: 索引完成")
            
        except Exception as e:
            doc_info.status = IndexingStatus.FAILED
            doc_info.message = f"索引失败: {str(e)}"
            print(f"❌ {doc_info.name}: 索引失败 - {e}")
        
        finally:
            self._save_document_registry()
            if doc_id in self._indexing_tasks:
                del self._indexing_tasks[doc_id]
    
    def _do_index_document(self, doc_id: str, progress_callback: Callable[[float, str], None]):
        """执行文档索引（实际工作）"""
        from skills.graphrag.utils import DocumentProcessor
        from skills.graphrag.services import GraphRAGService, HybridEntityExtractor
        from skills.graphrag.llm_client import SimpleLLMClient
        from skills.graphrag.knowledge_base import KnowledgeBaseManager
        import logging

        logger = logging.getLogger(__name__)
        doc_info = self.documents[doc_id]

        # 获取存储路径
        storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
        storage_path.mkdir(parents=True, exist_ok=True)

        progress_callback(0.05, "准备索引...")
        logger.info(f"[{doc_info.name}] 开始索引，文档路径: {doc_info.file_path}")

        try:
            progress_callback(0.1, "初始化服务...")
            graph_service = GraphRAGService(
                persist_dir=str(storage_path),
                embedding_model=doc_info.model_name,
                enable_relation_vector=self.enable_relation_vector,
                enable_dual_retrieval=self.enable_dual_retrieval,
                use_llm_merge=self.use_llm_merge
            )

            # 初始化混合实体抽取器（加载全局预设词典 + 文档私有词典）
            entity_extractor = HybridEntityExtractor(
                persist_dir=str(storage_path),
                doc_id=doc_id
            )

            # 初始化LLM客户端（用于动态学习）
            llm_client = SimpleLLMClient(model_name="qwen3.5:4b")
            use_llm = llm_client.is_available()
            if use_llm:
                logger.info(f"[{doc_info.name}] LLM服务可用，将使用混合模式抽取")
            else:
                logger.info(f"[{doc_info.name}] LLM服务不可用，仅使用词典匹配")

            # 根据 embedding 模型动态选择分块大小
            try:
                doc_processor = DocumentProcessor.from_model_name(doc_info.model_name)
                logger.info(f"[{doc_info.name}] 分块大小: {doc_processor.chunk_size:,} 字符 (基于 {doc_info.model_name})")
            except Exception as e:
                logger.warning(f"[{doc_info.name}] 动态分块失败，使用默认: {e}")
                doc_processor = DocumentProcessor()

            progress_callback(0.15, "读取文档...")

            # 读取文档
            text = doc_processor.read_document(doc_info.file_path)
            if not text:
                raise ValueError("无法读取文档内容")

            progress_callback(0.25, "分块处理...")

            # 分块
            chunks = doc_processor.chunk_text(text)
            doc_info.chunks_count = len(chunks)

            progress_callback(0.3, f"分块完成: {len(chunks)} 块")
            logger.info(f"[{doc_info.name}] 文档分块完成: {len(chunks)} 块")

            # 处理每个块（使用批量抽取优化）
            progress_callback(0.35, "开始向量化和实体抽取...")
            logger.info(f"[{doc_info.name}] 开始处理 {len(chunks)} 个块（批量模式，每批5个）")

            total_entities = 0
            total_relations = 0

            # 使用批量抽取（可配置批次大小，默认10）
            batch_size = self.batch_size
            batch_results = []

            # 定义批量进度回调
            def batch_progress_callback(batch_idx, total_batches, batch_start, total_chunks, total_entities_so_far=0):
                # 计算实际处理的块数
                current_chunk = min(batch_start + batch_size, total_chunks)
                chunk_progress = 0.35 + (0.6 * current_chunk / total_chunks)
                progress_callback(chunk_progress, f"处理块 {current_chunk}/{total_chunks} (批次 {batch_idx}/{total_batches})...")
                # 实时更新实体数
                doc_info.entities_count = total_entities_so_far
                # 保存注册表以更新进度
                self._save_document_registry()

            # 批量抽取，带进度回调
            batch_results = entity_extractor.extract_batch(
                chunks,
                use_llm=use_llm,
                llm_client=llm_client,
                batch_size=batch_size,
                top_k=20,
                progress_callback=batch_progress_callback
            )

            # 处理批量结果
            for i, (entities, relations) in enumerate(batch_results):
                total_entities += len(entities)
                total_relations += len(relations)

                # 实时更新实体数（让用户可以通过 /docs 查看进度）
                doc_info.entities_count = total_entities

                # 添加到图谱
                chunk_doc_id = f"{doc_id}_{i}"
                from skills.graphrag.services import Entity
                entity_objects = [
                    Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                    for e in entities
                ]

                # 将实体和关系信息添加到metadata
                metadata = {
                    "source": doc_info.file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "entities": ",".join([e.name for e in entities[:10]]),
                    "relations": ",".join([f"{r.source}-{r.relation}-{r.target}" for r in relations[:5]])
                }

                graph_service.add_document(
                    text=chunks[i],
                    doc_id=chunk_doc_id,
                    metadata=metadata,
                    entities=entity_objects
                )

                # 每5块保存一次
                if (i + 1) % 5 == 0:
                    graph_service._save_index()

            doc_info.entities_count = total_entities
            progress_callback(1.0, f"索引完成（{total_entities}实体，{total_relations}关系）")
            logger.info(f"[{doc_info.name}] 索引完成: {total_entities} 个实体, {total_relations} 个关系")

            # 保存词典统计信息
            stats = entity_extractor.get_statistics()
            logger.info(f"[{doc_info.name}] 词典统计: {stats}")

            # 清除缓存，确保搜索服务能加载新索引
            cache_key = f"{doc_id}_{doc_info.model_name}"
            if cache_key in self._graph_service_cache:
                del self._graph_service_cache[cache_key]
                logger.info(f"[{doc_info.name}] 已清除搜索缓存，新索引可立即搜索")

        except Exception as e:
            logger.error(f"[{doc_info.name}] 索引错误: {e}")
            progress_callback(0.0, f"索引错误: {e}")
            # 不重新抛出异常，让上层处理状态更新
    
    def get_document_status(self, doc_id: str) -> Optional[DocumentInfo]:
        """获取文档状态"""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[DocumentInfo]:
        """列出所有文档"""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id not in self.documents:
            return False
        
        doc_info = self.documents[doc_id]
        
        # 删除存储目录
        storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
        
        # 删除记录
        del self.documents[doc_id]
        self._save_document_registry()
        
        print(f"🗑️  文档已删除: {doc_info.name}")
        return True
    
    def _load_document_registry(self):
        """加载文档注册表"""
        import json
        
        registry_path = self.base_persist_dir / "document_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for doc_id, doc_data in data.items():
                    self.documents[doc_id] = DocumentInfo(
                        doc_id=doc_data["doc_id"],
                        name=doc_data["name"],
                        file_path=doc_data["file_path"],
                        status=IndexingStatus(doc_data["status"]),
                        model_name=doc_data.get("model_name"),
                        chunks_count=doc_data.get("chunks_count", 0),
                        entities_count=doc_data.get("entities_count", 0),
                        progress=doc_data.get("progress", 0.0),
                        message=doc_data.get("message", ""),
                        created_at=doc_data.get("created_at", 0.0),
                        started_at=doc_data.get("started_at"),
                        completed_at=doc_data.get("completed_at")
                    )
            except Exception as e:
                print(f"⚠️ 加载文档注册表失败: {e}")
    
    def _save_document_registry(self):
        """保存文档注册表"""
        import json
        
        registry_path = self.base_persist_dir / "document_registry.json"
        
        data = {}
        for doc_id, doc_info in self.documents.items():
            data[doc_id] = {
                "doc_id": doc_info.doc_id,
                "name": doc_info.name,
                "file_path": doc_info.file_path,
                "status": doc_info.status.value,
                "model_name": doc_info.model_name,
                "chunks_count": doc_info.chunks_count,
                "entities_count": doc_info.entities_count,
                "progress": doc_info.progress,
                "message": doc_info.message,
                "created_at": doc_info.created_at,
                "started_at": doc_info.started_at,
                "completed_at": doc_info.completed_at
            }
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def wait_for_indexing(self, doc_id: str, timeout: float = 300.0) -> bool:
        """
        等待索引完成
        
        Args:
            doc_id: 文档ID
            timeout: 超时时间（秒）
            
        Returns:
            是否成功完成
        """
        start_time = time.time()
        
        while doc_id in self._indexing_tasks:
            if time.time() - start_time > timeout:
                print(f"⏰ 等待索引超时: {doc_id}")
                return False
            time.sleep(0.5)
        
        doc_info = self.documents.get(doc_id)
        if doc_info:
            return doc_info.status == IndexingStatus.COMPLETED
        return False
    
    def resume_failed_indexing(self) -> List[str]:
        """
        恢复所有失败的索引任务
        
        Returns:
            恢复的文档ID列表
        """
        failed_docs = [
            doc_id for doc_id, doc_info in self.documents.items()
            if doc_info.status == IndexingStatus.FAILED
        ]
        
        if not failed_docs:
            print("✅ 没有需要恢复的失败任务")
            return []
        
        print(f"🔄 发现 {len(failed_docs)} 个失败的索引任务，开始恢复...")
        
        resumed = []
        for doc_id in failed_docs:
            if self.start_indexing(doc_id, silent=True, resume=True):
                resumed.append(doc_id)
        
        print(f"✅ 已恢复 {len(resumed)}/{len(failed_docs)} 个任务")
        return resumed

    def _get_cached_graph_service(self, doc_id: str, doc_info: DocumentInfo):
        """获取缓存的 GraphRAGService 实例"""
        from skills.graphrag.services import GraphRAGService

        cache_key = f"{doc_id}_{doc_info.model_name}"

        if cache_key not in self._graph_service_cache:
            storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
            self._graph_service_cache[cache_key] = GraphRAGService(
                persist_dir=str(storage_path),
                embedding_model=doc_info.model_name or self.embedding_model
            )

        return self._graph_service_cache[cache_key]

    def search_all_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        搜索所有已索引的文档（使用缓存优化）

        Args:
            query: 查询文本
            n_results: 返回结果数量

        Returns:
            搜索结果列表
        """
        all_results = []

        for doc_id, doc_info in self.documents.items():
            if doc_info.status != IndexingStatus.COMPLETED:
                continue

            try:
                # 使用缓存的 GraphRAGService
                graph_service = self._get_cached_graph_service(doc_id, doc_info)
                results = graph_service.vector_search(query, n_results=n_results)

                # 添加文档信息到结果
                for result in results:
                    result['document_name'] = doc_info.name
                    result['document_id'] = doc_id

                all_results.extend(results)

            except Exception as e:
                print(f"⚠️ 搜索文档 {doc_info.name} 失败: {e}")
                continue

        # 按相似度排序并返回前 n_results 个
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:n_results]



    def get_global_stats(self) -> Dict[str, Any]:
        """
        获取所有文档的全局统计信息

        Returns:
            全局统计信息
        """
        from skills.graphrag.services import GraphRAGService

        total_documents = 0
        total_entities = 0
        total_relations = 0
        entity_types = set()

        for doc_id, doc_info in self.documents.items():
            if doc_info.status != IndexingStatus.COMPLETED:
                continue

            try:
                storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
                graph_service = GraphRAGService(
                    persist_dir=str(storage_path),
                    embedding_model=doc_info.model_name or self.embedding_model
                )

                stats = graph_service.get_stats()
                total_documents += stats['total_documents']
                total_entities += stats['total_entities']
                total_relations += stats['total_relations']
                entity_types.update(stats['entity_types'])

            except Exception as e:
                continue

        return {
            "total_documents": total_documents,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "entity_types": list(entity_types)
        }

    def search_by_category(self, category: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        搜索指定分类的知识库
        
        扫描 data/graphrag/{category}/ 目录及其子目录下的所有文档存储，
        在这些文档中执行向量搜索。支持多级文件夹结构。
        
        Args:
            category: 分类名称（用户自定义的文件夹名），支持多级路径如"中医经典/伤寒论"
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        from skills.graphrag.services import GraphRAGService
        
        category_path = self.base_persist_dir / category
        if not category_path.exists():
            return []
        
        all_results = []
        
        # 递归扫描分类目录下的所有文档存储
        def scan_directory(dir_path: Path, relative_path: str = ""):
            """递归扫描目录，查找所有文档存储"""
            for item in dir_path.iterdir():
                if not item.is_dir():
                    continue
                
                # 构建相对路径（用于多级分类）
                current_relative = f"{relative_path}/{item.name}" if relative_path else item.name
                
                # 检查是否是文档存储目录（包含向量索引文件）
                # SimpleVectorStore 使用 index_*.json 和 vectors_*.json 文件
                is_doc_store = any(
                    f.name.startswith(('index_', 'vectors_')) and f.name.endswith('.json')
                    for f in item.iterdir() if f.is_file()
                )
                
                if is_doc_store:
                    # 这是一个文档存储目录
                    try:
                        # 从文件名中提取 embedding 模型名称
                        # 例如：index_qwen3_embedding_0_6b.json -> qwen3-embedding:0.6b
                        embedding_model = self.embedding_model
                        if not embedding_model:
                            for f in item.iterdir():
                                if f.is_file() and f.name.startswith('index_') and f.name.endswith('.json'):
                                    # 提取模型名称：index_qwen3_embedding_0_6b.json -> qwen3-embedding:0.6b
                                    model_part = f.name[6:-5]  # 去掉 'index_' 和 '.json'
                                    embedding_model = model_part.replace('_', '-').replace('--', ':')
                                    break
                        
                        graph_service = GraphRAGService(
                            persist_dir=str(item),
                            embedding_model=embedding_model
                        )
                        
                        # 执行搜索
                        results = graph_service.vector_search(query, n_results=n_results)
                        
                        # 添加分类和来源信息
                        for result in results:
                            result['category'] = category
                            result['document_path'] = current_relative  # 完整路径如"伤寒论/原文"
                            result['document_name'] = item.name  # 目录名
                            
                        all_results.extend(results)
                        
                    except Exception as e:
                        # 跳过无法加载的目录
                        continue
                else:
                    # 这是一个子分类目录，递归扫描
                    scan_directory(item, current_relative)
        
        # 开始递归扫描
        scan_directory(category_path)
        
        # 按相似度排序并返回前 n_results 个
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:n_results]
    
    def list_categories(self) -> List[str]:
        """
        列出所有可用的知识库分类
        
        扫描 data/graphrag/ 目录，返回所有子文件夹名称
        （排除单个文档存储，只返回分类文件夹）
        
        Returns:
            分类名称列表
        """
        categories = []
        
        if not self.base_persist_dir.exists():
            return categories
        
        for item in self.base_persist_dir.iterdir():
            if item.is_dir():
                # 检查是否是分类文件夹（包含子目录）
                has_subdirs = any(sub.is_dir() for sub in item.iterdir())
                if has_subdirs:
                    categories.append(item.name)
        
        return sorted(categories)
