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
from skills.graphrag.services.graphrag_service import Entity


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
    completed_chunks: List[int] = None  # 已完成的块索引列表（用于断点续传）
    
    def __post_init__(self):
        """初始化后处理"""
        if self.completed_chunks is None:
            self.completed_chunks = []


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
    
    def start_indexing(self, doc_id: str, silent: bool = True, resume: bool = True, force: bool = False) -> bool:
        """
        开始索引文档（后台静默处理）
        
        Args:
            doc_id: 文档ID
            silent: 是否静默模式（只显示50%和100%进度）
            resume: 是否支持断点续传（默认True）
            force: 是否强制重新索引（即使已完成也会重新索引）
            
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
        
        if doc_info.status == IndexingStatus.COMPLETED and not force:
            print(f"✅ 文档已索引: {doc_info.name}")
            return False
        
        # 强制重新索引时，清除之前的状态
        if force and doc_info.status == IndexingStatus.COMPLETED:
            print(f"🔄 强制重新索引: {doc_info.name}")
            # 清除旧的存储数据
            storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
            if storage_path.exists():
                import shutil
                shutil.rmtree(storage_path)
                print(f"   已清除旧索引数据")
            # 清空已完成的块记录
            doc_info.completed_chunks = []
            print(f"   已清除断点续传记录")
        
        # 检查是否支持断点续传（支持 FAILED 和 RUNNING 状态）
        if resume and not force and doc_info.status in [IndexingStatus.FAILED, IndexingStatus.RUNNING]:
            completed_count = len(doc_info.completed_chunks) if doc_info.completed_chunks else 0
            if completed_count > 0:
                status_text = "中断" if doc_info.status == IndexingStatus.RUNNING else "失败"
                print(f"🔄 检测到上次索引{status_text}，尝试断点续传: {doc_info.name}")
                print(f"   已记录 {completed_count} 个已完成的块")
                doc_info.status = IndexingStatus.PENDING
                # 根据已完成的块计算进度
                if doc_info.chunks_count > 0:
                    resume_progress = completed_count / doc_info.chunks_count * 0.95  # 95% 是块处理进度
                else:
                    resume_progress = 0.0
                # 保存进度到临时属性，启动线程时会使用
                doc_info._resume_progress = resume_progress
                doc_info.message = f"准备断点续传（{completed_count}块已完成）..."
            else:
                status_text = "中断" if doc_info.status == IndexingStatus.RUNNING else "失败"
                print(f"🔄 检测到上次索引{status_text}，重新开始: {doc_info.name}")
                doc_info.status = IndexingStatus.PENDING
                doc_info.progress = 0.0
                doc_info.message = "准备重新索引..."
            self._save_document_registry()
        
        # 启动后台线程
        doc_info.status = IndexingStatus.RUNNING
        # 如果是断点续传，保留已计算的进度；否则从0开始
        if not hasattr(doc_info, '_resume_progress'):
            doc_info.progress = 0.0
        else:
            doc_info.progress = doc_info._resume_progress
            delattr(doc_info, '_resume_progress')
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
            # 清空已完成的块记录（索引已完成，无需断点续传）
            doc_info.completed_chunks = []
            
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
            last_batch_start = 0

            # 定义批量进度回调（不保存 completed_chunks，避免时序问题）
            # 重要：completed_chunks 只在 add_document 真正成功后才会更新
            def batch_progress_callback(batch_idx, total_batches, batch_end, total_chunks, total_entities_so_far=0):
                nonlocal last_batch_start
                chunk_progress = 0.35 + (0.6 * batch_end / total_chunks)
                progress_callback(chunk_progress, f"处理块 {batch_end}/{total_chunks} (批次 {batch_idx}/{total_batches})...")
                doc_info.entities_count = total_entities_so_far
                last_batch_start = batch_end

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
            total_chunks = len(chunks)
            completed_chunks_set = set(doc_info.completed_chunks) if doc_info.completed_chunks else set()
            skipped_chunks = 0

            # 预生成所有 embedding（块 + 实体 + 关系）
            chunk_embeddings = {}
            entity_embeddings = {}  # chunk_idx -> list of embeddings
            relation_data = {}  # chunk_idx -> list of (source, target, relation_text)

            chunks_to_embed = []
            chunk_indices_to_embed = []

            for i, (entities, relations) in enumerate(batch_results):
                if i not in completed_chunks_set:
                    chunks_to_embed.append(chunks[i])
                    chunk_indices_to_embed.append(i)

            # 1. 批量生成 chunk embedding（多线程并行）
            if chunks_to_embed and graph_service._embedding_fn:
                logger.info(f"[{doc_info.name}] 批量生成 {len(chunks_to_embed)} 个块的 embedding...")
                progress_callback(0.95, f"生成 {len(chunks_to_embed)} 个块的向量...")

                # 动态并行处理
                current_workers = 50
                max_workers = 200
                current_batch_size = 10
                max_batch_size = 50
                embedded_count = 0
                lock = threading.Lock()
                speed_history = []
                start_time = time.time()
                last_speed_check = start_time
                last_completed = 0

                def process_chunk_batch(batch_data):
                    """处理一批 chunk 的 embedding"""
                    indices, texts = batch_data
                    embeddings = graph_service._embedding_fn(texts)
                    return list(zip(indices, embeddings))

                # 准备数据：分成小批次
                all_chunk_batches = []
                for i in range(0, len(chunks_to_embed), current_batch_size):
                    batch_indices = chunk_indices_to_embed[i:i + current_batch_size]
                    batch_texts = chunks_to_embed[i:i + current_batch_size]
                    all_chunk_batches.append((batch_indices, batch_texts))

                total_batches = len(all_chunk_batches)
                batch_idx = 0
                completed_batches = 0
                failed_batches = 0

                print(f"[DEBUG] 开始并行处理 {total_batches} 批 chunk (并行 {current_workers}, 每批 {current_batch_size})")

                while batch_idx < len(all_chunk_batches):
                    # 动态调整
                    if completed_batches > 0 and completed_batches % 20 == 0:
                        now = time.time()
                        recent_time = now - last_speed_check
                        recent_completed = completed_batches - last_completed
                        if recent_time > 0:
                            recent_speed = recent_completed / recent_time
                            speed_history.append(recent_speed)
                            if len(speed_history) > 10:
                                speed_history.pop(0)
                            avg_speed = sum(speed_history) / len(speed_history)

                            if avg_speed < 3.0 and current_workers < max_workers:
                                current_workers = min(current_workers + 30, max_workers)
                                print(f"[DEBUG] Chunk 速度过慢({avg_speed:.2f} 批/秒)，增加并行数到 {current_workers}")

                            last_speed_check = now
                            last_completed = completed_batches

                    # 并行处理
                    remaining_batches = all_chunk_batches[batch_idx:]
                    batches_this_round = min(current_workers, len(remaining_batches))

                    with ThreadPoolExecutor(max_workers=current_workers) as executor:
                        futures = []
                        for i in range(batches_this_round):
                            if batch_idx + i < len(all_chunk_batches):
                                futures.append(executor.submit(process_chunk_batch, all_chunk_batches[batch_idx + i]))

                        for future in futures:
                            try:
                                results = future.result()
                                with lock:
                                    for idx, embedding in results:
                                        chunk_embeddings[idx] = embedding
                                        embedded_count += 1
                                    completed_batches += 1

                                    if completed_batches % 20 == 0:
                                        embed_progress = 0.95 + (0.04 * embedded_count / len(chunks_to_embed))
                                        progress_callback(embed_progress, f"生成向量 {embedded_count}/{len(chunks_to_embed)}...")

                            except Exception as e:
                                with lock:
                                    failed_batches += 1

                    batch_idx += batches_this_round

                logger.info(f"[{doc_info.name}] 完成 chunk embedding 生成，共 {embedded_count} 个 (失败: {failed_batches})")

            # 2. 准备实体和关系数据，并批量生成 embedding
            entity_texts = []  # (chunk_idx, entity_idx, entity_text)
            relation_texts = []  # (chunk_idx, relation_idx, relation_text)

            print(f"[DEBUG] 开始收集 entity_texts 和 relation_texts，batch_results 长度: {len(batch_results)}")

            for i, (entities, relations) in enumerate(batch_results):
                if i in completed_chunks_set:
                    continue

                for entity in entities:
                    entity_text = f"{entity.name} {entity.type}"
                    entity_texts.append((i, len(entity_embeddings.get(i, [])), entity_text))

                for rel in relations:
                    rel_text = f"{rel.source} 与 {rel.target} 相关"
                    relation_texts.append((i, len(relation_data.get(i, [])), rel_text))

            print(f"[DEBUG] 收集完成，entity_texts: {len(entity_texts)}, relation_texts: {len(relation_texts)}")

            # 3. 批量生成 entity embeddings（使用多线程并行加速）
            print(f"[DEBUG] 开始生成 entity embeddings，entity_texts 数量: {len(entity_texts) if entity_texts else 0}")
            if entity_texts and graph_service._embedding_fn:
                logger.info(f"[{doc_info.name}] 批量生成 {len(entity_texts)} 个实体的 embedding...")

                # 多线程并行处理（Ollama API 逐个处理，用多线程并行多个请求）
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading
                import time

                # 动态并行数：根据 Ollama 能力和实测动态调整
                initial_workers = 50  # Ollama 一般支持 50-100 并发，起始值设高减少调整时间
                max_workers = 200     # 最大并行数
                min_workers = 20
                current_workers = initial_workers
                current_batch_size = 10
                max_batch_size = 50  # 最大批次大小
                entity_embed_count = 0
                lock = threading.Lock()

                def process_entity_batch(batch_data):
                    results = []
                    try:
                        batch_texts = [t[2] for t in batch_data]
                        embeddings = graph_service._embedding_fn(batch_texts)
                        results = list(zip(batch_data, embeddings))
                    except Exception as e:
                        logger.error(f"实体批次处理失败: {e}")
                    return results

                # 初始分批
                def split_batches(texts, batch_size):
                    batches = []
                    for i in range(0, len(texts), batch_size):
                        batches.append(texts[i:i+batch_size])
                    return batches

                all_batches = split_batches(entity_texts, current_batch_size)
                total_batches = len(all_batches)
                batch_idx = 0
                completed_batches = 0
                failed_batches = 0
                start_time = time.time()
                last_speed_check = start_time
                last_completed = 0
                speed_history = []

                print(f"[DEBUG] 开始动态并行处理 {total_batches} 批实体 (初始: 每批 {current_batch_size}, 并行 {current_workers})")

                while batch_idx < len(all_batches):
                    # 动态调整：根据速度调整并行数和批次大小
                    if completed_batches > 0 and completed_batches % 20 == 0:
                        now = time.time()
                        recent_time = now - last_speed_check
                        recent_completed = completed_batches - last_completed
                        if recent_time > 0:
                            recent_speed = recent_completed / recent_time
                            speed_history.append(recent_speed)
                            if len(speed_history) > 10:
                                speed_history.pop(0)
                            avg_speed = sum(speed_history) / len(speed_history)

                            # 动态调整策略：如果速度 < 3 批/秒，就认为需要加速
                            target_speed = 3.0  # 目标速度：每秒至少 3 批
                            if avg_speed < target_speed:
                                if current_workers < max_workers:
                                    current_workers = min(current_workers + 30, max_workers)
                                    print(f"[DEBUG] 速度过慢({avg_speed:.2f} 批/秒)，增加并行数到 {current_workers}")
                                elif current_batch_size < max_batch_size:
                                    current_batch_size = min(current_batch_size + 20, max_batch_size)
                                    # 重新分批
                                    all_batches = split_batches(entity_texts, current_batch_size)
                                    total_batches = len(all_batches)
                                    print(f"[DEBUG] 增加批次大小到 {current_batch_size}，重新分批为 {total_batches} 批")

                            last_speed_check = now
                            last_completed = completed_batches

                    # 使用当前并行数创建线程池处理一批
                    remaining_batches = all_batches[batch_idx:]
                    batches_this_round = min(current_workers, len(remaining_batches))

                    with ThreadPoolExecutor(max_workers=current_workers) as executor:
                        futures = []
                        for i in range(batches_this_round):
                            if batch_idx + i < len(all_batches):
                                futures.append((batch_idx + i, executor.submit(process_entity_batch, all_batches[batch_idx + i])))

                        for idx, future in futures:
                            results = future.result()
                            with lock:
                                if results:
                                    for (chunk_idx, entity_idx, entity_text), embedding in results:
                                        if chunk_idx not in entity_embeddings:
                                            entity_embeddings[chunk_idx] = []
                                        entity_embeddings[chunk_idx].append(embedding)
                                        entity_embed_count += 1
                                    completed_batches += 1
                                else:
                                    failed_batches += 1

                    batch_idx += batches_this_round

                    if completed_batches % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed_batches / elapsed if elapsed > 0 else 0
                        eta = (total_batches - completed_batches) / rate if rate > 0 else 0
                        print(f"[DEBUG] 实体进度: {completed_batches}/{total_batches} 批, 速度: {rate:.2f} 批/秒, 剩余: {eta:.0f} 秒, 并行: {current_workers}, 批次: {current_batch_size}")

                print(f"[DEBUG] 完成 entity embeddings 生成，共 {entity_embed_count} 个 (失败: {failed_batches})")
            else:
                print(f"[DEBUG] 跳过 entity embeddings（entity_texts 为空或无 embedding 函数）")

            # 4. 批量生成 relation embeddings（同样使用多线程并行）
            print(f"[DEBUG] 开始生成 relation embeddings，relation_texts 数量: {len(relation_texts) if relation_texts else 0}")
            if relation_texts and graph_service._embedding_fn:
                logger.info(f"[{doc_info.name}] 批量生成 {len(relation_texts)} 个关系的 embedding...")

                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading
                import time

                # 动态并行（复用 entity 的配置）
                current_workers = 50
                max_workers = 200
                current_batch_size = 10
                max_batch_size = 50
                rel_embed_count = 0
                lock = threading.Lock()

                def process_relation_batch(batch_data):
                    results = []
                    try:
                        batch_texts = [t[2] for t in batch_data]
                        embeddings = graph_service._embedding_fn(batch_texts)
                        results = list(zip(batch_data, embeddings))
                    except Exception as e:
                        logger.error(f"关系批次处理失败: {e}")
                    return results

                def split_batches(texts, batch_size):
                    batches = []
                    for i in range(0, len(texts), batch_size):
                        batches.append(texts[i:i+batch_size])
                    return batches

                all_batches = split_batches(relation_texts, current_batch_size)
                total_batches = len(all_batches)
                batch_idx = 0
                completed_batches = 0
                failed_batches = 0
                start_time = time.time()
                last_speed_check = start_time
                last_completed = 0
                speed_history = []

                print(f"[DEBUG] 开始动态并行处理 {total_batches} 批关系 (初始: 每批 {current_batch_size}, 并行 {current_workers})")

                while batch_idx < len(all_batches):
                    if completed_batches > 0 and completed_batches % 20 == 0:
                        now = time.time()
                        recent_time = now - last_speed_check
                        recent_completed = completed_batches - last_completed
                        if recent_time > 0:
                            recent_speed = recent_completed / recent_time
                            speed_history.append(recent_speed)
                            if len(speed_history) > 10:
                                speed_history.pop(0)
                            avg_speed = sum(speed_history) / len(speed_history)

                            target_speed = 3.0
                            if avg_speed < target_speed:
                                if current_workers < max_workers:
                                    current_workers = min(current_workers + 30, max_workers)
                                    print(f"[DEBUG] 关系速度过慢({avg_speed:.2f} 批/秒)，增加并行数到 {current_workers}")
                                elif current_batch_size < max_batch_size:
                                    current_batch_size = min(current_batch_size + 20, max_batch_size)
                                    all_batches = split_batches(relation_texts, current_batch_size)
                                    total_batches = len(all_batches)
                                    print(f"[DEBUG] 关系增加批次大小到 {current_batch_size}，重新分批为 {total_batches} 批")

                            last_speed_check = now
                            last_completed = completed_batches

                    remaining_batches = all_batches[batch_idx:]
                    batches_this_round = min(current_workers, len(remaining_batches))

                    with ThreadPoolExecutor(max_workers=current_workers) as executor:
                        futures = []
                        for i in range(batches_this_round):
                            if batch_idx + i < len(all_batches):
                                futures.append((batch_idx + i, executor.submit(process_relation_batch, all_batches[batch_idx + i])))

                        for idx, future in futures:
                            results = future.result()
                            with lock:
                                if results:
                                    for (chunk_idx, rel_idx, rel_text), embedding in results:
                                        if chunk_idx not in relation_data:
                                            relation_data[chunk_idx] = []
                                        relation_data[chunk_idx].append((rel_text, embedding))
                                        rel_embed_count += 1
                                    completed_batches += 1
                                else:
                                    failed_batches += 1

                    batch_idx += batches_this_round

                    if completed_batches % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed_batches / elapsed if elapsed > 0 else 0
                        eta = (total_batches - completed_batches) / rate if rate > 0 else 0
                        print(f"[DEBUG] 关系进度: {completed_batches}/{total_batches} 批, 速度: {rate:.2f} 批/秒, 剩余: {eta:.0f} 秒")

                print(f"[DEBUG] 完成 relation embeddings 生成，共 {rel_embed_count} 个 (失败: {failed_batches})")
            else:
                print(f"[DEBUG] 跳过 relation embeddings（relation_texts 为空或无 embedding 函数）")

            # 5. 批量添加文档到图谱
            documents_to_add = []
            print(f"[DEBUG] 开始构建 documents_to_add，batch_results 长度: {len(batch_results)}, completed_chunks_set: {completed_chunks_set}")
            print(f"[DEBUG] chunk_embeddings 键数量: {len(chunk_embeddings)}")
            print(f"[DEBUG] entity_embeddings 键数量: {len(entity_embeddings)}")
            print(f"[DEBUG] relation_data 键数量: {len(relation_data)}")

            for i, (entities, relations) in enumerate(batch_results):
                if i in completed_chunks_set:
                    skipped_chunks += 1
                    total_entities += len(entities)
                    total_relations += len(relations)
                    continue

                # 检查实体和关系数据
                has_entities = len(entities) > 0
                has_embedding = i in chunk_embeddings and chunk_embeddings[i] is not None

                if not has_embedding:
                    logger.warning(f"[{doc_info.name}] 块 {i} 没有 embedding 或 embedding 为 None")

                chunk_doc_id = f"{doc_id}_{i}"
                entity_objs = [Entity(name=e.name, type=e.type, positions=[(e.start, e.end)]) for e in entities]

                metadata = {
                    "source": doc_info.file_path,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "entities": ",".join([e.name for e in entities[:10]]),
                    "relations": ",".join([f"{r.source}-{r.relation}-{r.target}" for r in relations[:5]])
                }

                documents_to_add.append({
                    "text": chunks[i],
                    "doc_id": chunk_doc_id,
                    "metadata": metadata,
                    "entities": entity_objs,
                    "entity_embeddings": entity_embeddings.get(i, []),
                    "relation_texts": [r[0] for r in relation_data.get(i, [])],
                    "relation_embeddings": [r[1] for r in relation_data.get(i, [])],
                    "embedding": chunk_embeddings.get(i)
                })

            logger.info(f"[{doc_info.name}] documents_to_add 构建完成，长度: {len(documents_to_add)}")
            print(f"[DEBUG] documents_to_add 构建完成，长度: {len(documents_to_add)}")

            # 批量处理
            if documents_to_add:
                print(f"[DEBUG] 开始批量保存 {len(documents_to_add)} 个文档...")
                logger.info(f"[{doc_info.name}] 开始批量保存 {len(documents_to_add)} 个文档...")

                # 禁用自动保存以提升性能
                if graph_service.triple_store:
                    graph_service.triple_store.set_auto_save(False)

                # 收集所有块、实体、关系数据
                all_chunk_ids = []
                all_chunk_texts = []
                all_chunk_embeddings = []
                all_chunk_metadatas = []

                all_entity_ids = []
                all_entity_texts = []
                all_entity_embeddings = []
                all_entity_metadatas = []

                all_relation_ids = []
                all_relation_texts = []
                all_relation_embeddings = []
                all_relation_metadatas = []

                for doc in documents_to_add:
                    if doc["embedding"] is None:
                        continue

                    # 收集块数据
                    all_chunk_ids.append(doc["doc_id"])
                    all_chunk_texts.append(doc["text"])
                    all_chunk_embeddings.append(doc["embedding"])
                    all_chunk_metadatas.append(doc["metadata"])

                    # 收集实体数据
                    for entity, embedding in zip(doc["entities"], doc["entity_embeddings"]):
                        entity_id = f"{doc['doc_id']}_{entity.name}_{entity.type}"
                        entity_text = f"{entity.name} {entity.type}"
                        all_entity_ids.append(entity_id)
                        all_entity_texts.append(entity_text)
                        all_entity_embeddings.append(embedding)
                        all_entity_metadatas.append({
                            "name": entity.name,
                            "type": entity.type,
                            "doc_id": doc["doc_id"]
                        })
                        graph_service._index_entity(entity, doc["doc_id"])

                    # 收集关系数据
                    for rel_text, embedding in zip(doc["relation_texts"], doc["relation_embeddings"]):
                        rel_id = f"{doc['doc_id']}_{rel_text[:50]}"
                        all_relation_ids.append(rel_id)
                        all_relation_texts.append(rel_text)
                        all_relation_embeddings.append(embedding)
                        all_relation_metadatas.append({
                            "source": doc["doc_id"],
                            "description": rel_text
                        })

                        from skills.graphrag.services import Relation
                        relation = Relation(
                            source=doc["doc_id"],
                            target="",
                            relation="CO_OCCUR",
                            confidence=0.5,
                            doc_id=doc["doc_id"]
                        )
                        graph_service.relations.append(relation)

                    if doc["metadata"]["chunk_index"] not in doc_info.completed_chunks:
                        doc_info.completed_chunks.append(doc["metadata"]["chunk_index"])

                    total_entities += len(doc["entities"])
                    total_relations += len(doc["relation_texts"])
                    doc_info.entities_count = total_entities

                # 批量保存块
                if all_chunk_ids:
                    print(f"[DEBUG] 批量保存 {len(all_chunk_ids)} 个块...")
                    graph_service.collection.add(
                        ids=all_chunk_ids,
                        documents=all_chunk_texts,
                        metadatas=all_chunk_metadatas,
                        embeddings=all_chunk_embeddings
                    )
                    if graph_service.triple_store:
                        for i in range(len(all_chunk_ids)):
                            graph_service.triple_store.add_chunk(
                                chunk_id=all_chunk_ids[i],
                                text=all_chunk_texts[i],
                                embedding=all_chunk_embeddings[i],
                                metadata=all_chunk_metadatas[i]
                            )

                # 批量保存实体
                if all_entity_ids:
                    print(f"[DEBUG] 批量保存 {len(all_entity_ids)} 个实体...")
                    from skills.graphrag.services.triple_vector_store import EntityInfo
                    for i in range(len(all_entity_ids)):
                        entity_info = EntityInfo(
                            name=all_entity_metadatas[i]["name"],
                            entity_type=all_entity_metadatas[i]["type"],
                            description=all_entity_texts[i],
                            source_ids=[all_entity_metadatas[i]["doc_id"]]
                        )
                        graph_service.triple_store.add_entity(entity_info, all_entity_embeddings[i])

                # 批量保存关系
                if all_relation_ids:
                    print(f"[DEBUG] 批量保存 {len(all_relation_ids)} 个关系...")
                    from skills.graphrag.services.triple_vector_store import RelationInfo
                    for i in range(len(all_relation_ids)):
                        relation_info = RelationInfo(
                            source=all_relation_metadatas[i]["source"],
                            target="",
                            relation="CO_OCCUR",
                            description=all_relation_texts[i],
                            confidence=0.5,
                            source_ids=[all_relation_metadatas[i]["source"]]
                        )
                        graph_service.triple_store.add_relation(relation_info, all_relation_embeddings[i])

                # 最后一次性保存
                if graph_service.triple_store:
                    print("[DEBUG] 批量保存完成，调用 flush...")
                    graph_service.triple_store.flush()
                    graph_service.triple_store.set_auto_save(True)
                graph_service._save_index()
                self._save_document_registry()
                progress_callback(1.0, f"保存完成")

            # 显示断点续传统计
            if skipped_chunks > 0:
                logger.info(f"[{doc_info.name}] 断点续传: 跳过 {skipped_chunks}/{total_chunks} 个已完成的块")

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
                        completed_at=doc_data.get("completed_at"),
                        completed_chunks=doc_data.get("completed_chunks", [])
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
                "completed_at": doc_info.completed_at,
                "completed_chunks": doc_info.completed_chunks if doc_info.completed_chunks else []
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
        恢复所有失败或中断的索引任务
        
        Returns:
            恢复的文档ID列表
        """
        failed_docs = [
            doc_id for doc_id, doc_info in self.documents.items()
            if doc_info.status in [IndexingStatus.FAILED, IndexingStatus.RUNNING]
        ]
        
        if not failed_docs:
            print("✅ 没有需要恢复的失败或中断任务")
            return []
        
        # 分类统计
        failed_count = sum(1 for doc_id in failed_docs if self.documents[doc_id].status == IndexingStatus.FAILED)
        running_count = len(failed_docs) - failed_count
        
        status_msg = []
        if failed_count > 0:
            status_msg.append(f"{failed_count}个失败")
        if running_count > 0:
            status_msg.append(f"{running_count}个中断")
        
        print(f"🔄 发现 {len(failed_docs)} 个需要恢复的任务（{'，'.join(status_msg)}），开始恢复...")
        
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
