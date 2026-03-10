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

from ollamapilot.ollama_lock import OllamaLockContext


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
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ):
        """
        初始化文档管理器
        
        Args:
            base_persist_dir: 基础持久化目录
            embedding_model: Embedding模型名称
            progress_callback: 进度回调函数(doc_id, progress, message)
        """
        self.base_persist_dir = Path(base_persist_dir)
        self.base_persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.progress_callback = progress_callback
        
        # 文档信息存储
        self.documents: Dict[str, DocumentInfo] = {}
        
        # 后台任务
        self._indexing_tasks: Dict[str, threading.Thread] = {}
        
        # 加载已有文档信息
        self._load_document_registry()
    
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
        """执行文档索引（实际工作）
        
        使用Ollama锁防止与生成模型并发冲突
        """
        from skills.graphrag.utils import DocumentProcessor
        from skills.graphrag.services import GraphRAGService, LightweightEntityExtractor
        from skills.graphrag.knowledge_base import KnowledgeBaseManager
        import logging
        
        logger = logging.getLogger(__name__)
        doc_info = self.documents[doc_id]
        
        # 获取存储路径
        storage_path = self._get_document_storage_path(doc_info.name, doc_info.model_name)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        progress_callback(0.05, "准备索引...")
        logger.info(f"[{doc_info.name}] 开始索引，文档路径: {doc_info.file_path}")
        
        # 初始化服务（不需要锁）
        try:
            progress_callback(0.1, "初始化服务...")
            graph_service = GraphRAGService(
                persist_dir=str(storage_path),
                embedding_model=doc_info.model_name
            )
            
            entity_extractor = LightweightEntityExtractor()
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
            
        except Exception as e:
            logger.error(f"[{doc_info.name}] 初始化或分块失败: {e}")
            progress_callback(0.0, f"准备阶段失败: {e}")
            raise
        
        # 获取Ollama锁，防止与生成模型并发
        # 使用较长的超时时间（10分钟），因为PDF索引可能需要很长时间
        progress_callback(0.35, "等待Ollama资源（可能需要几分钟）...")
        
        try:
            with OllamaLockContext(owner=f"index_{doc_info.name}", timeout=600):
                progress_callback(0.4, "开始向量化和实体抽取...")
                logger.info(f"[{doc_info.name}] 获取Ollama锁，开始处理 {len(chunks)} 个块")
                
                # 处理每个块
                total_entities = 0
                for i, chunk in enumerate(chunks):
                    chunk_progress = 0.4 + (0.55 * (i + 1) / len(chunks))
                    progress_callback(chunk_progress, f"处理块 {i+1}/{len(chunks)}...")
                    
                    # 抽取实体
                    entities = entity_extractor.extract(chunk)
                    total_entities += len(entities)
                    
                    # 添加到图谱
                    chunk_doc_id = f"{doc_id}_{i}"
                    from skills.graphrag.services import Entity
                    entity_objects = [
                        Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                        for e in entities
                    ]
                    
                    graph_service.add_document(
                        text=chunk,
                        doc_id=chunk_doc_id,
                        metadata={
                            "source": doc_info.file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        },
                        entities=entity_objects
                    )
                    
                    # 每5块保存一次
                    if (i + 1) % 5 == 0:
                        graph_service._save_index()
                
                doc_info.entities_count = total_entities
                progress_callback(1.0, "索引完成")
                logger.info(f"[{doc_info.name}] 索引完成: {total_entities} 个实体")
                
        except TimeoutError as e:
            logger.error(f"[{doc_info.name}] 获取Ollama锁超时: {e}")
            progress_callback(0.0, f"等待Ollama资源超时，请稍后重试")
            raise
        except Exception as e:
            logger.error(f"[{doc_info.name}] 索引错误: {e}")
            progress_callback(0.0, f"索引错误: {e}")
            raise
    
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
