"""
知识库管理器

自动扫描、索引和管理知识库文档。
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import os

from skills.graphrag.utils import DocumentProcessor
from skills.graphrag.services import GraphRAGService, LightweightEntityExtractor, Entity


class KnowledgeBaseManager:
    """
    知识库管理器

    功能：
    1. 自动扫描知识库目录
    2. 增量索引新文档
    3. 管理文档状态

    使用方法：
        kb = KnowledgeBaseManager(graph_service, entity_extractor)
        kb.scan_and_index("./knowledge_base")  # 扫描并索引
    """

    # 支持的文档类型
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}

    def __init__(
        self,
        graph_service: GraphRAGService,
        entity_extractor: LightweightEntityExtractor,
        document_processor: Optional[DocumentProcessor] = None
    ):
        self.graph_service = graph_service
        self.entity_extractor = entity_extractor
        # 使用更大的分块大小以减少块数量
        self.document_processor = document_processor or DocumentProcessor(
            chunk_size=2000,
            chunk_overlap=200
        )

    def scan_and_index(self, kb_dir: str) -> Dict[str, Any]:
        """
        扫描知识库目录并索引新文档

        Args:
            kb_dir: 知识库目录路径

        Returns:
            索引结果统计
        """
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            print(f"⚠️ 知识库目录不存在: {kb_dir}")
            return {"indexed": 0, "skipped": 0, "errors": 0, "files": []}

        # 获取所有支持的文档
        files = self._scan_directory(kb_path)

        if not files:
            print(f"📭 知识库目录为空: {kb_dir}")
            return {"indexed": 0, "skipped": 0, "errors": 0, "files": []}

        print(f"📚 发现 {len(files)} 个文档，开始索引...")

        # 获取已索引的文档ID
        indexed_docs = self._get_indexed_documents()

        # 索引新文档
        results = {"indexed": 0, "skipped": 0, "errors": 0, "files": []}

        for file_path in files:
            doc_id = self._generate_doc_id(file_path)

            if doc_id in indexed_docs:
                print(f"  ⏭️  跳过（已索引）: {file_path.name}")
                results["skipped"] += 1
                continue

            try:
                self._index_document(file_path, doc_id)
                print(f"  ✅ 已索引: {file_path.name}")
                results["indexed"] += 1
                results["files"].append(str(file_path))
            except Exception as e:
                print(f"  ❌ 索引失败: {file_path.name} - {e}")
                results["errors"] += 1

        return results

    def _scan_directory(self, kb_path: Path) -> List[Path]:
        """扫描目录获取所有支持的文档"""
        files = []

        for item in kb_path.rglob("*"):
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(item)

        return sorted(files)

    def _generate_doc_id(self, file_path: Path) -> str:
        """生成文档ID（基于相对路径）"""
        # 使用文件路径的哈希作为文档ID
        import hashlib
        return hashlib.md5(str(file_path).encode()).hexdigest()[:16]

    def _get_indexed_documents(self) -> set:
        """获取已索引的文档ID集合"""
        # 从向量存储中获取所有文档ID
        try:
            all_ids = self.graph_service.collection.get_all_ids()
            # 提取基础文档ID（去掉 _chunk_index 后缀）
            base_ids = set()
            for doc_id in all_ids:
                # 文档ID格式: {base_id}_{chunk_index}
                if "_" in doc_id:
                    base_id = doc_id.rsplit("_", 1)[0]
                    base_ids.add(base_id)
            return base_ids
        except Exception:
            return set()

    def _index_document(self, file_path: Path, doc_id: str, verbose: bool = True) -> None:
        """索引单个文档

        Args:
            file_path: 文件路径
            doc_id: 文档ID
            verbose: 是否打印详细输出
        """
        import time

        if verbose:
            print(f"    📖 读取文档: {file_path.name}")

        # 读取文档
        text = self.document_processor.read_document(str(file_path))

        if not text:
            raise ValueError("无法读取文档内容")

        if verbose:
            print(f"    ✅ 读取完成，长度: {len(text)} 字符")

        # 分块
        if verbose:
            print(f"    🔄 分块中...")
        chunks = self.document_processor.chunk_text(text)
        if verbose:
            print(f"    ✅ 分块完成: {len(chunks)} 块")

        # 如果块数太多，提示用户
        if len(chunks) > 100 and verbose:
            print(f"    ⚠️ 文档较大 ({len(chunks)} 块)，索引可能需要较长时间...")

        # 处理每个块
        if verbose:
            print(f"    🔄 索引 {len(chunks)} 个块...")
        start_time = time.time()

        for i, chunk in enumerate(chunks):
            try:
                block_start = time.time()

                # 步骤1: 抽取实体
                step_start = time.time()
                entities = self.entity_extractor.extract(chunk)
                extract_time = time.time() - step_start

                # 步骤2: 转换为 Entity 对象
                entity_objects = [
                    Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                    for e in entities
                ]

                # 步骤3: 添加到图谱（这会调用 Embedding 模型）
                step_start = time.time()
                chunk_doc_id = f"{doc_id}_{i}"
                self.graph_service.add_document(
                    text=chunk,
                    doc_id=chunk_doc_id,
                    metadata={
                        "source": str(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": file_path.name
                    },
                    entities=entity_objects
                )
                add_time = time.time() - step_start

                # 每块都显示进度（前3块、后3块、以及每10块）
                if verbose and (i < 3 or i >= len(chunks) - 3 or (i + 1) % 10 == 0):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (len(chunks) - i - 1)
                    print(f"      已处理 {i + 1}/{len(chunks)} 块 (实体抽取:{extract_time:.1f}s, 添加到图谱:{add_time:.1f}s, 剩余约 {remaining:.0f}s)")

                # 每处理 20 块后暂停一下，给系统喘息时间
                if (i + 1) % 20 == 0:
                    time.sleep(0.5)

                # 每 50 块保存一次索引
                if (i + 1) % 50 == 0:
                    self.graph_service._save_index()
                    if verbose:
                        print(f"      💾 索引已保存")

            except Exception as e:
                if verbose:
                    print(f"      ⚠️ 处理第 {i+1} 块时出错: {e}")
                # 出错后暂停一下再继续
                time.sleep(1)
                continue

        total_time = time.time() - start_time
        if verbose:
            print(f"    ✅ 文档索引完成，总耗时: {total_time:.1f}s")

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.graph_service.get_stats()

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索知识库"""
        # 提取查询实体
        query_entities = self.entity_extractor.extract_from_query(query)

        if query_entities:
            # 实体增强检索
            results = self.graph_service.enhanced_search(
                query=query,
                query_entities=query_entities,
                n_results=n_results
            )
            return results.get("documents", [])
        else:
            # 纯向量检索
            return self.graph_service.vector_search(query, n_results=n_results)
