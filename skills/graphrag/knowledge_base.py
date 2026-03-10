"""
知识库管理器

自动扫描、索引和管理知识库文档。
集成 WordAligner 提供精确位置映射能力。
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import time

from skills.graphrag.utils import DocumentProcessor
from skills.graphrag.services import GraphRAGService, HybridEntityExtractor, Entity
from skills.graphrag.llm_client import SimpleLLMClient
from skills.graphrag.word_aligner import (
    WordAligner,
    AlignedEntity,
    AlignmentStatus,
    format_alignment_report,
    calculate_chunk_offsets
)


class KnowledgeBaseManager:
    """
    知识库管理器

    功能：
    1. 自动扫描知识库目录
    2. 增量索引新文档
    3. 管理文档状态
    4. 集成 WordAligner 精确位置映射

    使用方法：
        kb = KnowledgeBaseManager(graph_service, entity_extractor)
        kb.scan_and_index("./knowledge_base")  # 扫描并索引
    """

    # 支持的文档类型
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}

    def __init__(
        self,
        graph_service: GraphRAGService,
        entity_extractor: Optional[HybridEntityExtractor] = None,
        document_processor: Optional[DocumentProcessor] = None,
        enable_word_aligner: bool = True,
        fuzzy_threshold: float = 0.75,
        persist_dir: str = "./data/graphrag"
    ):
        self.graph_service = graph_service
        # 初始化混合实体抽取器（如果没有提供）
        self.entity_extractor = entity_extractor or HybridEntityExtractor(persist_dir=persist_dir)
        # 初始化LLM客户端
        self.llm_client = SimpleLLMClient()
        self.use_llm = self.llm_client.is_available()

        # 使用更大的分块大小以减少块数量
        self.document_processor = document_processor or DocumentProcessor(
            chunk_size=2000,
            chunk_overlap=200
        )

        # 初始化 WordAligner
        self.enable_word_aligner = enable_word_aligner
        if enable_word_aligner:
            self.word_aligner = WordAligner(fuzzy_threshold=fuzzy_threshold)
        else:
            self.word_aligner = None

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
        if self.enable_word_aligner:
            print(f"   🎯 WordAligner 已启用 (模糊阈值: {self.word_aligner.fuzzy_threshold})")

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
        """索引单个文档（集成 WordAligner）

        Args:
            file_path: 文件路径
            doc_id: 文档ID
            verbose: 是否打印详细输出
        """
        if verbose:
            print(f"    📖 读取文档: {file_path.name}")

        # 读取文档（保存全文用于对齐）
        full_text = self.document_processor.read_document(str(file_path))

        if not full_text:
            raise ValueError("无法读取文档内容")

        if verbose:
            print(f"    ✅ 读取完成，长度: {len(full_text)} 字符")

        # 分块
        if verbose:
            print(f"    🔄 分块中...")
        chunks = self.document_processor.chunk_text(full_text)
        if verbose:
            print(f"    ✅ 分块完成: {len(chunks)} 块")

        # 如果块数太多，提示用户
        if len(chunks) > 100 and verbose:
            print(f"    ⚠️ 文档较大 ({len(chunks)} 块)，索引可能需要较长时间...")

        # 计算每块在全文中的偏移量
        chunk_offsets = calculate_chunk_offsets(full_text, chunks)

        # 处理每个块
        if verbose:
            print(f"    🔄 索引 {len(chunks)} 个块...")
        start_time = time.time()

        # 收集所有实体用于对齐
        all_raw_entities = []

        for i, chunk in enumerate(chunks):
            try:
                block_start = time.time()

                # 步骤1: 使用混合模式抽取实体和关系
                step_start = time.time()
                entities, relations = self.entity_extractor.extract(
                    chunk,
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    top_k=20
                )
                extract_time = time.time() - step_start

                # 收集原始实体信息
                for e in entities:
                    all_raw_entities.append({
                        'name': e.name,
                        'type': e.type,
                        'chunk_index': i,
                        'start': e.start,
                        'end': e.end,
                        'source': e.source
                    })

                # 步骤2: 添加到图谱（这会调用 Embedding 模型）
                step_start = time.time()
                chunk_doc_id = f"{doc_id}_{i}"
                
                # 临时使用简单位置，后续会更新为对齐后的位置
                entity_objects = [
                    Entity(name=e.name, type=e.type, positions=[(e.start, e.end)])
                    for e in entities
                ]
                
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

        # 步骤3: 使用 WordAligner 对齐所有实体
        if self.enable_word_aligner and all_raw_entities:
            if verbose:
                print(f"    🎯 对齐 {len(all_raw_entities)} 个实体到原文...")
            
            align_start = time.time()
            aligned_entities = self.word_aligner.align_entities(
                entities=all_raw_entities,
                source_text=full_text,
                chunks=chunks,
                chunk_offsets=chunk_offsets
            )
            align_time = time.time() - align_start
            
            if verbose:
                print(f"    ✅ 对齐完成: {len(aligned_entities)} 个实体 ({align_time:.2f}s)")
                # 显示对齐报告
                report = format_alignment_report(aligned_entities, full_text, max_display=5)
                print(report)
            
            # 更新图谱中的实体位置（使用对齐后的位置）
            self._update_entity_positions(doc_id, aligned_entities, full_text)

        total_time = time.time() - start_time
        if verbose:
            print(f"    ✅ 文档索引完成，总耗时: {total_time:.1f}s")

    def _update_entity_positions(
        self, 
        doc_id: str, 
        aligned_entities: List[AlignedEntity],
        full_text: str
    ):
        """更新图谱中实体的位置信息（使用对齐后的精确位置）"""
        for aligned in aligned_entities:
            # 更新实体索引中的位置信息
            if aligned.name in self.graph_service.entity_index:
                entity_info = self.graph_service.entity_index[aligned.name]
                
                # 添加对齐后的位置（如果还没有）
                new_position = (aligned.start, aligned.end)
                if new_position not in entity_info.get("positions", []):
                    if "positions" not in entity_info:
                        entity_info["positions"] = []
                    entity_info["positions"].append(new_position)
                
                # 保存对齐状态
                entity_info["alignment_status"] = aligned.status.value
                entity_info["similarity"] = aligned.similarity
                
                # 保存上下文片段（用于验证）
                context = full_text[max(0, aligned.start-20):min(len(full_text), aligned.end+20)]
                entity_info["context"] = context

    def get_entity_verification_report(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        获取实体验证报告
        
        Args:
            entity_name: 实体名称
            
        Returns:
            验证报告字典
        """
        if entity_name not in self.graph_service.entity_index:
            return None
        
        entity_info = self.graph_service.entity_index[entity_name]
        
        return {
            "entity_name": entity_name,
            "entity_type": entity_info.get("type", "Unknown"),
            "document_count": len(entity_info.get("doc_ids", set())),
            "alignment_status": entity_info.get("alignment_status", "unknown"),
            "similarity": entity_info.get("similarity", 1.0),
            "positions": entity_info.get("positions", []),
            "context": entity_info.get("context", ""),
        }

    def format_entity_highlight(
        self, 
        entity_name: str, 
        source_text: str,
        context_chars: int = 30
    ) -> Optional[str]:
        """
        格式化实体高亮显示
        
        Args:
            entity_name: 实体名称
            source_text: 原文
            context_chars: 上下文字符数
            
        Returns:
            带高亮的文本
        """
        if entity_name not in self.graph_service.entity_index:
            return None
        
        entity_info = self.graph_service.entity_index[entity_name]
        positions = entity_info.get("positions", [])
        
        if not positions:
            return None
        
        highlights = []
        for start, end in positions[:5]:  # 最多显示5个位置
            before = source_text[max(0, start-context_chars):start]
            entity = source_text[start:end]
            after = source_text[end:min(len(source_text), end+context_chars)]
            
            highlights.append(f"  ...{before}[[{entity}]]{after}...")
        
        return "\n".join(highlights)

    def get_alignment_stats(self) -> Dict[str, Any]:
        """
        获取对齐统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_entities": len(self.graph_service.entity_index),
            "exact_match": 0,
            "lesser_match": 0,
            "fuzzy_match": 0,
            "unmatched": 0,
            "unknown": 0,
        }
        
        for entity_name, entity_info in self.graph_service.entity_index.items():
            status = entity_info.get("alignment_status", "unknown")
            if status == "exact":
                stats["exact_match"] += 1
            elif status == "lesser":
                stats["lesser_match"] += 1
            elif status == "fuzzy":
                stats["fuzzy_match"] += 1
            elif status == "unmatched":
                stats["unmatched"] += 1
            else:
                stats["unknown"] += 1
        
        # 计算百分比
        total = stats["total_entities"]
        if total > 0:
            stats["exact_match_pct"] = f"{stats['exact_match'] / total * 100:.1f}%"
            stats["fuzzy_match_pct"] = f"{stats['fuzzy_match'] / total * 100:.1f}%"
        
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        base_stats = self.graph_service.get_stats()
        
        if self.enable_word_aligner:
            alignment_stats = self.get_alignment_stats()
            base_stats["alignment"] = alignment_stats
        
        return base_stats

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
