"""
混合模式实体抽取器

结合词典匹配 + LLM抽取，支持动态学习和人工干预
集成全局预设词典和文档私有词典
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
import re
import json
from dataclasses import dataclass, asdict


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "dictionary"  # "dictionary" 或 "llm"


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    source: str
    target: str
    relation: str
    confidence: float


class HybridEntityExtractor:
    """
    混合模式实体抽取器

    工作流程：
    1. 加载全局预设词典 + 文档私有词典
    2. 词典匹配（快速、高置信度）
    3. LLM抽取（智能、发现新实体）
    4. 合并结果（去重、排序）
    5. 动态学习（新实体加入文档私有词典）

    词典架构：
    - 全局预设词典：config/dictionaries/（用户维护，只读）
    - 文档私有词典：data/graphrag/{doc_id}/dictionary.json（LLM学习，可写）
    """

    def __init__(
        self,
        persist_dir: str = "./data/graphrag",
        doc_id: Optional[str] = None,
        selected_dictionaries: Optional[List[str]] = None
    ):
        """
        初始化抽取器

        Args:
            persist_dir: 词典持久化目录
            doc_id: 文档ID（用于加载文档私有词典）
            selected_dictionaries: 选择的全局词典列表，None表示全部
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.doc_id = doc_id

        # 停用词
        self.stopwords = self._load_stopwords()

        # 加载词典管理器
        from skills.graphrag.dictionary_manager import get_dictionary_manager
        self.dict_manager = get_dictionary_manager()

        # 确定要加载的全局词典
        if selected_dictionaries is None:
            # 默认加载所有全局词典
            selected_dictionaries = self.dict_manager.list_global_dictionaries()
        self.selected_dictionaries = selected_dictionaries

        # 文档私有词典路径
        self.doc_dictionary_path = None
        if doc_id:
            self.doc_dictionary_path = self.persist_dir / doc_id / "dictionary.json"

        # 加载合并后的词典
        self.entity_dict = self._load_merged_dictionary()

        # 构建快速查找索引
        self._rebuild_index()

    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        return set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '之', '与', '及', '等', '或', '但', '而', '因为', '所以',
        ])

    def _load_merged_dictionary(self) -> Dict[str, List[str]]:
        """
        加载合并后的词典

        合并顺序：
        1. 全局预设词典（用户维护，只读）
        2. 文档私有词典（LLM学习，可写）
        """
        # 使用词典管理器加载合并后的词典
        merged = self.dict_manager.get_merged_dictionary(
            doc_dictionary_path=self.doc_dictionary_path,
            selected_globals=self.selected_dictionaries
        )

        # 如果合并后的词典为空，使用默认词典
        if not merged:
            print("⚠️ 未找到全局词典，使用默认词典")
            return self._get_default_dictionary()

        print(f"📚 已加载 {len(merged)} 类实体，共 {sum(len(v) for v in merged.values())} 个词条")
        return merged

    def _get_default_dictionary(self) -> Dict[str, List[str]]:
        """获取默认词典（备用）"""
        return {
            "人名": ["张三", "李四", "王五"],
            "组织": ["公司", "集团"],
            "地点": ["北京", "上海"],
        }

    def _save_dictionary(self):
        """保存词典到文件"""
        try:
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(self.entity_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存词典失败: {e}")

    def _rebuild_index(self):
        """重建快速查找索引"""
        self.all_terms = {}  # term -> type
        for entity_type, terms in self.entity_dict.items():
            for term in terms:
                self.all_terms[term] = entity_type

    def extract(
        self,
        text: str,
        use_llm: bool = True,
        llm_client = None,
        top_k: int = 20
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """
        抽取实体和关系

        Args:
            text: 输入文本
            use_llm: 是否使用LLM补充抽取
            llm_client: LLM客户端（如Ollama）
            top_k: 最多返回的实体数

        Returns:
            (实体列表, 关系列表)
        """
        entities = []
        relations = []
        found_positions = set()

        # 1. 词典匹配（快速路径）
        dict_entities = self._extract_from_dictionary(text, found_positions)
        entities.extend(dict_entities)

        # 2. 规则匹配（补充路径）
        # 参考v0.1.3实现，识别词典中没有的实体模式
        pattern_entities = self._extract_with_patterns(text, found_positions)
        entities.extend(pattern_entities)

        # 3. LLM抽取（智能路径）
        # 大模型会从文本中自动识别实体
        if use_llm and llm_client:
            llm_entities, llm_relations = self._extract_with_llm(text, llm_client, found_positions)
            entities.extend(llm_entities)
            relations.extend(llm_relations)

            # 4. 动态学习：LLM发现的新实体加入词典
            self._learn_from_llm(llm_entities)

        # 4. 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: (e.confidence, len(e.name)), reverse=True)

        # 5. 推断共现关系（如果LLM没有提供足够关系）
        if len(relations) < 3 and len(entities) > 1:
            cooccurrence_relations = self._infer_cooccurrence_relations(entities, text)
            relations.extend(cooccurrence_relations)

        # 限制数量
        entities = entities[:top_k]
        relations = relations[:top_k]

        return entities, relations

    def extract_batch(
        self,
        chunks: List[str],
        use_llm: bool = True,
        llm_client = None,
        batch_size: int = 5,
        top_k: int = 20,
        progress_callback = None
    ) -> List[Tuple[List[ExtractedEntity], List[ExtractedRelation]]]:
        """
        批量抽取实体和关系

        将多个文本块合并，一次性调用LLM，减少API调用次数

        Args:
            chunks: 文本块列表
            use_llm: 是否使用LLM补充抽取
            llm_client: LLM客户端
            batch_size: 每批处理的块数（默认5）
            top_k: 每块最多返回的实体数
            progress_callback: 进度回调函数(batch_index, total_batches)

        Returns:
            每块的(实体列表, 关系列表)
        """
        results = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        # 按批次处理
        for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            # 如果只有一个块，直接处理
            if len(batch_chunks) == 1:
                entities, relations = self.extract(
                    batch_chunks[0],
                    use_llm=use_llm,
                    llm_client=llm_client,
                    top_k=top_k
                )
                results.append((entities, relations))
                # 调用进度回调（包含实体数）
                if progress_callback:
                    total_entities_so_far = sum(len(r[0]) for r in results)
                    progress_callback(batch_idx + 1, total_batches, batch_start, len(chunks), total_entities_so_far)
                continue

            # 多个块，批量处理
            if use_llm and llm_client:
                # 批量LLM抽取
                batch_results = self._extract_batch_with_llm(
                    batch_chunks, llm_client, top_k
                )
                results.extend(batch_results)
            else:
                # 不用LLM，逐个处理
                for chunk in batch_chunks:
                    entities, relations = self.extract(
                        chunk,
                        use_llm=False,
                        llm_client=None,
                        top_k=top_k
                    )
                    results.append((entities, relations))

            # 调用进度回调（包含实体数）
            if progress_callback:
                total_entities_so_far = sum(len(r[0]) for r in results)
                progress_callback(batch_idx + 1, total_batches, batch_start, len(chunks), total_entities_so_far)

        return results

    def _extract_batch_with_llm(
        self,
        chunks: List[str],
        llm_client,
        top_k: int = 20
    ) -> List[Tuple[List[ExtractedEntity], List[ExtractedRelation]]]:
        """
        使用LLM批量抽取多个文本块

        Args:
            chunks: 文本块列表（2-5个）
            llm_client: LLM客户端
            top_k: 每块最多返回的实体数

        Returns:
            每块的(实体列表, 关系列表)
        """
        results = []

        # 先进行词典匹配（每块独立）
        for chunk in chunks:
            entities = []
            relations = []
            found_positions = set()

            # 词典匹配
            dict_entities = self._extract_from_dictionary(chunk, found_positions)
            entities.extend(dict_entities)

            # 规则匹配
            pattern_entities = self._extract_with_patterns(chunk, found_positions)
            entities.extend(pattern_entities)

            results.append((entities, relations, found_positions, chunk))

        # 构建批量提示词
        prompt = self._build_batch_extraction_prompt(chunks)

        try:
            # 调用LLM（使用较长的超时，因为处理多个块，20个块约需3-5分钟）
            response = llm_client.generate(prompt, timeout=300, silent=True)

            if not response:
                # LLM失败，返回词典匹配结果
                return [(e, r) for e, r, _, _ in results]

            # 解析批量结果
            batch_results = self._parse_batch_llm_response(response, chunks)

            # 合并结果
            final_results = []
            for i, (dict_entities, _, found_positions, chunk) in enumerate(results):
                llm_entities = batch_results.get(i, {}).get("entities", [])
                llm_relations = batch_results.get(i, {}).get("relations", [])

                entities = list(dict_entities)  # 复制词典匹配结果

                # 处理LLM抽取的实体
                for entity_data in llm_entities:
                    name = entity_data.get("name", "")
                    if name and name not in self.all_terms:
                        for match in re.finditer(re.escape(name), chunk):
                            start, end = match.start(), match.end()
                            if not self._is_overlapping(start, end, found_positions):
                                entities.append(ExtractedEntity(
                                    name=name,
                                    type=entity_data.get("type", "未知"),
                                    start=start,
                                    end=end,
                                    confidence=entity_data.get("confidence", 0.7),
                                    source="llm"
                                ))
                                for pos in range(start, end):
                                    found_positions.add(pos)
                                break

                # 处理关系
                relations = [
                    ExtractedRelation(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        relation=r.get("relation", "RELATED"),
                        confidence=r.get("confidence", 0.6)
                    )
                    for r in llm_relations
                ]

                # 动态学习
                llm_entity_objects = [e for e in entities if e.source == "llm"]
                self._learn_from_llm(llm_entity_objects)

                # 去重和排序
                entities = self._deduplicate_entities(entities)
                entities.sort(key=lambda e: (e.confidence, len(e.name)), reverse=True)

                # 推断共现关系
                if len(relations) < 3 and len(entities) > 1:
                    cooccurrence = self._infer_cooccurrence_relations(entities, chunk)
                    relations.extend(cooccurrence)

                final_results.append((entities[:top_k], relations[:top_k]))

            return final_results

        except Exception as e:
            # 出错时返回词典匹配结果
            return [(e, r) for e, r, _, _ in results]

    def _build_batch_extraction_prompt(self, chunks: List[str]) -> str:
        """构建批量实体抽取提示词"""
        prompt_parts = [
            "你是一个专业的实体抽取助手。请从以下文本块中抽取实体和关系。",
            "",
            "要求：",
            "1. 对每个文本块分别抽取",
            "2. 实体类型包括：人名、组织、地点、病症、方剂、药物、症状、穴位、治法、法律、法规等",
            "3. 关系类型包括：组成、治疗、引起、属于、位于等",
            "4. 返回JSON格式",
            "",
            f"共 {len(chunks)} 个文本块：",
            ""
        ]

        for i, chunk in enumerate(chunks):
            prompt_parts.append(f"--- 文本块 {i+1} ---")
            prompt_parts.append(chunk[:500])  # 限制长度
            prompt_parts.append("")

        prompt_parts.extend([
            "请返回以下JSON格式：",
            "{",
            '  "results": [',
            "    {",
            '      "block_index": 0,',
            '      "entities": [',
            '        {"name": "实体名", "type": "类型", "confidence": 0.9}',
            "      ],",
            '      "relations": [',
            '        {"source": "实体1", "target": "实体2", "relation": "关系", "confidence": 0.8}',
            "      ]",
            "    }",
            "  ]",
            "}"
        ])

        return "\n".join(prompt_parts)

    def _parse_batch_llm_response(self, response: str, chunks: List[str]) -> Dict[int, Dict]:
        """解析批量LLM响应"""
        import json
        import re

        try:
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                results = data.get("results", [])

                # 转换为字典
                batch_results = {}
                for result in results:
                    block_index = result.get("block_index", 0)
                    batch_results[block_index] = {
                        "entities": result.get("entities", []),
                        "relations": result.get("relations", [])
                    }
                return batch_results
        except Exception:
            pass

        # 解析失败，返回空结果
        return {i: {"entities": [], "relations": []} for i in range(len(chunks))}

        return entities[:top_k], relations

    def _extract_from_dictionary(
        self,
        text: str,
        found_positions: Set[int]
    ) -> List[ExtractedEntity]:
        """从词典中抽取实体"""
        entities = []

        for term, entity_type in self.all_terms.items():
            for match in re.finditer(re.escape(term), text):
                start, end = match.start(), match.end()
                if not self._is_overlapping(start, end, found_positions):
                    entities.append(ExtractedEntity(
                        name=term,
                        type=entity_type,
                        start=start,
                        end=end,
                        confidence=0.95,
                        source="dictionary"
                    ))
                    for i in range(start, end):
                        found_positions.add(i)

        return entities

    def _extract_with_llm(
        self,
        text: str,
        llm_client,
        found_positions: Set[int]
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """使用LLM抽取实体和关系"""
        entities = []
        relations = []

        try:
            # 构建提示词
            prompt = self._build_extraction_prompt(text)

            # 调用LLM（使用静默模式，不打印超时错误）
            response = llm_client.generate(prompt, timeout=120, silent=True)

            # 如果LLM返回为空，静默返回（不打印错误）
            if not response:
                return entities, relations

            # 解析结果
            result = self._parse_llm_response(response)

            # 处理实体
            for entity_data in result.get("entities", []):
                name = entity_data.get("name", "")
                # 检查是否已存在或重叠
                if name and name not in self.all_terms:
                    # 找到位置
                    for match in re.finditer(re.escape(name), text):
                        start, end = match.start(), match.end()
                        if not self._is_overlapping(start, end, found_positions):
                            entities.append(ExtractedEntity(
                                name=name,
                                type=entity_data.get("type", "未知"),
                                start=start,
                                end=end,
                                confidence=entity_data.get("confidence", 0.7),
                                source="llm"
                            ))
                            for i in range(start, end):
                                found_positions.add(i)
                            break

            # 处理关系
            for relation_data in result.get("relations", []):
                relations.append(ExtractedRelation(
                    source=relation_data.get("source", ""),
                    target=relation_data.get("target", ""),
                    relation=relation_data.get("relation", "RELATED"),
                    confidence=relation_data.get("confidence", 0.6)
                ))

        except Exception as e:
            # 静默处理异常，不打印错误信息
            pass

        return entities, relations

    def _extract_with_patterns(
        self,
        text: str,
        found_positions: Set[int]
    ) -> List[ExtractedEntity]:
        """
        使用规则模式匹配抽取实体

        参考v0.1.3实现，识别词典中没有的实体模式
        添加过滤条件减少误匹配
        """
        entities = []

        # 定义规则模式（参考v0.1.3）
        patterns = {
            "人名": r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|博士|教授|经理|老师)?",
            "组织": r"[\u4e00-\u9fa5]{2,10}(?:公司|集团|大学|学院|研究所|银行|医院)?",
            "地点": r"[\u4e00-\u9fa5]{2,8}(?:省|市|区|县|镇|村|路|街)?",
        }

        # 常见姓氏（用于过滤人名）
        common_surnames = set([
            '张', '王', '李', '刘', '陈', '杨', '黄', '赵', '周', '吴',
            '徐', '孙', '马', '朱', '胡', '郭', '何', '高', '林', '罗',
            '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
            '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
            '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
        ])

        # 停用词（用于过滤）
        stop_words = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '那', '之',
            '与', '及', '等', '或', '但', '而', '因为', '所以', '如果',
            '虽然', '但是', '然后', '而且', '或者', '还是', '可以', '可能',
            '应该', '需要', '进行', '通过', '根据', '由于', '因此', '其中',
            '其他', '已经', '正在', '曾经', '现在', '当时', '这里', '那里',
            '什么', '怎么', '为什么', '如何', '谁', '哪', '个', '为', '以',
        ])

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                name = match.group()

                # 过滤条件1：长度检查
                if len(name) < 2 or len(name) > 10:
                    continue

                # 过滤条件2：停用词检查
                if any(c in stop_words for c in name):
                    continue

                # 过滤条件3：人名检查（必须以常见姓氏开头）
                if entity_type == "人名" and name[0] not in common_surnames:
                    continue

                # 过滤条件4：组织名检查（不应该包含病症相关词汇）
                if entity_type == "组织":
                    # 排除明显的非组织词汇
                    medical_terms = ['病', '症', '炎', '痛', '热', '寒', '风', '汗', '脉', '血']
                    if any(term in name for term in medical_terms):
                        continue
                    # 必须以组织后缀结尾
                    org_suffixes = ['公司', '集团', '大学', '学院', '研究所', '银行', '医院', '学校', '政府', '部门', '协会', '基金会', '联盟']
                    if not any(name.endswith(suffix) for suffix in org_suffixes):
                        continue

                # 过滤条件5：地点名检查（必须以地点后缀结尾）
                if entity_type == "地点":
                    location_suffixes = ['省', '市', '区', '县', '镇', '村', '路', '街']
                    if not any(name.endswith(suffix) for suffix in location_suffixes):
                        continue

                # 过滤条件4：检查是否已存在（避免重复）
                is_duplicate = False
                for existing in entities:
                    if existing.start == start:
                        is_duplicate = True
                        break

                # 过滤条件5：检查是否与词典匹配的结果重叠
                if is_duplicate or self._is_overlapping(start, end, found_positions):
                    continue

                entities.append(ExtractedEntity(
                    name=name,
                    type=entity_type,
                    start=start,
                    end=end,
                    confidence=0.7,
                    source="pattern"
                ))
                for i in range(start, end):
                    found_positions.add(i)

        return entities

    def extract_from_query(self, text: str) -> List[Dict[str, str]]:
        """
        从查询中提取关键词（简化格式）

        Args:
            text: 查询文本

        Returns:
            关键词字典列表
        """
        # 使用词典匹配提取关键词
        keywords = []
        found_names = set()

        for term, entity_type in self.all_terms.items():
            if term in text and term not in found_names:
                keywords.append({"name": term, "type": entity_type})
                found_names.add(term)

        return keywords

    def _build_extraction_prompt(self, text: str) -> str:
        """构建实体抽取提示词"""
        return f"""请从以下文本中抽取实体和关系。

文本：
{text}

请按JSON格式输出：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "confidence": 0.8}}
  ],
  "relations": [
    {{"source": "实体1", "target": "实体2", "relation": "关系类型", "confidence": 0.7}}
  ]
}}

注意：
1. 实体类型可以是：病名、方剂、中药、症状、治法、人名、地名、组织等
2. 关系类型可以是：治疗、导致、包含、属于等
3. confidence 范围 0-1
"""

    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"entities": [], "relations": []}

    def _learn_from_llm(self, llm_entities: List[ExtractedEntity]):
        """从LLM抽取结果中学习，更新文档私有词典"""
        if not self.doc_id:
            # 如果没有文档ID，不保存学习结果
            return

        new_entities: Dict[str, List[str]] = {}

        for entity in llm_entities:
            if entity.source == "llm" and entity.confidence > 0.7:
                # 新类型
                if entity.type not in new_entities:
                    new_entities[entity.type] = []

                # 新实体
                if entity.name not in new_entities[entity.type]:
                    new_entities[entity.type].append(entity.name)

        if new_entities:
            # 更新文档私有词典
            self.dict_manager.update_document_dictionary(
                doc_id=self.doc_id,
                new_entities=new_entities,
                persist_dir=str(self.persist_dir)
            )
            print(f"📝 文档词典已更新，新增 {sum(len(v) for v in new_entities.values())} 个实体")

    def _infer_cooccurrence_relations(
        self,
        entities: List[ExtractedEntity],
        text: str
    ) -> List[ExtractedRelation]:
        """基于共现推断关系"""
        relations = []
        entity_names = [e.name for e in entities]

        # 简单的共现关系
        for i, source in enumerate(entity_names):
            for target in entity_names[i+1:]:
                relations.append(ExtractedRelation(
                    source=source,
                    target=target,
                    relation="CO_OCCUR",
                    confidence=0.5
                ))

        return relations

    def _is_overlapping(self, start: int, end: int, found_positions: Set[int]) -> bool:
        """检查位置是否重叠"""
        for i in range(start, end):
            if i in found_positions:
                return True
        return False

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """去重"""
        seen = {}
        for e in entities:
            if e.name not in seen or e.confidence > seen[e.name].confidence:
                seen[e.name] = e
        return list(seen.values())

    # ==================== 人工干预接口 ====================

    def export_dictionary(self, filepath: str):
        """导出词典供人工编辑"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.entity_dict, f, ensure_ascii=False, indent=2)
            print(f"✅ 词典已导出: {filepath}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")

    def import_dictionary(self, filepath: str, merge: bool = True):
        """导入人工编辑后的词典"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                new_dict = json.load(f)

            if merge:
                # 合并模式
                for entity_type, terms in new_dict.items():
                    if entity_type not in self.entity_dict:
                        self.entity_dict[entity_type] = []
                    self.entity_dict[entity_type].extend(terms)
                    self.entity_dict[entity_type] = list(set(self.entity_dict[entity_type]))
            else:
                # 覆盖模式
                self.entity_dict = new_dict

            self._rebuild_index()
            self._save_dictionary()
            print(f"✅ 词典已导入，共 {len(self.all_terms)} 个实体")

        except Exception as e:
            print(f"❌ 导入失败: {e}")

    def add_entity(self, name: str, entity_type: str, save: bool = True):
        """手动添加实体"""
        if entity_type not in self.entity_dict:
            self.entity_dict[entity_type] = []

        if name not in self.entity_dict[entity_type]:
            self.entity_dict[entity_type].append(name)
            self.all_terms[name] = entity_type

            if save:
                self._save_dictionary()

    def get_statistics(self) -> Dict[str, Any]:
        """获取词典统计信息"""
        return {
            "total_entities": len(self.all_terms),
            "entity_types": list(self.entity_dict.keys()),
            "type_counts": {k: len(v) for k, v in self.entity_dict.items()},
            "dictionary_path": str(self.dictionary_path)
        }
