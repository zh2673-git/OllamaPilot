"""
混合模式实体抽取器

结合词典匹配 + LLM抽取，支持动态学习和人工干预
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
    1. 词典匹配（快速、高置信度）
    2. LLM抽取（智能、发现新实体）
    3. 合并结果（去重、排序）
    4. 动态学习（新实体加入词典）

    支持人工干预：
    - 导出词典 → 用更强模型优化 → 导入更新
    """

    def __init__(self, persist_dir: str = "./data/graphrag"):
        """
        初始化抽取器

        Args:
            persist_dir: 词典持久化目录
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dictionary_path = self.persist_dir / "entity_dictionary.json"

        # 停用词
        self.stopwords = self._load_stopwords()

        # 加载或初始化词典
        self.entity_dict = self._load_dictionary()

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

    def _load_dictionary(self) -> Dict[str, List[str]]:
        """从文件加载词典，不存在则初始化"""
        if self.dictionary_path.exists():
            try:
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载词典失败: {e}，使用默认词典")

        # 默认词典（通用实体类型）
        # 包含跨领域的通用实体，确保即使LLM失败也能抽取实体
        return {
            "人名": [
                # 常见姓氏 + 常见名字
                "张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十",
                "张伟", "李娜", "王强", "刘洋", "陈明", "杨华", "黄丽", "赵军",
            ],
            "组织": [
                # 常见组织类型
                "公司", "集团", "大学", "学院", "研究所", "医院", "银行",
                "学校", "医院", "政府", "部门", "协会", "基金会", "联盟",
            ],
            "地点": [
                # 常见地点
                "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉",
                "中国", "美国", "日本", "德国", "法国", "英国", "俄罗斯",
                "亚洲", "欧洲", "北美", "非洲", "大洋洲",
            ],
            "时间": [
                # 时间单位
                "年", "月", "日", "时", "分", "秒", "周", "季度", "世纪",
                "春天", "夏天", "秋天", "冬天", "上午", "下午", "晚上",
            ],
            "数值": [
                # 数值单位
                "个", "十", "百", "千", "万", "亿", "兆",
                "米", "千米", "克", "千克", "升", "毫升", "度", "百分比",
            ],
            "概念": [],
            "事件": [],
            "产品": [],
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

        # 2. LLM抽取（智能路径）
        if use_llm and llm_client:
            llm_entities, llm_relations = self._extract_with_llm(text, llm_client, found_positions)
            entities.extend(llm_entities)
            relations.extend(llm_relations)

            # 3. 动态学习：LLM发现的新实体加入词典
            self._learn_from_llm(llm_entities)

        # 4. 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: (e.confidence, len(e.name)), reverse=True)

        # 5. 推断共现关系（如果LLM没有提供足够关系）
        if len(relations) < 3 and len(entities) > 1:
            cooccurrence_relations = self._infer_cooccurrence_relations(entities, text)
            relations.extend(cooccurrence_relations)

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

            # 调用LLM
            response = llm_client.generate(prompt)

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
            print(f"⚠️ LLM抽取失败: {e}")

        return entities, relations

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
        """从LLM抽取结果中学习，更新词典"""
        updated = False

        for entity in llm_entities:
            if entity.source == "llm" and entity.confidence > 0.7:
                # 新类型
                if entity.type not in self.entity_dict:
                    self.entity_dict[entity.type] = []

                # 新实体
                if entity.name not in self.entity_dict[entity.type]:
                    self.entity_dict[entity.type].append(entity.name)
                    self.all_terms[entity.name] = entity.type
                    updated = True

        if updated:
            self._save_dictionary()
            print(f"📝 词典已更新，当前包含 {len(self.all_terms)} 个实体")

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
