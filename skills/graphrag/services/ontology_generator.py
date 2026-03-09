"""
本体生成器

使用 LLM 从文档中生成实体类型和关系类型定义
"""

from typing import Dict, List, Optional, Any
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


class OntologyGenerator:
    """
    本体生成器

    使用 LLM 从文档中生成实体类型和关系类型定义
    """

    SYSTEM_PROMPT = """你是一个知识图谱本体设计专家。

你的任务是分析文档内容，设计适合问答系统的实体类型和关系类型。

要求：
1. 实体类型必须是具体可识别的（人物、组织、地点、产品、概念等）
2. 关系类型描述实体间的联系（工作于、位于、拥有、属于等）
3. 最多10个实体类型，6-10个关系类型
4. 最后2个实体类型必须是：Person 和 Organization（兜底类型）

返回严格的 JSON 格式：
{
    "entity_types": [
        {
            "name": "实体类型名（英文）",
            "description": "描述",
            "examples": ["示例1", "示例2"]
        }
    ],
    "relation_types": [
        {
            "name": "关系名（英文大写）",
            "description": "描述",
            "examples": [["实体1", "实体2"], ["实体3", "实体4"]]
        }
    ],
    "analysis_summary": "文档内容摘要"
}"""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        初始化生成器

        Args:
            llm: LLM 客户端（可以是 4B 模型），如果为None则使用默认本体
        """
        self.llm = llm

    def generate(
        self,
        document_texts: List[str],
        max_text_length: int = 10000
    ) -> Dict[str, Any]:
        """
        生成本体定义

        Args:
            document_texts: 文档文本列表
            max_text_length: 最大处理文本长度

        Returns:
            本体定义字典
        """
        if not self.llm:
            return self._get_default_ontology()

        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        if len(combined_text) > max_text_length:
            combined_text = combined_text[:max_text_length] + "..."

        # 构建消息
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"请分析以下文档，生成本体定义：\n\n{combined_text}")
        ]

        # 调用 LLM
        try:
            response = self.llm.invoke(messages)

            # 解析 JSON
            content = response.content if hasattr(response, 'content') else str(response)

            # 提取 JSON 部分
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            ontology = json.loads(content.strip())
            return self._validate_ontology(ontology)
        except Exception as e:
            print(f"⚠️ 本体生成失败: {e}，使用默认本体")
            return self._get_default_ontology()

    def _validate_ontology(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """验证和修复本体定义"""
        # 确保有 entity_types
        if "entity_types" not in ontology or not ontology["entity_types"]:
            ontology["entity_types"] = []

        # 确保有 relation_types
        if "relation_types" not in ontology:
            ontology["relation_types"] = []

        # 添加兜底类型
        entity_names = [e["name"] for e in ontology["entity_types"]]
        if "Person" not in entity_names:
            ontology["entity_types"].append({
                "name": "Person",
                "description": "人物",
                "examples": ["张三", "李四"]
            })
        if "Organization" not in entity_names:
            ontology["entity_types"].append({
                "name": "Organization",
                "description": "组织",
                "examples": ["腾讯", "阿里巴巴"]
            })

        return ontology

    def _get_default_ontology(self) -> Dict[str, Any]:
        """获取默认本体定义"""
        return {
            "entity_types": [
                {"name": "Person", "description": "人物", "examples": ["张三"]},
                {"name": "Organization", "description": "组织", "examples": ["腾讯"]},
                {"name": "Location", "description": "地点", "examples": ["北京"]},
                {"name": "Product", "description": "产品", "examples": ["iPhone"]},
                {"name": "Concept", "description": "概念", "examples": ["人工智能"]}
            ],
            "relation_types": [
                {"name": "WORKS_AT", "description": "工作于", "examples": [["张三", "腾讯"]]},
                {"name": "LOCATED_IN", "description": "位于", "examples": [["腾讯", "深圳"]]},
                {"name": "PART_OF", "description": "属于", "examples": [["产品部", "腾讯"]]}
            ],
            "analysis_summary": "使用默认本体定义"
        }
