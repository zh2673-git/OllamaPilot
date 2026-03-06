"""
Skill 单元测试

测试 Skill 基础功能、注册中心和中间件
"""

import sys
sys.path.insert(0, "..")

import unittest
from base_agent.skill import Skill, SkillMetadata
from base_agent.registry import SkillRegistry
from base_agent.middleware import AgentMiddleware, AgentState


class TestSkillMetadata(unittest.TestCase):
    """测试 SkillMetadata"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        metadata = SkillMetadata(
            name="test",
            description="测试 Skill"
        )
        self.assertEqual(metadata.name, "test")
        self.assertEqual(metadata.description, "测试 Skill")
        self.assertEqual(metadata.version, "1.0.0")  # 默认值
        self.assertEqual(metadata.tags, [])  # 默认值
    
    def test_full_creation(self):
        """测试完整创建"""
        metadata = SkillMetadata(
            name="test",
            description="测试 Skill",
            tags=["测试", "示例"],
            version="2.0.0",
            author="Tester",
            dependencies=["dep1", "dep2"]
        )
        self.assertEqual(metadata.name, "test")
        self.assertEqual(metadata.tags, ["测试", "示例"])
        self.assertEqual(metadata.version, "2.0.0")
        self.assertEqual(metadata.author, "Tester")
        self.assertEqual(metadata.dependencies, ["dep1", "dep2"])


class TestSkillRegistry(unittest.TestCase):
    """测试 SkillRegistry"""
    
    def setUp(self):
        """测试前准备"""
        self.registry = SkillRegistry()
    
    def test_register_and_get(self):
        """测试注册和获取"""
        from base_agent.skill import Skill
        
        class TestSkill(Skill):
            @property
            def name(self):
                return "test_skill"
            
            @property
            def description(self):
                return "测试 Skill"
        
        skill = TestSkill()
        self.registry.register(skill)
        
        # 验证注册成功
        self.assertEqual(len(self.registry.list_skills()), 1)
        self.assertIsNotNone(self.registry.get("test_skill"))
    
    def test_get_nonexistent(self):
        """测试获取不存在的 Skill"""
        result = self.registry.get("nonexistent")
        self.assertIsNone(result)


class TestMiddleware(unittest.TestCase):
    """测试中间件系统"""
    
    def test_agent_state_creation(self):
        """测试 AgentState 创建"""
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content="测试")]
        state = AgentState(messages=messages)
        
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.messages[0].content, "测试")
    
    def test_middleware_chain(self):
        """测试中间件链"""
        from base_agent.middleware import MiddlewareChain
        
        chain = MiddlewareChain()
        
        # 创建简单的测试中间件
        class TestMiddleware(AgentMiddleware):
            def before_model(self, state, config=None):
                state["test_key"] = "test_value"
                return {"test_key": "test_value"}
        
        middleware = TestMiddleware()
        chain.add(middleware)
        
        # 测试执行
        from langchain_core.messages import HumanMessage
        state = AgentState(messages=[HumanMessage(content="测试")])
        result = chain.execute_before_model(state)
        
        self.assertEqual(result.get("test_key"), "test_value")


class TestSkillDiscovery(unittest.TestCase):
    """测试 Skill 发现功能"""
    
    def test_discover_skills_metadata(self):
        """测试发现 Skill 元数据"""
        from base_agent import discover_skills_metadata
        
        metadata_list = discover_skills_metadata("../skills")
        
        # 验证发现了一些 Skill
        self.assertGreater(len(metadata_list), 0)
        
        # 验证每个元数据都有必需的字段
        for metadata in metadata_list:
            self.assertIsNotNone(metadata.name)
            self.assertIsNotNone(metadata.description)


if __name__ == "__main__":
    unittest.main()
