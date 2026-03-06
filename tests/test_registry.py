"""
SkillRegistry 单元测试
"""

import sys
sys.path.insert(0, "..")

import unittest
from base_agent import SkillRegistry, Skill
from skills import WeatherSkill, CalculatorSkill


class TestSkillRegistry(unittest.TestCase):
    """测试 SkillRegistry"""
    
    def setUp(self):
        """测试前准备"""
        self.registry = SkillRegistry()
        self.weather_skill = WeatherSkill()
        self.calculator_skill = CalculatorSkill()
    
    def test_register(self):
        """测试注册 Skill"""
        self.registry.register(self.weather_skill)
        self.assertEqual(len(self.registry), 1)
        self.assertIn("weather", self.registry)
    
    def test_register_duplicate(self):
        """测试重复注册"""
        self.registry.register(self.weather_skill)
        
        # 重复注册相同版本应该被忽略
        self.registry.register(self.weather_skill)
        self.assertEqual(len(self.registry), 1)
    
    def test_unregister(self):
        """测试注销 Skill"""
        self.registry.register(self.weather_skill)
        self.registry.unregister("weather")
        self.assertEqual(len(self.registry), 0)
        self.assertNotIn("weather", self.registry)
    
    def test_unregister_not_exist(self):
        """测试注销不存在的 Skill"""
        with self.assertRaises(KeyError):
            self.registry.unregister("not_exist")
    
    def test_get(self):
        """测试获取 Skill"""
        self.registry.register(self.weather_skill)
        
        skill = self.registry.get("weather")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "weather")
        
        # 获取不存在的 Skill
        skill = self.registry.get("not_exist")
        self.assertIsNone(skill)
    
    def test_list_skills(self):
        """测试列出所有 Skill"""
        self.registry.register(self.weather_skill)
        self.registry.register(self.calculator_skill)
        
        skills = self.registry.list_skills()
        self.assertEqual(len(skills), 2)
        
        # 按标签筛选
        skills = self.registry.list_skills(tag="天气")
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "weather")
    
    def test_get_all_tools(self):
        """测试获取所有工具"""
        self.registry.register(self.weather_skill)
        self.registry.register(self.calculator_skill)
        
        # 获取所有工具
        tools = self.registry.get_all_tools()
        self.assertEqual(len(tools), 6)  # 2 + 4
        
        # 获取指定 Skill 的工具
        tools = self.registry.get_all_tools(["weather"])
        self.assertEqual(len(tools), 2)
        
        tools = self.registry.get_all_tools(["calculator"])
        self.assertEqual(len(tools), 4)
    
    def test_dependency_check(self):
        """测试依赖检查"""
        # 创建一个带依赖的 Skill 类
        class DependentSkill(Skill):
            name = "dependent"
            description = "依赖 Skill"
            version = "1.0.0"
            dependencies = ["weather"]
            
            def get_tools(self):
                return []
            
            def on_activate(self):
                pass
            
            def on_deactivate(self):
                pass
        
        # 先注册依赖
        self.registry.register(self.weather_skill)
        
        # 再注册依赖者
        dependent = DependentSkill()
        self.registry.register(dependent)
        
        # 尝试注销被依赖的 Skill 应该失败
        with self.assertRaises(ValueError):
            self.registry.unregister("weather")
    
    def test_clear(self):
        """测试清空"""
        self.registry.register(self.weather_skill)
        self.registry.register(self.calculator_skill)
        
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)


class TestSkillRegistryAutoLoad(unittest.TestCase):
    """测试自动加载功能"""
    
    def test_load_from_directory_not_exist(self):
        """测试加载不存在的目录"""
        registry = SkillRegistry()
        
        with self.assertRaises(FileNotFoundError):
            registry.load_from_directory("/not/exist/path")
    
    def test_load_from_directory_not_dir(self):
        """测试加载非目录路径"""
        registry = SkillRegistry()
        
        with self.assertRaises(NotADirectoryError):
            registry.load_from_directory(__file__)


if __name__ == "__main__":
    unittest.main()
