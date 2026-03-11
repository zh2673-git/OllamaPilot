"""
增强搜索测试

测试 EnhancedSearchSkill 和各搜索引擎
"""

import pytest
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.enhanced_search import EnhancedSearchSkill
from skills.enhanced_search.engines import (
    SearchEngineFactory,
    ArXivSearchEngine,
    WikipediaSearchEngine,
    BaiduBaikeSearchEngine,
    PubMedSearchEngine,
    GitHubSearchEngine,
    GiteeSearchEngine,
)


class TestEnhancedSearchSkill:
    """测试增强搜索 Skill"""
    
    @pytest.fixture
    def skill(self):
        return EnhancedSearchSkill()
    
    def test_skill_metadata(self, skill):
        """测试 Skill 元数据"""
        assert skill.name == "enhanced_search"
        assert "增强搜索" in skill.description
        assert len(skill.triggers) > 0
        
    def test_triggers_distinct_from_web_search(self, skill):
        """测试触发词与 web_search 不重叠"""
        web_search_triggers = {"搜索", "查询", "网上", "网络", "查找", "资料"}
        skill_triggers = set(skill.triggers)
        
        # 应该没有重叠，或者重叠很少
        overlap = web_search_triggers & skill_triggers
        assert len(overlap) <= 1, f"触发词重叠过多: {overlap}"
        
    def test_get_tools(self, skill):
        """测试工具列表"""
        tools = skill.get_tools()
        assert len(tools) >= 4
        
        tool_names = [t.name for t in tools]
        assert "academic_search" in tool_names
        assert "code_search" in tool_names
        assert "encyclopedia_search" in tool_names
        assert "multi_engine_search" in tool_names

    def test_get_available_engines(self, skill):
        """测试获取可用引擎"""
        engines = skill.get_available_engines()
        assert isinstance(engines, dict)
        assert len(engines) > 0


class TestSearchEngines:
    """测试各搜索引擎"""
    
    @pytest.mark.asyncio
    async def test_arxiv_engine(self):
        """测试 arXiv 引擎"""
        engine = ArXivSearchEngine()
        
        # 测试可用性
        is_available = engine.is_available()
        print(f"arXiv 可用: {is_available}")
        
        if is_available:
            results = await engine.search("quantum computing", num_results=3)
            assert len(results) > 0
            assert all(r.source == "arxiv" for r in results)
            print(f"arXiv 搜索结果: {len(results)} 条")
        else:
            pytest.skip("arXiv 不可用")
    
    @pytest.mark.asyncio
    async def test_wikipedia_engine(self):
        """测试 Wikipedia 引擎"""
        engine = WikipediaSearchEngine()
        
        is_available = engine.is_available()
        print(f"Wikipedia 可用: {is_available}")
        
        if is_available:
            results = await engine.search("Python programming", num_results=3)
            assert len(results) > 0
            print(f"Wikipedia 搜索结果: {len(results)} 条")
        else:
            pytest.skip("Wikipedia 不可用")
    
    @pytest.mark.asyncio
    async def test_baidu_baike_engine(self):
        """测试百度百科引擎"""
        engine = BaiduBaikeSearchEngine()
        
        is_available = engine.is_available()
        print(f"百度百科 可用: {is_available}")
        
        if is_available:
            results = await engine.search("人工智能", num_results=3)
            assert len(results) > 0
            print(f"百度百科 搜索结果: {len(results)} 条")
        else:
            pytest.skip("百度百科 不可用")
    
    @pytest.mark.asyncio
    async def test_pubmed_engine(self):
        """测试 PubMed 引擎"""
        engine = PubMedSearchEngine()
        
        is_available = engine.is_available()
        print(f"PubMed 可用: {is_available}")
        
        if is_available:
            results = await engine.search("cancer treatment", num_results=3)
            assert len(results) > 0
            print(f"PubMed 搜索结果: {len(results)} 条")
        else:
            pytest.skip("PubMed 不可用")
    
    @pytest.mark.asyncio
    async def test_github_engine(self):
        """测试 GitHub 引擎"""
        engine = GitHubSearchEngine()
        
        is_available = engine.is_available()
        print(f"GitHub 可用: {is_available}")
        
        if is_available:
            results = await engine.search("machine learning", num_results=3)
            assert len(results) > 0
            print(f"GitHub 搜索结果: {len(results)} 条")
        else:
            pytest.skip("GitHub 不可用")
    
    @pytest.mark.asyncio
    async def test_gitee_engine(self):
        """测试 Gitee 引擎"""
        engine = GiteeSearchEngine()
        
        is_available = engine.is_available()
        print(f"Gitee 可用: {is_available}")
        
        if is_available:
            results = await engine.search("python", num_results=3)
            assert len(results) > 0
            print(f"Gitee 搜索结果: {len(results)} 条")
        else:
            pytest.skip("Gitee 不可用")


class TestSearchEngineFactory:
    """测试搜索引擎工厂"""
    
    def test_list_engines(self):
        """测试列出引擎"""
        engines = SearchEngineFactory.list_engines()
        assert len(engines) > 0
        print(f"已注册引擎: {engines}")
    
    def test_list_by_category(self):
        """测试按类别列出引擎"""
        academic = SearchEngineFactory.list_engines(category="academic")
        code = SearchEngineFactory.list_engines(category="code")
        encyclopedia = SearchEngineFactory.list_engines(category="encyclopedia")
        
        print(f"学术引擎: {academic}")
        print(f"代码引擎: {code}")
        print(f"百科引擎: {encyclopedia}")
    
    def test_create_engine(self):
        """测试创建引擎"""
        engine = SearchEngineFactory.create("arxiv")
        assert engine is not None
        assert engine.name == "arxiv"


class TestAcademicSearch:
    """测试学术搜索功能"""
    
    def test_academic_search(self):
        """测试学术搜索工具"""
        skill = EnhancedSearchSkill()
        
        # 使用同步方式测试
        result = skill._academic_search("machine learning", num_results=5)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"学术搜索结果:\n{result[:500]}")


class TestCodeSearch:
    """测试代码搜索功能"""
    
    def test_code_search(self):
        """测试代码搜索工具"""
        skill = EnhancedSearchSkill()
        
        result = skill._code_search("python web framework", num_results=5)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"代码搜索结果:\n{result[:500]}")


class TestEncyclopediaSearch:
    """测试百科搜索功能"""
    
    def test_encyclopedia_search(self):
        """测试百科搜索工具"""
        skill = EnhancedSearchSkill()
        
        result = skill._encyclopedia_search("人工智能", num_results=5)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"百科搜索结果:\n{result[:500]}")


if __name__ == "__main__":
    # 运行简单测试
    print("=" * 50)
    print("增强搜索测试")
    print("=" * 50)
    
    # 测试 Skill 初始化
    skill = EnhancedSearchSkill()
    print(f"\n✅ Skill 初始化成功: {skill.name}")
    print(f"   描述: {skill.description}")
    print(f"   触发词数量: {len(skill.triggers)}")
    
    # 测试工具
    tools = skill.get_tools()
    print(f"\n✅ 工具数量: {len(tools)}")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:50]}...")
    
    # 测试引擎可用性
    print("\n" + "=" * 50)
    print("引擎可用性检查")
    print("=" * 50)
    
    engines = skill.get_available_engines()
    for name, available in engines.items():
        status = "✅" if available else "❌"
        print(f"{status} {name}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
