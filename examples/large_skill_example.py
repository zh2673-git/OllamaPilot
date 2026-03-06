"""
超大 Skill 处理示例

展示如何使用自适应加载器处理 4万字+ 的超大 Skill
完美适配 Ollama 本地小模型
"""

import sys
sys.path.insert(0, 'd:\\self_test\\project\\trae\\agent\\baseagent+skill')

from base_agent import (
    Agent,
    create_agent,
    AgentConfig,
    AdaptiveSkillLoader,
    SkillChunker,
    init_ollama_model,
)


def demo_chunker():
    """演示 Skill 分块功能"""
    print("=" * 60)
    print("📦 Skill 分块演示")
    print("=" * 60)
    
    # 模拟一个 4万字的超大 Skill
    large_skill_content = """
# 超大 Skill 示例

## 概述
这是一个超大 Skill 的示例内容，用于演示分块加载功能。

## 功能模块 A
这是功能模块 A 的详细说明...
""" + "这是详细内容。" * 5000 + """

## 功能模块 B
这是功能模块 B 的详细说明...
""" + "这是更多详细内容。" * 5000 + """

## 功能模块 C
这是功能模块 C 的详细说明...
""" + "这是更多详细内容。" * 5000 + """

## 故障排除
常见问题解决方案...
""" + "这是故障排除内容。" * 2000 + """

## 高级用法
高级功能和技巧...
""" + "这是高级用法内容。" * 2000
    
    print(f"原始 Skill 大小: {len(large_skill_content)} 字符")
    print()
    
    # 创建分块器
    chunker = SkillChunker()
    
    # 处理 Skill
    chunks = chunker.process_skill("large_skill", large_skill_content)
    
    # 显示分块信息
    summary = chunker.get_chunk_summary("large_skill")
    print(f"分块数量: {summary['total_chunks']}")
    print(f"总字符数: {summary['total_chars']}")
    print(f"平均块大小: {summary['avg_chunk_size']} 字符")
    print(f"章节: {', '.join(summary['sections'])}")
    print()
    
    # 模拟用户查询
    queries = [
        "如何使用功能模块 A",
        "功能模块 B 的配置方法",
        "遇到错误怎么解决",
    ]
    
    for query in queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 40)
        
        # 检索相关块
        matches = chunker.retrieve_relevant_chunks("large_skill", query, top_k=2)
        
        print(f"找到 {len(matches)} 个相关片段:")
        for i, match in enumerate(matches, 1):
            print(f"  [{i}] 章节: {match.chunk.section}")
            print(f"      匹配分数: {match.score:.2f}")
            print(f"      匹配关键词: {', '.join(match.matched_keywords[:3])}")
            print(f"      内容预览: {match.chunk.content[:80]}...")
    
    return chunker


def demo_adaptive_loader():
    """演示自适应加载器"""
    print("\n" + "=" * 60)
    print("🎯 自适应加载演示")
    print("=" * 60)
    
    # 不同大小的 Skill
    skills = {
        "small": "这是一个小 Skill，只有几百字。" * 10,
        "medium": "这是一个中等 Skill。" * 500,
        "large": "这是一个大 Skill。" * 2000,
        "huge": "这是一个超大 Skill（4万字）。" * 10000,
    }
    
    loader = AdaptiveSkillLoader()
    
    for name, content in skills.items():
        info = loader.get_loading_info(name, content)
        print(f"\n📄 {name}:")
        print(f"   大小: {info['total_chars']} 字符")
        print(f"   策略: {info['strategy']}")
        print(f"   估计 tokens: {info['estimated_tokens']}")
        print(f"   适合小模型: {'✅' if info['suitable_for_small_model'] else '❌'}")
    
    # 演示检索加载
    print("\n" + "-" * 40)
    print("🧠 智能检索加载演示")
    print("-" * 40)
    
    huge_skill = skills["huge"]
    query = "如何使用 large 功能"
    
    print(f"\n查询: {query}")
    print(f"原始内容: {len(huge_skill)} 字符")
    
    # 加载相关内容
    loaded = loader.load("huge", huge_skill, query)
    print(f"加载内容: {len(loaded)} 字符")
    print(f"压缩率: {len(loaded)/len(huge_skill)*100:.1f}%")
    print("\n加载的内容片段:")
    print(loaded[:500] + "...")
    
    return loader


def demo_with_ollama():
    """演示与 Ollama 集成"""
    print("\n" + "=" * 60)
    print("🦙 Ollama 集成演示")
    print("=" * 60)
    
    # 初始化 Ollama 模型（小模型）
    print("\n正在初始化 Ollama 模型...")
    try:
        model = init_ollama_model(
            model_name="qwen2.5:3b",  # 3B 小模型
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"⚠️ 模型初始化失败: {e}")
        print("请确保 Ollama 已启动并安装了 qwen2.5:3b 模型")
        return
    
    # 创建配置（小模型优化）
    config = AgentConfig(
        use_manual_tool_parsing=True,  # 手工解析工具调用
        max_iterations=3,              # 减少迭代次数
        skill_load_strategy="gradual", # 使用自适应加载
        enable_self_assessment=False,  # 关闭自评估
    )
    
    # 创建 Agent
    print("\n创建 Agent...")
    agent = create_agent(
        model=model,
        skills_dir="../skills",
        config=config,
    )
    print("✅ Agent 创建成功")
    
    # 测试查询
    test_queries = [
        "你好",
        "帮我查一下天气",
    ]
    
    print("\n" + "-" * 40)
    for query in test_queries:
        print(f"\n📝 测试: {query}")
        try:
            result = agent.invoke(query, verbose=True)
            print(f"\n💬 回复: {result[:200]}...")
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("超大 Skill 处理 + Ollama 小模型优化示例")
    print("🚀" * 30 + "\n")
    
    # 1. 演示分块功能
    demo_chunker()
    
    # 2. 演示自适应加载
    demo_adaptive_loader()
    
    # 3. 演示 Ollama 集成
    # demo_with_ollama()  # 取消注释以运行实际测试
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print("\n核心优化点:")
    print("1. 超大 Skill 自动分块（最大1500字符/块）")
    print("2. 智能检索：根据查询加载最相关的片段")
    print("3. 自适应策略：小Skill完整加载，大Skill检索加载")
    print("4. 完美适配 Ollama 3B-7B 小模型")


if __name__ == "__main__":
    main()
