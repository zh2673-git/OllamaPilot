"""
自动加载 Skill 示例

演示如何自动扫描并加载 skills 目录下的所有 Skill
"""

import sys
sys.path.insert(0, ".")


def main():
    print("=" * 60)
    print("自动加载 Skill 示例")
    print("=" * 60)
    
    # 方法1: 自动发现并加载所有 Skill 元数据
    print("\n[方法1] 自动发现所有 Skill 元数据...")
    from base_agent import discover_skills_metadata
    
    metadata_list = discover_skills_metadata("skills")
    print(f"\n共发现 {len(metadata_list)} 个 Skill:")
    for metadata in metadata_list:
        print(f"  - {metadata.name}: {metadata.description}")
    
    # 方法2: 自动注册到 SkillRegistry
    print("\n[方法2] 自动注册到 SkillRegistry...")
    from base_agent import SkillRegistry, create_agent
    
    registry = SkillRegistry()
    print(f"\n成功创建 SkillRegistry")
    
    # 查看已注册的 Skill（通过 create_agent 会自动注册）
    print("\n已注册的 Skill:")
    for skill in registry.list_skills():
        print(f"  - {skill.name} (v{skill.version})")
    
    # 方法3: 使用 Agent
    print("\n[方法3] 使用 Agent...")
    from base_agent import init_ollama_model, create_agent
    
    # 初始化模型
    model = init_ollama_model()
    
    # 创建 Agent
    agent = create_agent(model, skills_dir="skills")
    
    print("\n✅ Agent 创建成功！")
    print("\n现在你可以与 Agent 对话，它会自动决定调用哪个 Skill。")
    print("示例：")
    print('  - "帮我搜索一下 Python 教程"')
    print('  - "用浏览器打开 GitHub"')
    print('  - "帮我创建一个自动笔记"')


if __name__ == "__main__":
    main()
