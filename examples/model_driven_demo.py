"""
Agent 架构示例

演示如何使用模型驱动的智能体
支持自动 Skill 发现和路由决策
"""

import sys
sys.path.insert(0, ".")

from base_agent import init_ollama_model, create_agent


def main():
    """Agent 示例主函数"""
    
    print("=" * 60)
    print("🤖 Agent 架构示例")
    print("=" * 60)
    
    # 1. 初始化模型
    print("\n🔄 初始化模型...")
    model = init_ollama_model(
        model="qwen3.5:9b",
        temperature=0.7
    )
    
    # 2. 创建智能体
    print("🔄 创建智能体...")
    agent = create_agent(
        model=model,
        skills_dir="skills"  # Skill 目录
    )
    
    print("\n" + "=" * 60)
    print("💡 示例对话：")
    print("-" * 60)
    
    # 示例 1: 触发自动笔记 Skill
    print("\n📝 示例 1: 创建笔记")
    print("用户: 帮我创建一个关于 Python 的笔记")
    response = agent.invoke("帮我创建一个关于 Python 的笔记", verbose=True)
    print(f"\n助手: {response}")
    
    # 示例 2: 触发浏览器 Skill
    print("\n" + "-" * 60)
    print("\n🌐 示例 2: 浏览器操作")
    print("用户: 打开浏览器访问 baidu.com")
    response = agent.invoke("打开浏览器访问 baidu.com", verbose=True)
    print(f"\n助手: {response}")
    
    # 示例 3: 触发文件系统 Skill
    print("\n" + "-" * 60)
    print("\n📁 示例 3: 文件操作")
    print("用户: 列出当前目录的文件")
    response = agent.invoke("列出当前目录的文件", verbose=True)
    print(f"\n助手: {response}")


if __name__ == "__main__":
    main()
