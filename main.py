"""
OllamaPilot - 智能助手启动入口

直接运行: python main.py

使用方法:
    python main.py                    # 交互式选择模型
    python main.py --model qwen3.5:4b # 指定模型
    python main.py --list             # 列出可用模型
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ollamapilot import init_ollama_model, create_agent, list_ollama_models


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OllamaPilot 智能助手")
    parser.add_argument("--model", "-m", type=str, help="指定模型名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="温度参数")
    parser.add_argument("--skills-dir", "-s", type=str, default="skills", help="Skill 目录")
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        models = list_ollama_models()
        if models:
            print("\n📋 Ollama 可用模型:")
            print("-" * 40)
            for model in models:
                print(f"  • {model}")
            print()
        else:
            print("\n⚠️ 未检测到 Ollama 模型或服务未启动\n")
        return
    
    # 选择模型
    if args.model:
        model_name = args.model
        print(f"✅ 使用模型: {model_name}")
    else:
        models = list_ollama_models()
        if not models:
            print("❌ 未检测到 Ollama 模型，请确保 Ollama 服务已启动")
            print("   或使用: python main.py --model <模型名称>")
            return
        
        print("\n📋 可用模型:")
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        
        try:
            choice = input(f"\n请选择 (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            model_name = models[idx] if 0 <= idx < len(models) else models[0]
        except (ValueError, IndexError):
            model_name = models[0]
        
        print(f"✅ 已选择: {model_name}")
    
    # 初始化模型
    print("\n🔄 正在初始化模型...")
    try:
        model = init_ollama_model(model_name, temperature=args.temperature)
        print("✅ 模型初始化完成")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return
    
    # 创建 Agent
    print("🔄 正在加载 Skill...")
    try:
        agent = create_agent(model, skills_dir=args.skills_dir)
        print("✅ Agent 创建完成")
    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        return
    
    # 打印欢迎信息
    print("\n" + "=" * 60)
    print("🤖 OllamaPilot 智能助手")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"温度: {args.temperature}")
    print(f"Skill 目录: {args.skills_dir}")
    print("-" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    # 对话循环
    thread_id = "main_session"
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 再见！")
                break
            
            print("🤖 思考中...")
            response = agent.invoke(user_input, thread_id=thread_id)
            print(f"\n助手: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
