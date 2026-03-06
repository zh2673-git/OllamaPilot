"""
Agent 启动入口 - 模型驱动架构

直接运行: python main.py
即可启动交互式对话

使用方法:
  python main.py              # 交互式选择模型
  python main.py --model qwen3.5:9b  # 指定模型
  python main.py --list       # 列出可用模型
"""

import sys
import argparse
sys.path.insert(0, ".")

from base_agent import (
    create_agent,
    select_model,
    get_ollama_models,
    quick_select_model,
    create_ollama_model,  # 使用原生 Ollama 客户端
)


def main():
    """主函数 - 启动模型驱动的交互式 Agent"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="智能助手 - 模型驱动架构")
    parser.add_argument("--model", "-m", type=str, help="指定模型名称 (如: qwen3.5:9b, glm-4.7-flash)")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型并退出")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="模型温度参数 (默认: 0.7)")
    args = parser.parse_args()
    
    # 如果指定了 --list，列出模型并退出
    if args.list:
        models = get_ollama_models()
        if models:
            print("\n📋 Ollama 可用模型:")
            print("-" * 40)
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            print()
        else:
            print("\n⚠️ 未检测到 Ollama 模型或服务未启动\n")
        return
    
    # 选择模型
    if args.model:
        selected_model = quick_select_model(args.model)
    else:
        selected_model = select_model()
    
    print("=" * 60)
    print("🤖 智能助手已启动 (模型驱动架构)")
    print("=" * 60)
    print(f"模型: Ollama ({selected_model})")
    print("架构: 基座Agent + Skill (模型自主决策)")
    print("-" * 60)
    
    # 初始化模型（使用原生 Ollama 客户端，保留思维链）
    print("\n🔄 正在初始化模型...")
    model = create_ollama_model(
        model=selected_model,
        temperature=args.temperature,
        use_native=True  # 使用原生 API，保留完整的思维链输出
    )
    
    # 创建模型驱动的智能体
    print("🔄 正在加载 Skill 元数据...")
    agent = create_agent(model, skills_dir="skills")
    
    print("\n" + "=" * 60)
    print("💬 开始对话:\n")
    print("提示: 系统会根据你的输入自动决定调用哪个 Skill\n")
    
    # 是否显示详细过程
    verbose = True
    
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 切换详细模式
            if user_input.lower() == 'verbose on':
                verbose = True
                print("\n✅ 已开启详细输出\n")
                continue
            elif user_input.lower() == 'verbose off':
                verbose = False
                print("\n✅ 已关闭详细输出\n")
                continue
            
            # 执行请求
            response = agent.invoke(user_input, verbose=verbose)
            print()  # 空行分隔
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
