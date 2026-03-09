"""
OllamaPilot - 智能助手启动入口

直接运行: python main.py

使用方法:
    python main.py                    # 交互式选择模型
    python main.py --model qwen3.5:4b # 指定对话模型
    python main.py --list             # 列出可用模型
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ollamapilot import (
    init_ollama_model,
    create_agent,
    list_ollama_models,
    list_ollama_chat_models,
    list_ollama_embedding_models,
    select_model_interactive,
)


def select_models_interactive():
    """交互式选择对话模型和 Embedding 模型"""
    # 选择对话模型
    chat_models = list_ollama_chat_models()
    if not chat_models:
        print("⚠️ 未检测到对话模型")
        return None, None

    print("\n📋 可用对话模型:")
    for i, model in enumerate(chat_models, 1):
        print(f"  {i}. {model}")

    try:
        choice = input(f"\n请选择对话模型 (1-{len(chat_models)}): ").strip()
        idx = int(choice) - 1
        chat_model = chat_models[idx] if 0 <= idx < len(chat_models) else chat_models[0]
        print(f"✅ 已选择对话模型: {chat_model}")
    except (ValueError, IndexError):
        chat_model = chat_models[0]
        print(f"✅ 默认使用对话模型: {chat_model}")

    # 选择 Embedding 模型
    embedding_models = list_ollama_embedding_models()
    if not embedding_models:
        print("⚠️ 未检测到 Embedding 模型，GraphRAG 将使用默认配置")
        return chat_model, None

    print("\n📋 可用 Embedding 模型:")
    for i, model in enumerate(embedding_models, 1):
        print(f"  {i}. {model}")

    try:
        choice = input(f"\n请选择 Embedding 模型 (1-{len(embedding_models)}, 直接回车跳过): ").strip()
        if choice == "":
            embedding_model = None
            print("⚠️ 跳过 Embedding 模型选择")
        else:
            idx = int(choice) - 1
            embedding_model = embedding_models[idx] if 0 <= idx < len(embedding_models) else embedding_models[0]
            print(f"✅ 已选择 Embedding 模型: {embedding_model}")
    except (ValueError, IndexError):
        embedding_model = embedding_models[0]
        print(f"✅ 默认使用 Embedding 模型: {embedding_model}")

    return chat_model, embedding_model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OllamaPilot 智能助手")
    parser.add_argument("--model", "-m", type=str, help="对话模型名称")
    parser.add_argument("--embedding-model", "-e", type=str, help="Embedding模型名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="温度参数")
    parser.add_argument("--skills-dir", "-s", type=str, default="skills", help="Skill 目录")
    args = parser.parse_args()

    # 列出模型
    if args.list:
        print("\n📋 Ollama 可用模型:\n")

        chat_models = list_ollama_chat_models()
        if chat_models:
            print("对话模型:")
            for model in chat_models:
                print(f"  • {model}")

        embedding_models = list_ollama_embedding_models()
        if embedding_models:
            print("\nEmbedding模型:")
            for model in embedding_models:
                print(f"  • {model}")

        if not chat_models and not embedding_models:
            print("⚠️ 未检测到 Ollama 模型或服务未启动\n")
        return

    # 选择模型
    chat_model = None
    embedding_model = None

    if args.model and args.embedding_model:
        # 命令行指定了两种模型
        chat_model = args.model
        embedding_model = args.embedding_model
        print(f"✅ 使用对话模型: {chat_model}")
        print(f"✅ 使用 Embedding 模型: {embedding_model}")
    elif args.model:
        # 只指定了对话模型，交互式选择 Embedding
        chat_model = args.model
        print(f"✅ 使用对话模型: {chat_model}")

        embedding_models = list_ollama_embedding_models()
        if embedding_models:
            print("\n📋 可用 Embedding 模型:")
            for i, m in enumerate(embedding_models, 1):
                print(f"  {i}. {m}")
            try:
                choice = input(f"\n请选择 (1-{len(embedding_models)}, 直接回车跳过): ").strip()
                if choice:
                    idx = int(choice) - 1
                    embedding_model = embedding_models[idx] if 0 <= idx < len(embedding_models) else embedding_models[0]
                    print(f"✅ 已选择: {embedding_model}")
            except (ValueError, IndexError):
                pass
        if not embedding_model:
            print("⚠️ 未选择 Embedding 模型")
    else:
        # 完全交互式选择
        chat_model, embedding_model = select_models_interactive()
        if not chat_model:
            print("❌ 未选择有效的对话模型")
            return

    # 初始化模型
    print("\n🔄 正在初始化模型...")
    try:
        model = init_ollama_model(chat_model, temperature=args.temperature)
        print("✅ 对话模型初始化完成")
    except Exception as e:
        print(f"❌ 对话模型初始化失败: {e}")
        return

    # 创建 Agent（传入 embedding_model 配置）
    print("🔄 正在加载 Skill...")
    try:
        agent_kwargs = {
            "skills_dir": args.skills_dir,
        }

        # 如果有 Embedding 模型，传递给 Agent
        if embedding_model:
            agent_kwargs["embedding_model"] = embedding_model

        agent = create_agent(model, **agent_kwargs)
        print("✅ Agent 创建完成")
    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        return

    # 打印欢迎信息
    mode = "GraphRAG模式" if embedding_model else "标准模式"
    print("\n" + "=" * 60)
    print(f"🤖 OllamaPilot 智能助手 ({mode})")
    print("=" * 60)
    print(f"对话模型: {chat_model}")
    print(f"Embedding模型: {embedding_model or '未配置'}")
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

            print("\n助手: ", end="", flush=True)

            # 使用流式输出
            full_response = ""
            try:
                for chunk in agent.stream(user_input, thread_id=thread_id):
                    # 提取 chunk 中的内容
                    if isinstance(chunk, dict):
                        # 处理不同格式的 chunk
                        if "messages" in chunk:
                            messages = chunk["messages"]
                            if messages and hasattr(messages[-1], "content"):
                                content = messages[-1].content
                                # 只输出新增的部分
                                if len(content) > len(full_response):
                                    new_text = content[len(full_response):]
                                    print(new_text, end="", flush=True)
                                    full_response = content
                        elif "content" in chunk:
                            new_text = chunk["content"]
                            print(new_text, end="", flush=True)
                            full_response += new_text
                        elif "agent" in chunk and "messages" in chunk["agent"]:
                            messages = chunk["agent"]["messages"]
                            if messages and hasattr(messages[-1], "content"):
                                content = messages[-1].content
                                if len(content) > len(full_response):
                                    new_text = content[len(full_response):]
                                    print(new_text, end="", flush=True)
                                    full_response = content
                print("\n")  # 输出结束后换行
            except Exception as e:
                # 如果流式输出失败，回退到普通 invoke
                print(f"\n⚠️ 流式输出失败，使用普通模式: {e}\n")
                response = agent.invoke(user_input, thread_id=thread_id)
                print(f"助手: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
