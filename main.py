"""
OllamaPilot - 智能助手启动入口

直接运行: python main.py

使用方法:
    python main.py                    # 使用 .env 配置（推荐）
    python main.py --model qwen3.5:4b # 指定对话模型（覆盖.env）
    python main.py --list             # 列出可用模型
    python main.py --interactive      # 交互式选择模型

交互命令 (在对话中输入):
    /help              显示帮助信息
    /model             列出并切换对话模型
    /model <name>      直接切换到指定模型
    /new               开始新对话
    /sessions          列出所有会话
    /switch <id>       切换到指定会话
    /clear             清空当前对话历史
    /embedding         切换 Embedding 模型
    /index [path]      手动索引文档或文件夹（默认：knowledge_base）
    /reload            重新加载 .env 配置
    /resume            恢复失败的索引任务
    quit/exit          退出程序
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ollamapilot import (
    list_ollama_chat_models,
    list_ollama_embedding_models,
)
from ollamapilot.config import get_config
from ollamapilot.cli import OllamaPilotChat

# 加载配置
config = get_config()


def select_models_interactive() -> Tuple[Optional[str], Optional[str]]:
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
    parser.add_argument("--model", "-m", type=str, help="对话模型名称（覆盖.env）")
    parser.add_argument("--embedding-model", "-e", type=str, help="Embedding模型名称（覆盖.env）")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--temperature", "-t", type=float, default=None, help="温度参数")
    parser.add_argument("--skills-dir", "-s", type=str, default="skills", help="Skill 目录")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式选择模型（忽略.env）")
    parser.add_argument("--auto-index", action="store_true", help="自动索引知识库（默认关闭）")
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

    # 显示配置信息
    print(f"✅ 已加载配置: {config.env_file}")
    print(f"   默认对话模型: {config.chat_model}")
    print(f"   默认向量模型: {config.embedding_model}")

    # 选择模型
    chat_model = None
    embedding_model = None

    if args.interactive:
        # 交互式选择（忽略.env）
        chat_model, embedding_model = select_models_interactive()
        if not chat_model:
            print("❌ 未选择有效的对话模型")
            return
    elif args.model and args.embedding_model:
        # 命令行指定两个模型
        chat_model = args.model
        embedding_model = args.embedding_model
        print(f"\n✅ 使用命令行指定的对话模型: {chat_model}")
        print(f"✅ 使用命令行指定的 Embedding 模型: {embedding_model}")
    elif args.model:
        # 命令行只指定对话模型，Embedding使用.env或交互选择
        chat_model = args.model
        embedding_model = config.embedding_model
        print(f"\n✅ 使用命令行指定的对话模型: {chat_model}")
        print(f"✅ 使用配置文件的 Embedding 模型: {embedding_model}")
    else:
        # 使用.env配置
        chat_model = config.chat_model
        embedding_model = config.embedding_model
        print(f"\n✅ 使用配置文件的对话模型: {chat_model}")
        print(f"✅ 使用配置文件的 Embedding 模型: {embedding_model}")

    # 温度参数
    temperature = args.temperature if args.temperature is not None else config.chat_temperature

    # 创建聊天管理器
    chat_manager = OllamaPilotChat(
        skills_dir=args.skills_dir,
        temperature=temperature
    )

    # 初始化（默认关闭自动索引）
    if not chat_manager.initialize(chat_model, embedding_model, auto_index=args.auto_index):
        return

    # 显示欢迎信息
    chat_manager.show_welcome()

    # 对话循环
    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 再见！")
                break

            # 处理命令
            if chat_manager.handle_command(user_input):
                continue

            # 处理普通对话
            chat_manager.chat(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
