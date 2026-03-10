"""
OllamaPilot - 智能助手启动入口

直接运行: python main.py

使用方法:
    python main.py                    # 交互式选择模型
    python main.py --model qwen3.5:4b # 指定对话模型
    python main.py --list             # 列出可用模型

交互命令 (在对话中输入):
    /help              显示帮助信息
    /model             列出并切换对话模型
    /model <name>      直接切换到指定模型
    /new               开始新对话
    /sessions          列出所有会话
    /switch <id>       切换到指定会话
    /clear             清空当前对话历史
    /embedding         切换 Embedding 模型
    quit/exit          退出程序
"""

import argparse
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
from ollamapilot.agent import OllamaPilotAgent
from langchain_core.language_models.chat_models import BaseChatModel


class Session:
    """会话对象，保存会话状态"""
    def __init__(self, session_id: str, name: str, model_name: str, embedding_model: Optional[str] = None):
        self.id = session_id
        self.name = name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.message_count = 0


class OllamaPilotChat:
    """
    OllamaPilot 聊天管理器
    
    管理多会话、模型切换、对话历史等功能
    """
    
    def __init__(self, skills_dir: str = "skills", temperature: float = 0.7):
        self.skills_dir = skills_dir
        self.temperature = temperature
        
        # 当前状态
        self.current_model: Optional[BaseChatModel] = None
        self.current_model_name: Optional[str] = None
        self.current_embedding_model: Optional[str] = None
        self.agent: Optional[OllamaPilotAgent] = None
        
        # 会话管理
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: str = "default"
        self.session_counter = 0
        
        # 初始化默认会话
        self._create_session("default", "默认会话")
    
    def _create_session(self, session_id: str, name: str) -> Session:
        """创建新会话"""
        session = Session(
            session_id=session_id,
            name=name,
            model_name=self.current_model_name or "未配置",
            embedding_model=self.current_embedding_model
        )
        self.sessions[session_id] = session
        return session
    
    def initialize(self, chat_model: str, embedding_model: Optional[str] = None) -> bool:
        """初始化聊天管理器"""
        try:
            print("\n🔄 正在初始化模型...")
            self.current_model = init_ollama_model(chat_model, temperature=self.temperature)
            self.current_model_name = chat_model
            self.current_embedding_model = embedding_model
            print(f"✅ 对话模型初始化完成: {chat_model}")
            
            print("🔄 正在加载 Skill...")
            agent_kwargs = {"skills_dir": self.skills_dir}
            if embedding_model:
                agent_kwargs["embedding_model"] = embedding_model
            
            self.agent = create_agent(self.current_model, **agent_kwargs)
            print("✅ Agent 创建完成")
            
            # 更新默认会话
            self.sessions["default"].model_name = chat_model
            self.sessions["default"].embedding_model = embedding_model
            
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """切换对话模型"""
        if model_name == self.current_model_name:
            print(f"⚠️ 当前已经是模型: {model_name}")
            return True
        
        try:
            print(f"\n🔄 正在切换到模型: {model_name}...")
            new_model = init_ollama_model(model_name, temperature=self.temperature)
            
            # 重新创建 Agent
            agent_kwargs = {"skills_dir": self.skills_dir}
            if self.current_embedding_model:
                agent_kwargs["embedding_model"] = self.current_embedding_model
            
            new_agent = create_agent(new_model, **agent_kwargs)
            
            # 更新状态
            self.current_model = new_model
            self.current_model_name = model_name
            self.agent = new_agent
            
            # 更新当前会话
            if self.current_session_id in self.sessions:
                self.sessions[self.current_session_id].model_name = model_name
            
            print(f"✅ 已切换到模型: {model_name}")
            return True
        except Exception as e:
            print(f"❌ 切换模型失败: {e}")
            return False
    
    def switch_embedding_model(self, embedding_model: Optional[str] = None) -> bool:
        """切换 Embedding 模型"""
        try:
            print(f"\n🔄 正在重新初始化 Embedding 模型...")
            
            # 重新创建 Agent
            agent_kwargs = {"skills_dir": self.skills_dir}
            if embedding_model:
                agent_kwargs["embedding_model"] = embedding_model
            
            new_agent = create_agent(self.current_model, **agent_kwargs)
            self.agent = new_agent
            self.current_embedding_model = embedding_model
            
            # 更新当前会话
            if self.current_session_id in self.sessions:
                self.sessions[self.current_session_id].embedding_model = embedding_model
            
            if embedding_model:
                print(f"✅ 已切换到 Embedding 模型: {embedding_model}")
            else:
                print("✅ 已禁用 Embedding 模型")
            return True
        except Exception as e:
            print(f"❌ 切换 Embedding 模型失败: {e}")
            return False
    
    def new_session(self, name: Optional[str] = None) -> str:
        """创建新会话"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}_{uuid.uuid4().hex[:8]}"
        
        if not name:
            name = f"会话 {self.session_counter}"
        
        session = self._create_session(
            session_id, 
            name,
        )
        session.model_name = self.current_model_name or "未配置"
        session.embedding_model = self.current_embedding_model
        
        self.current_session_id = session_id
        print(f"\n✅ 已创建新会话: {name} (ID: {session_id[:20]}...)")
        return session_id
    
    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话"""
        if session_id == self.current_session_id:
            print("⚠️ 已经在当前会话中")
            return True
        
        # 支持前缀匹配
        matching_sessions = [sid for sid in self.sessions if sid.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0]
        elif len(matching_sessions) > 1:
            print(f"⚠️ 找到多个匹配的会话，请输入更完整的ID")
            return False
        
        if session_id not in self.sessions:
            print(f"❌ 会话不存在: {session_id}")
            return False
        
        self.current_session_id = session_id
        session = self.sessions[session_id]
        print(f"\n✅ 已切换到会话: {session.name}")
        
        # 如果会话使用了不同的模型，提示用户
        if session.model_name != self.current_model_name:
            print(f"💡 提示: 该会话使用模型 '{session.model_name}'，当前模型是 '{self.current_model_name}'")
        
        return True
    
    def list_sessions(self):
        """列出所有会话"""
        if not self.sessions:
            print("\n📭 没有会话")
            return
        
        print("\n" + "=" * 70)
        print("📋 会话列表")
        print("=" * 70)
        
        for session_id, session in self.sessions.items():
            marker = "👉" if session_id == self.current_session_id else "  "
            print(f"{marker} {session.name}")
            print(f"   ID: {session_id}")
            print(f"   模型: {session.model_name}")
            if session.embedding_model:
                print(f"   Embedding: {session.embedding_model}")
            print(f"   消息数: {session.message_count}")
            print(f"   更新时间: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)
    
    def clear_current_session(self):
        """清空当前会话历史"""
        if self.agent and self.current_session_id:
            self.agent.clear_history(thread_id=self.current_session_id)
            if self.current_session_id in self.sessions:
                self.sessions[self.current_session_id].message_count = 0
            print("\n✅ 已清空当前会话历史")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
┌─────────────────────────────────────────────────────────────────────┐
│                         OllamaPilot 帮助                            │
├─────────────────────────────────────────────────────────────────────┤
│  命令                          说明                                 │
├─────────────────────────────────────────────────────────────────────┤
│  /help                         显示此帮助信息                       │
│  /model                        列出并切换对话模型                   │
│  /model <name>                 直接切换到指定模型                   │
│  /embedding                    切换 Embedding 模型                  │
│  /new                          开始新对话                           │
│  /sessions                     列出所有会话                         │
│  /switch <id>                  切换到指定会话                       │
│  /clear                        清空当前对话历史                     │
│  /info                         显示当前状态信息                     │
│  quit/exit/q                   退出程序                             │
└─────────────────────────────────────────────────────────────────────┘
"""
        print(help_text)
    
    def show_info(self):
        """显示当前状态信息"""
        session = self.sessions.get(self.current_session_id)
        mode = "GraphRAG模式" if self.current_embedding_model else "标准模式"
        
        print("\n" + "=" * 60)
        print("📊 当前状态")
        print("=" * 60)
        print(f"运行模式: {mode}")
        print(f"当前会话: {session.name if session else '未知'}")
        print(f"会话ID: {self.current_session_id}")
        print(f"对话模型: {self.current_model_name or '未配置'}")
        print(f"Embedding模型: {self.current_embedding_model or '未配置'}")
        print(f"温度参数: {self.temperature}")
        print(f"Skill目录: {self.skills_dir}")
        print(f"会话数量: {len(self.sessions)}")
        print("=" * 60)
    
    def show_welcome(self):
        """显示欢迎信息"""
        mode = "GraphRAG模式" if self.current_embedding_model else "标准模式"
        session = self.sessions.get(self.current_session_id)
        
        print("\n" + "=" * 60)
        print(f"🤖 OllamaPilot 智能助手 ({mode})")
        print("=" * 60)
        print(f"当前会话: {session.name if session else '默认会话'}")
        print(f"对话模型: {self.current_model_name or '未配置'}")
        print(f"Embedding模型: {self.current_embedding_model or '未配置'}")
        print(f"温度: {self.temperature}")
        print("-" * 60)
        print("输入 /help 查看所有命令")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """
        处理斜杠命令
        
        Returns:
            True 表示命令已处理，False 表示不是命令
        """
        if not command.startswith('/'):
            return False
        
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        
        if cmd == '/help':
            self.show_help()
        
        elif cmd == '/model':
            if arg:
                # 直接切换到指定模型
                self.switch_model(arg)
            else:
                # 交互式选择
                chat_models = list_ollama_chat_models()
                if not chat_models:
                    print("⚠️ 未检测到对话模型")
                    return True
                
                print("\n📋 可用对话模型:")
                for i, model in enumerate(chat_models, 1):
                    marker = "👉" if model == self.current_model_name else "  "
                    print(f"  {marker} {i}. {model}")
                
                try:
                    choice = input(f"\n请选择 (1-{len(chat_models)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(chat_models):
                        self.switch_model(chat_models[idx])
                except (ValueError, IndexError):
                    print("⚠️ 无效选择")
        
        elif cmd == '/embedding':
            embedding_models = list_ollama_embedding_models()
            if not embedding_models:
                print("⚠️ 未检测到 Embedding 模型")
                return True
            
            print("\n📋 可用 Embedding 模型:")
            print("  0. 禁用 Embedding")
            for i, model in enumerate(embedding_models, 1):
                marker = "👉" if model == self.current_embedding_model else "  "
                print(f"  {marker} {i}. {model}")
            
            try:
                choice = input(f"\n请选择 (0-{len(embedding_models)}): ").strip()
                idx = int(choice)
                if idx == 0:
                    self.switch_embedding_model(None)
                elif 1 <= idx <= len(embedding_models):
                    self.switch_embedding_model(embedding_models[idx - 1])
            except (ValueError, IndexError):
                print("⚠️ 无效选择")
        
        elif cmd == '/new':
            name = arg if arg else None
            self.new_session(name)
        
        elif cmd == '/sessions':
            self.list_sessions()
        
        elif cmd == '/switch':
            if arg:
                self.switch_session(arg)
            else:
                print("⚠️ 请指定会话ID，例如: /switch session_1")
        
        elif cmd == '/clear':
            self.clear_current_session()
        
        elif cmd == '/info':
            self.show_info()
        
        else:
            print(f"❌ 未知命令: {cmd}，输入 /help 查看帮助")
        
        return True
    
    def chat(self, user_input: str):
        """处理用户输入并获取回复"""
        if not self.agent:
            print("❌ Agent 未初始化")
            return
        
        # 更新会话统计
        if self.current_session_id in self.sessions:
            self.sessions[self.current_session_id].message_count += 1
            self.sessions[self.current_session_id].updated_at = datetime.now()
        
        print("\n助手: ", end="", flush=True)
        
        full_response = ""
        has_output = False
        
        try:
            for chunk in self.agent.stream(user_input, thread_id=self.current_session_id):
                if isinstance(chunk, dict):
                    content = None
                    if "messages" in chunk:
                        messages = chunk["messages"]
                        if messages and hasattr(messages[-1], "content"):
                            content = messages[-1].content
                    elif "content" in chunk:
                        content = chunk["content"]
                    elif "agent" in chunk and "messages" in chunk["agent"]:
                        messages = chunk["agent"]["messages"]
                        if messages and hasattr(messages[-1], "content"):
                            content = messages[-1].content
                    
                    if content and len(content) > len(full_response):
                        new_text = content[len(full_response):]
                        if new_text.strip():
                            print(new_text, end="", flush=True)
                            has_output = True
                        full_response = content
            
            if has_output:
                print("\n")
            else:
                print("\n⏳ 生成回答中...")
                response = self.agent.invoke(user_input, thread_id=self.current_session_id)
                if response:
                    print(f"{response}\n")
                else:
                    print("（无回答）\n")
                    
        except Exception as e:
            print(f"\n⚠️ 流式输出失败，使用普通模式: {e}\n")
            try:
                response = self.agent.invoke(user_input, thread_id=self.current_session_id)
                if response:
                    print(f"助手: {response}\n")
            except Exception as e2:
                print(f"❌ 调用失败: {e2}\n")


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
        chat_model = args.model
        embedding_model = args.embedding_model
        print(f"✅ 使用对话模型: {chat_model}")
        print(f"✅ 使用 Embedding 模型: {embedding_model}")
    elif args.model:
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
        chat_model, embedding_model = select_models_interactive()
        if not chat_model:
            print("❌ 未选择有效的对话模型")
            return

    # 创建聊天管理器
    chat_manager = OllamaPilotChat(
        skills_dir=args.skills_dir,
        temperature=args.temperature
    )
    
    # 初始化
    if not chat_manager.initialize(chat_model, embedding_model):
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
