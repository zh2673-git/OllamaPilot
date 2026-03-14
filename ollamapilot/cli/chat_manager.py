"""
聊天管理器模块

管理多会话、模型切换、对话历史、文档索引等功能
"""

import sys
import uuid
import glob
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ollamapilot import (
    init_ollama_model,
    create_agent,
    list_ollama_chat_models,
    list_ollama_embedding_models,
)
from ollamapilot.agent import OllamaPilotAgent
from ollamapilot.config import get_config, reload_config

from .session import Session
from .completer import CommandCompleter, HAS_READLINE


class OllamaPilotChat:
    """
    OllamaPilot 聊天管理器
    
    管理多会话、模型切换、对话历史、文档索引等功能
    """
    
    # 支持的文档扩展名
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}
    
    def __init__(self, skills_dir: str = "skills", temperature: float = 0.7):
        self.skills_dir = skills_dir
        self.temperature = temperature
        self.config = get_config()
        
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
        
        # 文档管理器
        self.doc_manager = None
        
        # 命令补全器
        self.completer = CommandCompleter()
        self._setup_autocomplete()
    
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
    
    def _setup_autocomplete(self):
        """设置命令自动补全"""
        if self.completer.setup():
            print("✅ 命令自动补全已启用（按 Tab 键）")
    
    def _get_files_in_directory(self, dir_path: str) -> List[str]:
        """获取目录中所有支持的文档文件"""
        files = []
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            return files
        
        # 递归查找所有支持的文件
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
        
        return [str(f) for f in files]
    
    def _select_folder_interactive(self) -> Optional[str]:
        """交互式选择文件夹"""
        print("\n📁 选择要索引的文件夹:")
        
        # 列出当前目录下的文件夹
        current_dir = Path(".")
        folders = [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not folders:
            print("  当前目录下没有文件夹")
            return None
        
        print("\n  可用文件夹:")
        for i, folder in enumerate(folders, 1):
            # 计算文件夹中的文档数量
            doc_count = len(self._get_files_in_directory(str(folder)))
            print(f"  {i}. {folder.name} ({doc_count} 个文档)")
        
        print("  0. 输入自定义路径")
        
        try:
            choice = input(f"\n请选择 (0-{len(folders)}): ").strip()
            idx = int(choice)
            
            if idx == 0:
                custom_path = input("请输入文件夹路径: ").strip()
                return custom_path if custom_path else None
            elif 1 <= idx <= len(folders):
                return str(folders[idx - 1])
            else:
                print("⚠️ 无效选择")
                return None
        except (ValueError, IndexError):
            print("⚠️ 无效输入")
            return None
    
    def initialize(self, chat_model: str, embedding_model: Optional[str] = None, auto_index: bool = False) -> bool:
        """初始化聊天管理器"""
        try:
            print("\n🔄 正在初始化模型...")
            
            self.current_model = init_ollama_model(chat_model, temperature=self.temperature)
            self.current_model_name = chat_model
            print(f"✅ 对话模型初始化完成: {chat_model}")
            
            self.current_embedding_model = embedding_model
            
            print("🔄 正在加载 Skill...")
            agent_kwargs = {"skills_dir": self.skills_dir}
            if embedding_model:
                agent_kwargs["embedding_model"] = embedding_model
            
            self.agent = create_agent(self.current_model, **agent_kwargs)
            print("✅ Agent 创建完成")
            
            # 初始化文档管理器（手动控制模式）
            if embedding_model:
                from skills.graphrag.document_manager import DocumentManager
                self.doc_manager = DocumentManager(
                    base_persist_dir=self.config.graph_rag_persist_dir,
                    embedding_model=embedding_model
                )
                print(f"📚 文档管理器已加载（手动控制模式）")
                print(f"   使用 /index 命令手动索引文档")
                print(f"   使用 /resume 恢复失败的索引")
            
            # 更新默认会话
            self.sessions["default"].model_name = chat_model
            self.sessions["default"].embedding_model = embedding_model
            
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def index_document(self, doc_path: str, doc_name: Optional[str] = None, async_mode: bool = True) -> bool:
        """手动索引文档或文件夹"""
        if not self.doc_manager:
            print("❌ 文档管理器未初始化（可能没有配置Embedding模型）")
            return False
        
        path = Path(doc_path)
        
        # 处理文件夹
        if path.is_dir():
            return self._index_directory_async(doc_path) if async_mode else self._index_directory_sync(doc_path)
        
        # 处理单个文件
        if not doc_name:
            doc_name = path.stem
        
        try:
            print(f"\n📄 注册文档: {doc_name}")
            doc_id = self.doc_manager.register_document(
                doc_name=doc_name,
                file_path=doc_path,
                auto_index=False
            )
            
            if async_mode:
                print(f"🔄 开始后台索引（您可以继续对话）...")
                self.doc_manager.start_indexing(doc_id, silent=True)
                print(f"   使用 /docs 查看索引进度")
                return True
            else:
                print(f"🔄 开始索引（静默模式）...")
                self.doc_manager.start_indexing(doc_id, silent=True)
                success = self.doc_manager.wait_for_indexing(doc_id, timeout=300)
                
                if success:
                    doc_info = self.doc_manager.get_document_status(doc_id)
                    print(f"✅ 索引完成: {doc_info.chunks_count} 块, {doc_info.entities_count} 个实体")
                else:
                    print(f"❌ 索引失败或超时")
                
                return success
            
        except Exception as e:
            print(f"❌ 索引失败: {e}")
            return False
    
    def _index_directory_async(self, dir_path: str) -> bool:
        """异步索引整个文件夹（不阻塞对话）"""
        print(f"\n📁 索引文件夹: {dir_path}")
        
        files = self._get_files_in_directory(dir_path)
        
        if not files:
            print(f"⚠️ 文件夹中没有支持的文档（支持: {', '.join(self.SUPPORTED_EXTENSIONS)}）")
            return False
        
        print(f"  发现 {len(files)} 个文档")
        print("\n  文档列表:")
        for i, file in enumerate(files[:10], 1):
            print(f"    {i}. {Path(file).name}")
        if len(files) > 10:
            print(f"    ... 还有 {len(files) - 10} 个文档")
        
        confirm = input(f"\n是否索引这 {len(files)} 个文档? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  已取消")
            return False
        
        print(f"\n🔄 开始后台批量索引（您可以继续对话）...")
        print(f"   使用 /docs 查看实时进度\n")

        def batch_index_worker():
            doc_ids = []

            # 注册所有文档
            for i, file_path in enumerate(files, 1):
                file_name = Path(file_path).stem

                try:
                    doc_id = self.doc_manager.register_document(
                        doc_name=file_name,
                        file_path=file_path,
                        auto_index=False
                    )
                    doc_ids.append((doc_id, file_name))
                    print(f"  📄 已注册: {file_name}")

                except Exception as e:
                    print(f"  ❌ [{file_name}] 注册失败: {e}")

            # 启动所有索引（不等待）
            for doc_id, file_name in doc_ids:
                try:
                    self.doc_manager.start_indexing(doc_id, silent=True)
                    print(f"  🔄 开始索引: {file_name}")
                except Exception as e:
                    print(f"  ❌ [{file_name}] 启动失败: {e}")

            # 显示启动完成提示
            print(f"\n✅ 已启动 {len(doc_ids)} 个文档的后台索引")
            print(f"   使用 /docs 随时查看进度\n")

        thread = threading.Thread(target=batch_index_worker, daemon=True)
        thread.start()

        return True
    
    def _index_directory_sync(self, dir_path: str) -> bool:
        """同步索引整个文件夹（阻塞式）"""
        print(f"\n📁 索引文件夹: {dir_path}")
        
        files = self._get_files_in_directory(dir_path)
        
        if not files:
            print(f"⚠️ 文件夹中没有支持的文档")
            return False
        
        print(f"  发现 {len(files)} 个文档")
        
        confirm = input(f"\n是否索引这 {len(files)} 个文档? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  已取消")
            return False
        
        success_count = 0
        failed_count = 0
        
        print(f"\n🔄 开始批量索引...")
        for i, file_path in enumerate(files, 1):
            file_name = Path(file_path).stem
            print(f"\n  [{i}/{len(files)}] {file_name}")
            
            try:
                doc_id = self.doc_manager.register_document(
                    doc_name=file_name,
                    file_path=file_path,
                    auto_index=False
                )
                
                self.doc_manager.start_indexing(doc_id, silent=True)
                success = self.doc_manager.wait_for_indexing(doc_id, timeout=300)
                
                if success:
                    success_count += 1
                    doc_info = self.doc_manager.get_document_status(doc_id)
                    print(f"  ✅ 完成: {doc_info.chunks_count} 块, {doc_info.entities_count} 实体")
                else:
                    failed_count += 1
                    print(f"  ❌ 失败")
                    
            except Exception as e:
                failed_count += 1
                print(f"  ❌ 错误: {e}")
        
        print(f"\n📊 批量索引完成: {success_count} 成功, {failed_count} 失败, 总计 {len(files)}")
        return success_count > 0
    
    def resume_failed_indexing(self) -> bool:
        """恢复失败的索引任务"""
        if not self.doc_manager:
            print("❌ 文档管理器未初始化")
            return False
        
        resumed = self.doc_manager.resume_failed_indexing()
        return len(resumed) > 0
    
    def list_documents(self):
        """列出所有文档"""
        if not self.doc_manager:
            print("❌ 文档管理器未初始化")
            return
        
        docs = self.doc_manager.list_documents()
        if not docs:
            print("\n📭 没有文档")
            return
        
        print("\n" + "=" * 70)
        print("📚 文档列表")
        print("=" * 70)
        
        for doc in docs:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(doc.status.value, "❓")

            print(f"{status_icon} {doc.name}")
            print(f"   ID: {doc.doc_id}")

            # 显示详细进度
            if doc.status.value == "running" and doc.chunks_count > 0:
                # 计算当前处理的块数
                current_chunk = int(doc.progress * doc.chunks_count)
                print(f"   状态: 索引中 ({current_chunk}/{doc.chunks_count} 块, {doc.progress*100:.0f}%)")

                # 计算时间预估
                if doc.started_at and current_chunk > 0:
                    elapsed = time.time() - doc.started_at
                    avg_time_per_chunk = elapsed / current_chunk
                    remaining_chunks = doc.chunks_count - current_chunk
                    estimated_remaining = avg_time_per_chunk * remaining_chunks

                    # 格式化时间
                    if estimated_remaining < 60:
                        time_str = f"约 {int(estimated_remaining)} 秒"
                    elif estimated_remaining < 3600:
                        time_str = f"约 {int(estimated_remaining/60)} 分钟"
                    else:
                        time_str = f"约 {int(estimated_remaining/3600)} 小时 {int((estimated_remaining%3600)/60)} 分钟"

                    print(f"   预估剩余: {time_str}")
            else:
                print(f"   状态: {doc.status.value}")

            print(f"   模型: {doc.model_name}")

            if doc.chunks_count > 0:
                print(f"   分块: {doc.chunks_count}, 实体: {doc.entities_count}")

            if doc.message and doc.status.value == "running":
                print(f"   进度: {doc.message}")

            print("-" * 70)
    
    def switch_model(self, model_name: str) -> bool:
        """切换对话模型"""
        if model_name == self.current_model_name:
            print(f"⚠️ 当前已经是模型: {model_name}")
            return True
        
        try:
            print(f"\n🔄 正在切换到模型: {model_name}...")

            new_model = init_ollama_model(model_name, temperature=self.temperature)

            agent_kwargs = {"skills_dir": self.skills_dir}
            if self.current_embedding_model:
                agent_kwargs["embedding_model"] = self.current_embedding_model

            new_agent = create_agent(new_model, **agent_kwargs)

            self.current_model = new_model
            self.current_model_name = model_name
            self.agent = new_agent

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
            
            agent_kwargs = {"skills_dir": self.skills_dir}
            if embedding_model:
                agent_kwargs["embedding_model"] = embedding_model
            
            new_agent = create_agent(self.current_model, **agent_kwargs)
            self.agent = new_agent
            self.current_embedding_model = embedding_model
            
            if embedding_model:
                from skills.graphrag.document_manager import DocumentManager
                self.doc_manager = DocumentManager(
                    base_persist_dir=self.config.graph_rag_persist_dir,
                    embedding_model=embedding_model
                )
            else:
                self.doc_manager = None
            
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
        
        session = self._create_session(session_id, name)
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
    
    def reload_config(self):
        """重新加载配置"""
        try:
            reload_config()
            print("\n✅ 配置已重新加载")
            print(f"   对话模型: {self.config.chat_model}")
            print(f"   向量模型: {self.config.embedding_model}")
        except Exception as e:
            print(f"\n❌ 配置重载失败: {e}")
    
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
│  /docs                         列出所有文档                         │
│  /index [path]                 索引文档/文件夹(默认:knowledge_base) │
│  /resume                       恢复失败的索引任务                   │
│  /reload                       重新加载 .env 配置                   │
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
        print("输入 /index 手动索引文档")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """处理斜杠命令"""
        if not command.startswith('/'):
            return False
        
        parts = command.split(maxsplit=2)
        cmd = parts[0].lower()
        arg1 = parts[1] if len(parts) > 1 else None
        arg2 = parts[2] if len(parts) > 2 else None
        
        if cmd == '/help':
            self.show_help()
        
        elif cmd == '/model':
            if arg1:
                self.switch_model(arg1)
            else:
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
            name = arg1 if arg1 else None
            self.new_session(name)
        
        elif cmd == '/sessions':
            self.list_sessions()
        
        elif cmd == '/switch':
            if arg1:
                self.switch_session(arg1)
            else:
                print("⚠️ 请指定会话ID，例如: /switch session_1")
        
        elif cmd == '/clear':
            self.clear_current_session()
        
        elif cmd == '/info':
            self.show_info()
        
        elif cmd == '/docs':
            self.list_documents()
        
        elif cmd == '/index':
            if arg1:
                self.index_document(arg1)
            else:
                default_kb = Path("./knowledge_base")
                if default_kb.exists() and default_kb.is_dir():
                    files = self._get_files_in_directory(str(default_kb))
                    if files:
                        print(f"📁 使用默认知识库文件夹: {default_kb}")
                        print(f"   发现 {len(files)} 个文档")
                        self.index_document(str(default_kb))
                    else:
                        print(f"⚠️ 默认知识库文件夹为空: {default_kb}")
                        print(f"   支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS)}")
                        choice = input("\n是否选择其他文件夹? (y/n): ").strip().lower()
                        if choice == 'y':
                            folder = self._select_folder_interactive()
                            if folder:
                                self.index_document(folder)
                else:
                    print(f"⚠️ 默认知识库文件夹不存在: {default_kb}")
                    print("\n选项:")
                    print("  1. 创建默认知识库文件夹")
                    print("  2. 选择其他文件夹")
                    print("  3. 取消")
                    choice = input("\n请选择 (1-3): ").strip()
                    
                    if choice == '1':
                        default_kb.mkdir(parents=True, exist_ok=True)
                        print(f"✅ 已创建文件夹: {default_kb}")
                        print(f"   请将文档放入此文件夹，然后再次运行 /index")
                    elif choice == '2':
                        folder = self._select_folder_interactive()
                        if folder:
                            self.index_document(folder)
        
        elif cmd == '/resume':
            self.resume_failed_indexing()
        
        elif cmd == '/reload':
            self.reload_config()
        
        else:
            print(f"❌ 未知命令: {cmd}，输入 /help 查看帮助")
        
        return True
    
    def chat(self, user_input: str):
        """处理用户输入并获取回复（带超时保护）"""
        if not self.agent:
            print("❌ Agent 未初始化")
            return

        # 检查是否有正在进行的索引任务
        indexing_count = len(self.doc_manager._indexing_tasks)
        if indexing_count > 0:
            print(f"\n⏳ 检测到 {indexing_count} 个文档正在后台索引...")
            print(f"   使用 /docs 查看进度")
            print(f"   索引期间无法对话，请等待索引完成\n")
            return

        # 更新会话统计
        if self.current_session_id in self.sessions:
            self.sessions[self.current_session_id].increment_message()

        print("\n助手: ", end="", flush=True)

        full_response = ""
        has_content = False
        tool_call_count = 0
        start_time = time.time()
        timeout = self.config.cli_timeout  # 从配置读取超时时间，默认 120 秒

        async def _process_stream():
            nonlocal full_response, has_content, tool_call_count
            
            async for event in self.agent.astream_events(user_input, thread_id=self.current_session_id):
                # 检查超时
                if time.time() - start_time > timeout:
                    print(f"\n\n⏰ 响应超时（{timeout}秒）")
                    print(f"   可能原因: 模型生成较慢、网络搜索耗时、或知识库检索中")
                    print(f"   提示: 可通过 .env 文件设置 CLI_TIMEOUT 调整超时时间\n")
                    return

                event_type = event.get("event", "")
                
                # 工具开始执行
                if event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_call_count += 1
                    print(f"\n🔧 执行工具: {tool_name}...", end="", flush=True)
                
                # 工具执行结束
                elif event_type == "on_tool_end":
                    print(" ✅", end="", flush=True)
                
                # 模型流式输出
                elif event_type == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk", None)
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                            has_content = True

        async def _force_response():
            """强制生成回复"""
            nonlocal full_response, has_content
            print("\n🔄 正在生成回复...", end="", flush=True)
            try:
                forced_response = ""
                async for chunk in self.agent.force_response_after_tool(self.current_session_id):
                    print(chunk, end="", flush=True)
                    forced_response += chunk
                if forced_response:
                    print("\n")
                    full_response = forced_response
                    has_content = True
                else:
                    print("\n（工具执行完成，但无法生成回复内容）\n")
            except Exception as e:
                print(f"\n（工具执行完成，但生成回复失败: {e}）\n")

        try:
            # 运行异步流处理
            # 使用 get_event_loop() 而不是 asyncio.run() 避免嵌套事件循环问题
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_process_stream())
            else:
                loop.run_until_complete(_process_stream())
            
            if has_content:
                print("\n")
            elif tool_call_count > 0:
                # 有工具调用但没有内容输出，尝试强制生成回复
                if loop.is_running():
                    loop.create_task(_force_response())
                else:
                    loop.run_until_complete(_force_response())
            else:
                print("（无回答）\n")

        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断")
            print("   如需退出程序，请输入 'quit' 或 'exit'\n")
            return

        except Exception as e:
            print(f"\n⚠️ 流式输出失败: {e}\n")
            # 注意：不再回退到 invoke，因为 astream_events 已经消耗了消息历史
            # 如果回退会导致：1. 工具重复执行 2. 回答重复打印
            print("   提示: 如需重新回答，请再次输入问题\n")
