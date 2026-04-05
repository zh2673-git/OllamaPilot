"""
聊天管理器模块

管理多会话、模型切换、对话历史、文档索引等功能
"""

import sys
import uuid
import glob
import time
import json
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ollamapilot import (
    init_ollama_model,
    create_agent,
    list_ollama_chat_models,
    list_ollama_embedding_models,
)
from ollamapilot.agent import OllamaPilotAgent
from ollamapilot.config import get_config, reload_config

# Harness 架构支持
from ollamapilot.harness import create_harness_agent, OllamaPilotHarnessAgent

from .session import Session
from .session_store import SessionStore
from .history_manager import SimpleHistoryManager
from .completer import CommandCompleter, HAS_READLINE
from ollamapilot.logging_config import get_default_logger

get_default_logger()
logger = logging.getLogger("ollamapilot.cli.chat")


class OllamaPilotChat:
    """
    OllamaPilot 聊天管理器
    
    管理多会话、模型切换、对话历史、文档索引等功能
    """
    
    # 支持的文档扩展名
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}
    
    def __init__(
        self, 
        skills_dir: str = "skills", 
        temperature: float = 0.7,
        use_harness: bool = True,
        use_middleware_chain: bool = True,
        use_enhanced_tools: bool = True  # 默认启用增强工具
    ):
        self.skills_dir = skills_dir
        self.temperature = temperature
        self.config = get_config()
        
        # Harness 架构配置
        self.use_harness = use_harness
        self.use_middleware_chain = use_middleware_chain
        self.use_enhanced_tools = use_enhanced_tools
        
        # 当前状态
        self.current_model: Optional[BaseChatModel] = None
        self.current_model_name: Optional[str] = None
        self.current_embedding_model: Optional[str] = None
        self.agent: Optional[OllamaPilotAgent] = None
        
        # 会话管理
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: str = "default"
        self.session_counter = 0
        
        # 会话存储管理器
        self.session_store = SessionStore("./data/sessions/conversations.db")

        # 对话历史管理器（定期保存）
        self.history_manager: Optional[SimpleHistoryManager] = None

        # 初始化默认会话
        self._create_session("default", "默认会话")
        
        # 从数据库恢复会话
        self._restore_sessions_from_db()
        
        # 文档管理器
        self.doc_manager = None
        
        # 命令补全器
        self.completer = CommandCompleter()
        self._setup_autocomplete()
    
    def _create_session(self, session_id: str, name: str, source: str = "memory") -> Session:
        """创建新会话"""
        session = Session(
            session_id=session_id,
            name=name,
            model_name=self.current_model_name or "未配置",
            embedding_model=self.current_embedding_model
        )
        session.source = source
        self.sessions[session_id] = session
        return session
    
    def _restore_sessions_from_db(self):
        """从数据库恢复会话列表"""
        try:
            restored_sessions = self.session_store.restore_sessions(
                model_name=self.current_model_name or "unknown",
                embedding_model=self.current_embedding_model
            )
            
            if restored_sessions:
                # 合并恢复的会话（不覆盖内存中的默认会话）
                for session_id, session in restored_sessions.items():
                    if session_id not in self.sessions:
                        self.sessions[session_id] = session
                        # 更新会话计数器
                        if session_id.startswith("session_"):
                            try:
                                num = int(session_id.split("_")[1])
                                self.session_counter = max(self.session_counter, num)
                            except (ValueError, IndexError):
                                pass
                
                db_count = len([s for s in restored_sessions.values() if s.is_from_database])
                if db_count > 0:
                    logger.info(f"已从数据库恢复 {db_count} 个历史会话")

        except Exception as e:
            logger.warning(f"恢复会话失败: {e}")
    
    def _setup_autocomplete(self):
        """设置命令自动补全"""
        if self.completer.setup():
            logger.debug("命令自动补全已启用（按 Tab 键）")

    def close(self):
        """关闭聊天管理器，保存所有状态"""
        if self.history_manager:
            self._save_history_from_agent()
            self.history_manager.close()
        self.session_store.close()

    def _save_history_from_agent(self):
        """从 agent 的 checkpointer 提取完整历史并保存"""
        if not self.agent or not self.agent.checkpointer or not self.history_manager:
            return

        config = {"configurable": {"thread_id": self.current_session_id}}

        try:
            checkpoint_tuple = self.agent.checkpointer.get_tuple(config)
            if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
                return

            checkpoint = checkpoint_tuple.checkpoint
            messages = checkpoint.get("messages", [])

            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    msg_type = type(msg).__name__
                    if "Human" in msg_type:
                        self.history_manager.add_human_message(msg.content)
                    elif "AIMessage" in msg_type:
                        self.history_manager.add_ai_message(msg.content)
                    elif "ToolMessage" in msg_type:
                        tool_name = getattr(msg, "name", "unknown")
                        self.history_manager.add_tool_message(msg.content, tool_name)
        except Exception:
            pass

    def _restore_history_to_agent(self):
        """将历史从 history_manager 恢复到 agent 的 checkpointer"""
        if not self.agent or not self.agent.checkpointer:
            return

        messages = self.history_manager.get_messages()
        if not messages:
            return

        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        config = {"configurable": {"thread_id": self.current_session_id}}

        reconstructed = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "human":
                reconstructed.append(HumanMessage(content=content))
            elif role == "ai":
                reconstructed.append(AIMessage(content=content))
            elif role == "tool":
                tool_name = msg.get("metadata", {}).get("tool_name", "unknown")
                reconstructed.append(ToolMessage(content=content, tool_call_id="restored", name=tool_name))

        if reconstructed:
            checkpoint = {"messages": reconstructed}
            try:
                self.agent.checkpointer.put(
                    config,
                    checkpoint,
                    {"source": "checkpoint", "timestamp": time.time()}
                )
            except Exception:
                pass

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
            
            # 使用 Harness 架构创建 Agent
            if self.use_harness:
                print("🔧 使用 Harness 架构 (v0.6.0)")
                agent_kwargs.update({
                    "use_middleware_chain": self.use_middleware_chain,
                    "use_enhanced_tools": self.use_enhanced_tools,
                })
                self.agent = create_harness_agent(self.current_model, **agent_kwargs)
                print(f"✅ Harness Agent 创建完成 (中间件: {self.use_middleware_chain}, 增强工具: {self.use_enhanced_tools})")
            else:
                print("🔧 使用传统架构 (v0.5.0)")
                self.agent = create_agent(self.current_model, **agent_kwargs)
                print("✅ Agent 创建完成")

            # 初始化对话历史管理器
            self.history_manager = SimpleHistoryManager(
                session_id=self.current_session_id,
                storage_dir="./data/sessions/history",
                auto_save_interval=30,
            )
            self.history_manager.restore()
            self.history_manager.start_auto_save()

            # 将历史注入到 ContextBuilder（启动时加载，避免每次调用时传递）
            if self.agent and self.agent.context_builder:
                from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
                history_messages = []
                for msg in self.history_manager.get_messages():
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "human":
                        history_messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        history_messages.append(AIMessage(content=content))
                    elif role == "tool":
                        tool_name = msg.get("metadata", {}).get("tool_name", "unknown")
                        history_messages.append(ToolMessage(content=content, tool_call_id="loaded", name=tool_name))
                if history_messages:
                    self.agent.context_builder.set_preloaded_history(history_messages, thread_id=self.current_session_id)
                    print(f"💾 已加载 {len(history_messages)} 条历史消息到 Context")

            # 初始化文档管理器（手动控制模式，启用LightRAG增强）
            if embedding_model:
                from skills.graphrag.document_manager import DocumentManager
                self.doc_manager = DocumentManager(
                    base_persist_dir=self.config.graph_rag_persist_dir,
                    embedding_model=embedding_model,
                    enable_relation_vector=True,   # 启用关系向量化
                    enable_dual_retrieval=True,    # 启用双层检索
                    use_llm_merge=False            # 小模型建议关闭LLM合并
                )
                print(f"📚 文档管理器已加载（手动控制模式，LightRAG增强已启用）")
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
    
    def index_document(self, doc_path: str, doc_name: Optional[str] = None, async_mode: bool = True, force: bool = False) -> bool:
        """手动索引文档或文件夹
        
        Args:
            doc_path: 文档路径
            doc_name: 文档名称（可选）
            async_mode: 是否异步索引
            force: 是否强制重新索引（即使已完成也会重新索引）
        """
        if not self.doc_manager:
            print("❌ 文档管理器未初始化（可能没有配置Embedding模型）")
            return False

        path = Path(doc_path)

        # 处理文件夹
        if path.is_dir():
            return self._index_directory_async(doc_path, force=force) if async_mode else self._index_directory_sync(doc_path, force=force)
        
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
                self.doc_manager.start_indexing(doc_id, silent=True, force=force)
                print(f"   使用 /docs 查看索引进度")
                return True
            else:
                print(f"🔄 开始索引（静默模式）...")
                self.doc_manager.start_indexing(doc_id, silent=True, force=force)
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

    def _check_and_migrate_legacy_data(self):
        """检查并迁移旧版数据到三重向量存储"""
        if not self.doc_manager:
            return

        base_dir = Path(self.doc_manager.base_persist_dir)
        if not base_dir.exists():
            return

        # 查找所有包含旧版数据但未迁移的文档目录
        migrated_flag = "migrated_{}.flag"
        needs_migration = []

        for item in base_dir.iterdir():
            if not item.is_dir():
                continue

            # 检查是否有旧版 index_*.json 文件
            index_files = list(item.glob("index_*.json"))
            if not index_files:
                continue

            # 检查是否已迁移
            model_name = index_files[0].stem.replace("index_", "")
            flag_file = item / migrated_flag.format(model_name)

            if flag_file.exists():
                continue  # 已迁移

            # 检查是否有实体数据需要迁移
            try:
                with open(index_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("entity_index") or data.get("relations"):
                        needs_migration.append((item, model_name))
            except Exception:
                continue

        if not needs_migration:
            return

        print(f"\n🔄 检测到 {len(needs_migration)} 个文档需要数据迁移...")
        print("   升级到 LightRAG 三重向量存储格式")
        print("\n   文档列表:")
        for i, (doc_dir, _) in enumerate(needs_migration[:5], 1):
            print(f"     {i}. {doc_dir.name}")
        if len(needs_migration) > 5:
            print(f"     ... 还有 {len(needs_migration) - 5} 个文档")

        confirm = input(f"\n是否开始迁移? (y/n): ").strip().lower()
        if confirm != 'y':
            print("   已取消迁移")
            return

        print("\n   开始迁移...\n")

        migrated_count = 0
        import time

        for doc_dir, model_name in needs_migration:
            try:
                print(f"  📄 迁移: {doc_dir.name}")

                # 创建 GraphRAGService
                from skills.graphrag.services import GraphRAGService
                graph_service = GraphRAGService(
                    persist_dir=str(doc_dir),
                    embedding_model=model_name,
                    enable_relation_vector=True,
                    enable_dual_retrieval=True,
                    use_llm_merge=False
                )

                # 手动触发迁移
                migrated = graph_service.check_and_migrate()

                # 检查迁移结果
                stats = graph_service.get_stats()
                if stats.get("triple_store"):
                    triple_stats = stats["triple_store"]
                    if migrated:
                        print(f"     ✅ 完成: {triple_stats['entities']} 实体, {triple_stats['relations']} 关系")
                        migrated_count += 1
                    elif triple_stats["entities"] > 0 or triple_stats["relations"] > 0:
                        print(f"     ✅ 已迁移: {triple_stats['entities']} 实体, {triple_stats['relations']} 关系")
                    else:
                        print(f"     ⏭️  跳过: 无需迁移")
                else:
                    print(f"     ⏭️  跳过: 三重存储未启用")

                # 添加延迟，避免并发请求 Ollama
                time.sleep(0.5)

            except Exception as e:
                print(f"     ⚠️  失败: {e}")
                # 失败后等待更长时间
                time.sleep(1)

        if migrated_count > 0:
            print(f"\n✅ 数据迁移完成: {migrated_count}/{len(needs_migration)} 个文档")
        print()

    def _index_directory_async(self, dir_path: str, force: bool = False) -> bool:
        """异步索引整个文件夹（不阻塞对话）
        
        Args:
            dir_path: 文件夹路径
            force: 是否强制重新索引
        """
        print(f"\n📁 索引文件夹: {dir_path}")
        if force:
            print("  🔄 强制重新索引模式（将清除旧数据）")
        
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
                    self.doc_manager.start_indexing(doc_id, silent=True, force=force)
                    print(f"  🔄 开始索引: {file_name}")
                except Exception as e:
                    print(f"  ❌ [{file_name}] 启动失败: {e}")

            # 显示启动完成提示
            print(f"\n✅ 已启动 {len(doc_ids)} 个文档的后台索引")
            print(f"   使用 /docs 随时查看进度\n")

        thread = threading.Thread(target=batch_index_worker, daemon=True)
        thread.start()

        return True
    
    def _index_directory_sync(self, dir_path: str, force: bool = False) -> bool:
        """同步索引整个文件夹（阻塞式）
        
        Args:
            dir_path: 文件夹路径
            force: 是否强制重新索引
        """
        print(f"\n📁 索引文件夹: {dir_path}")
        if force:
            print("  🔄 强制重新索引模式（将清除旧数据）")
        
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
                
                self.doc_manager.start_indexing(doc_id, silent=True, force=force)
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
                    embedding_model=embedding_model,
                    enable_relation_vector=True,   # 启用关系向量化
                    enable_dual_retrieval=True,    # 启用双层检索
                    use_llm_merge=False            # 小模型建议关闭LLM合并
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

        if self.history_manager:
            self.history_manager.save()
            self.history_manager.session_id = session_id
            self.history_manager._storage_file = self.history_manager.storage_dir / f"{session_id}.json"
            self.history_manager.restore()

            # 更新 ContextBuilder 中的历史为新会话的历史
            if self.agent and self.agent.context_builder:
                from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
                history_messages = []
                for msg in self.history_manager.get_messages():
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "human":
                        history_messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        history_messages.append(AIMessage(content=content))
                    elif role == "tool":
                        tool_name = msg.get("metadata", {}).get("tool_name", "unknown")
                        history_messages.append(ToolMessage(content=content, tool_call_id="loaded", name=tool_name))
                self.agent.context_builder.set_preloaded_history(history_messages, thread_id=session_id)

        return True
    
    def list_sessions(self):
        """列出所有会话"""
        if not self.sessions:
            print("\n📭 没有会话")
            return
        
        # 获取统计信息
        stats = self.session_store.get_session_stats()
        
        print("\n" + "=" * 70)
        print("📋 会话列表")
        if stats["total_sessions"] > 0:
            print(f"   数据库: {stats['total_sessions']} 个会话, {stats['total_messages']} 条消息, {stats['db_size_mb']} MB")
        print("=" * 70)
        
        # 按更新时间排序
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].updated_at,
            reverse=True
        )
        
        for session_id, session in sorted_sessions:
            marker = "👉" if session_id == self.current_session_id else "  "
            source_icon = "📦" if session.is_from_database else "💾"
            print(f"{marker} {source_icon} {session.name}")
            print(f"   ID: {session_id}")
            print(f"   模型: {session.model_name}")
            if session.embedding_model:
                print(f"   Embedding: {session.embedding_model}")
            print(f"   消息数: {session.message_count}")
            print(f"   更新时间: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if session.description:
                print(f"   描述: {session.description}")
            print("-" * 70)
    
    def show_session_history(self, session_id: Optional[str] = None, limit: int = 20):
        """显示会话历史"""
        if session_id is None:
            session_id = self.current_session_id
        
        # 查找会话
        matching_sessions = [sid for sid in self.sessions if sid.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0]
        elif len(matching_sessions) > 1:
            print(f"⚠️ 找到多个匹配的会话，请输入更完整的ID")
            return
        
        if session_id not in self.sessions:
            print(f"❌ 会话不存在: {session_id}")
            return
        
        session = self.sessions[session_id]
        print(f"\n📜 会话历史: {session.name}")
        print(f"   ID: {session_id}")
        print("=" * 70)
        
        # 从数据库获取历史
        messages = self.session_store.get_session_history(session_id, limit=limit)
        
        if not messages:
            print("   (暂无消息)")
        else:
            for i, msg in enumerate(messages, 1):
                msg_type = msg.get("type", "")
                content = msg.get("content", "")
                
                if msg_type == "human":
                    print(f"\n{i}. 👤 用户: {content}")
                elif msg_type == "ai":
                    print(f"   🤖 助手: {content[:150]}..." if len(content) > 150 else f"   🤖 助手: {content}")
                elif msg_type == "system":
                    print(f"   ⚙️  [系统消息]")
        
        print(f"\n   共显示 {len(messages)} 条消息")
    
    def delete_session(self, session_id: str, confirm: bool = True):
        """删除会话"""
        # 查找会话
        matching_sessions = [sid for sid in self.sessions if sid.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0]
        elif len(matching_sessions) > 1:
            print(f"⚠️ 找到多个匹配的会话，请输入更完整的ID")
            return False
        
        if session_id not in self.sessions:
            print(f"❌ 会话不存在: {session_id}")
            return False
        
        session = self.sessions[session_id]
        
        if confirm:
            print(f"\n⚠️ 确定要删除会话 '{session.name}' 吗？")
            print(f"   ID: {session_id}")
            print(f"   消息数: {session.message_count}")
            print(f"   此操作不可恢复！")
            response = input("   输入 'yes' 确认删除: ")
            if response.lower() != 'yes':
                print("   已取消删除")
                return False
        
        # 从数据库删除
        if self.session_store.delete_session(session_id):
            # 从内存删除
            del self.sessions[session_id]
            
            # 如果删除的是当前会话，切换到默认会话
            if session_id == self.current_session_id:
                self.current_session_id = "default"
                print(f"\n✅ 已删除会话并切换到默认会话")
            else:
                print(f"\n✅ 已删除会话: {session.name}")
            
            return True
        else:
            print(f"\n❌ 删除会话失败")
            return False
    
    def rename_session(self, new_name: str):
        """重命名当前会话"""
        if self.current_session_id not in self.sessions:
            print("❌ 当前会话不存在")
            return
        
        session = self.sessions[self.current_session_id]
        old_name = session.name
        session.rename(new_name)
        print(f"✅ 已重命名会话: '{old_name}' -> '{new_name}'")
    
    def export_session(self, session_id: Optional[str] = None):
        """导出会话"""
        if session_id is None:
            session_id = self.current_session_id
        
        # 查找会话
        matching_sessions = [sid for sid in self.sessions if sid.startswith(session_id)]
        if len(matching_sessions) == 1:
            session_id = matching_sessions[0]
        elif len(matching_sessions) > 1:
            print(f"⚠️ 找到多个匹配的会话，请输入更完整的ID")
            return
        
        if session_id not in self.sessions:
            print(f"❌ 会话不存在: {session_id}")
            return
        
        session = self.sessions[session_id]
        filepath = self.session_store.export_session(session_id)
        
        if filepath:
            print(f"✅ 已导出会话: {session.name}")
            print(f"📄 文件: {filepath}")
        else:
            print(f"❌ 导出失败")
    
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

    def upload_and_analyze(self, filepath: str = None):
        """上传并分析文件/图片"""
        from ollamapilot.utils.file_processor import get_file_processor
        from pathlib import Path

        processor = get_file_processor()

        # 如果没有提供路径，提示用户输入
        if not filepath:
            filepath = input("请输入文件路径: ").strip()

        if not filepath:
            print("⚠️ 未指定文件路径")
            return

        path = Path(filepath)
        if not path.exists():
            print(f"❌ 文件不存在: {filepath}")
            return

        print(f"\n📄 正在分析文件: {path.name}")

        try:
            # 检查是否是图片
            suffix = path.suffix.lower()
            image_suffixes = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']

            if suffix in image_suffixes:
                # 图片分析
                print("🖼️  正在分析图片内容...")
                import asyncio
                description = asyncio.run(processor.analyze_image(
                    path,
                    query="描述这张图片的内容",
                    model=self.agent.model if hasattr(self.agent, 'model') else None
                ))
                content = f"🖼️ 图片分析结果:\n{description}"
            else:
                # 文档分析
                print("📄 正在提取文档内容...")
                text = processor.extract_text_content(path)

                if text.startswith('[') and text.endswith(']'):
                    # 错误信息
                    print(f"⚠️ {text}")
                    return

                content = f"📄 文件: {path.name}\n{text[:8000]}"  # 限制长度

            # 询问用户问题
            print("\n💬 文件内容已加载")
            user_question = input("请输入你的问题(直接回车总结内容): ").strip()

            if not user_question:
                user_question = "请总结这份文件的主要内容"

            # 构建提示词
            prompt = f"""请根据以下文件内容回答问题：

{content}

用户问题: {user_question}

请基于以上内容回答。"""

            print("\n🤖 正在分析...")
            print("\n助手: ", end="", flush=True)

            # 调用agent
            full_response = ""
            for chunk in self.agent.stream(prompt, thread_id=self.current_session_id):
                if isinstance(chunk, dict) and 'messages' in chunk:
                    for msg in chunk['messages']:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content, end='', flush=True)
                            full_response += msg.content

            print()  # 换行

        except Exception as e:
            print(f"\n❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()

    def show_help(self):
        """显示帮助信息"""
        help_text = """
┌─────────────────────────────────────────────────────────────────────┐
│                         OllamaPilot 帮助                            │
├─────────────────────────────────────────────────────────────────────┤
│  基础命令                                                           │
├─────────────────────────────────────────────────────────────────────┤
│  /help                         显示此帮助信息                       │
│  /model                        列出并切换对话模型                   │
│  /model <name>                 直接切换到指定模型                   │
│  /embedding                    切换 Embedding 模型                  │
│  /info                         显示当前状态信息                     │
│  /reload                       重新加载 .env 配置                   │
│  quit/exit/q                   退出程序                             │
├─────────────────────────────────────────────────────────────────────┤
│  会话管理 (新增)                                                    │
├─────────────────────────────────────────────────────────────────────┤
│  /new [名称]                   创建新会话                           │
│  /sessions                     列出所有会话(含数据库恢复的)          │
│  /switch <id>                  切换到指定会话                       │
│  /history [id]                 查看会话历史(默认当前会话)           │
│  /rename <名称>                重命名当前会话                       │
│  /delete <id>                  删除指定会话                         │
│  /export [id]                  导出会话为 Markdown                  │
│  /clear                        清空当前对话历史                     │
├─────────────────────────────────────────────────────────────────────┤
│  文件/图片分析                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  /upload [path]                上传并分析文件/图片                  │
│                                 支持: PDF, DOCX, TXT, MD, JPG, PNG  │
├─────────────────────────────────────────────────────────────────────┤
│  文档管理                                                           │
├─────────────────────────────────────────────────────────────────────┤
│  /docs                         列出所有文档                         │
│  /index [path]                 索引文档/文件夹(默认:knowledge_base) │
│  /index --force [path]         强制重新索引（清除旧数据）           │
│  /index --migrate              迁移旧版数据到 LightRAG 格式         │
│  /resume                       恢复失败的索引任务                   │
├─────────────────────────────────────────────────────────────────────┤
│  图例: 💾 内存会话  📦 数据库恢复的会话                              │
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

    def show_messages(self):
        """显示当前会话的消息列表"""
        if not self.agent:
            print("❌ Agent 未初始化")
            return

        history = self.agent.get_history(self.current_session_id)

        print("\n" + "=" * 60)
        print(f"📋 当前会话消息列表 (共 {len(history)} 条)")
        print("=" * 60)

        if not history:
            print("(暂无消息)")
            print("=" * 60 + "\n")
            return

        for i, msg in enumerate(history, 1):
            msg_type = type(msg).__name__
            content_preview = ""

            if hasattr(msg, 'content') and msg.content:
                content_str = str(msg.content)
                # 限制内容长度
                if len(content_str) > 100:
                    content_preview = content_str[:100] + "..."
                else:
                    content_preview = content_str
                # 替换换行符为空格，保持单行显示
                content_preview = content_preview.replace('\n', ' ')

            # 显示工具调用信息
            tool_info = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_detail = []
                for tc in msg.tool_calls:
                    tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                    args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                    if isinstance(args, dict) and args:
                        args_str = ", ".join([f"{k}={repr(v)[:50]}{'...' if len(str(v)) > 50 else ''}" for k, v in list(args.items())[:3]])
                        tool_calls_detail.append(f"{tool_name}({args_str})")
                    else:
                        tool_calls_detail.append(tool_name)
                tool_info = f" [工具: {', '.join(tool_calls_detail)}]"

            # 显示工具名称
            if hasattr(msg, 'name') and msg.name:
                tool_info += f" [名称: {msg.name}]"

            print(f"\n[{i}] {msg_type}{tool_info}")
            if content_preview:
                print(f"    {content_preview}")

        print("\n" + "=" * 60 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """处理斜杠命令"""
        if not command.startswith('/'):
            return False
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
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
            # 解析参数
            force_mode = '--force' in args or '-f' in args
            migrate_mode = '--migrate' in args or '-m' in args
            
            # 移除选项参数，获取实际路径参数
            clean_args = [a for a in args if a not in ['--force', '-f', '--migrate', '-m']]
            path_arg = clean_args[0] if clean_args else None
            
            if migrate_mode:
                self._check_and_migrate_legacy_data()
            elif path_arg:
                self.index_document(path_arg, force=force_mode)
            else:
                base_dir = Path(self.doc_manager.base_persist_dir) if self.doc_manager else None
                if base_dir and base_dir.exists():
                    needs_check = False
                    for item in base_dir.iterdir():
                        if item.is_dir():
                            index_files = list(item.glob("index_*.json"))
                            if index_files:
                                model_name = index_files[0].stem.replace("index_", "")
                                flag_file = item / f"migrated_{model_name}.flag"
                                if not flag_file.exists():
                                    needs_check = True
                                    break

                    if needs_check:
                        print("\n💡 提示: 检测到可能需要数据迁移")
                        print("   运行 '/index --migrate' 可迁移旧数据到 LightRAG 格式")
                        print("   跳过迁移不影响正常索引新文档\n")

                default_kb = Path("./data/knowledge_base")
                if default_kb.exists() and default_kb.is_dir():
                    files = self._get_files_in_directory(str(default_kb))
                    if files:
                        print(f"📁 使用默认知识库文件夹: {default_kb}")
                        print(f"   发现 {len(files)} 个文档")
                        self.index_document(str(default_kb), force=force_mode)
                    else:
                        print(f"⚠️ 默认知识库文件夹为空: {default_kb}")
                        print(f"   支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS)}")
                        choice = input("\n是否选择其他文件夹? (y/n): ").strip().lower()
                        if choice == 'y':
                            folder = self._select_folder_interactive()
                            if folder:
                                self.index_document(folder, force=force_mode)
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
                            self.index_document(folder, force=force_mode)
        
        elif cmd == '/resume':
            self.resume_failed_indexing()
        
        elif cmd == '/reload':
            self.reload_config()

        elif cmd == '/messages':
            self.show_messages()
        
        # 新增会话管理命令
        elif cmd == '/history':
            self.show_session_history(arg1)
        
        elif cmd == '/rename':
            if arg1:
                self.rename_session(arg1)
            else:
                print("⚠️ 请指定新名称，例如: /rename 项目讨论")
        
        elif cmd == '/delete':
            if arg1:
                self.delete_session(arg1)
            else:
                print("⚠️ 请指定会话ID，例如: /delete session_1")
        
        elif cmd == '/export':
            self.export_session(arg1)

        elif cmd == '/upload':
            self.upload_and_analyze(arg1)

        else:
            print(f"❌ 未知命令: {cmd}，输入 /help 查看帮助")
        
        return True
    
    def chat(self, user_input: str):
        """处理用户输入并获取回复（带超时保护）"""
        if not self.agent:
            print("❌ Agent 未初始化")
            return

        if self.history_manager:
            self._restore_history_to_agent()

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

        # 显示选中的 Skill
        active_skill = getattr(self.agent, '_active_skill_name', None)
        if active_skill:
            print(f"\n🎯 使用 Skill: {active_skill}", end="", flush=True)

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
                    
                    data = event.get("data", {})
                    if data:
                        input_data = data.get("input", data.get("chunk", ""))
                        if input_data and isinstance(input_data, dict):
                            args_preview = []
                            for k, v in list(input_data.items())[:5]:
                                v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                                args_preview.append(f"{k}={v_str}")
                            if args_preview:
                                print(f"\n   参数: {', '.join(args_preview)}", end="", flush=True)
                
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

                if self.history_manager:
                    self.history_manager.add_human_message(user_input)
                    self.history_manager.add_ai_message(full_response)
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
