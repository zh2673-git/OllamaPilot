"""
会话存储管理模块

管理会话的持久化存储，支持从 SQLite 数据库恢复会话列表
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .session import Session


class SessionStore:
    """
    会话存储管理器
    
    负责：
    - 从 SQLite 数据库恢复会话列表
    - 保存会话元数据
    - 删除会话数据
    - 导出会话历史
    """
    
    def __init__(self, db_path: str = "./data/sessions/conversations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn
    
    def restore_sessions(self, model_name: str = "unknown", embedding_model: Optional[str] = None) -> Dict[str, Session]:
        """
        从数据库恢复所有会话
        
        Returns:
            Dict[str, Session]: 会话字典 {session_id: Session}
        """
        sessions = {}
        
        if not self.db_path.exists():
            return sessions
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # 查询所有不同的 thread_id 和最新的检查点时间
            cursor.execute("""
                SELECT thread_id, MAX(datetime(json_extract(metadata, '$.timestamp'))) as last_time
                FROM checkpoints
                GROUP BY thread_id
            """)
            
            rows = cursor.fetchall()
            
            for row in rows:
                thread_id = row[0]
                last_time = row[1]
                
                # 统计消息数量（通过检查点数量估算）
                cursor.execute("""
                    SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?
                """, (thread_id,))
                checkpoint_count = cursor.fetchone()[0]
                
                # 尝试获取第一条消息的时间作为创建时间
                cursor.execute("""
                    SELECT datetime(json_extract(metadata, '$.timestamp'))
                    FROM checkpoints 
                    WHERE thread_id = ?
                    ORDER BY rowid ASC LIMIT 1
                """, (thread_id,))
                first_row = cursor.fetchone()
                created_time = first_row[0] if first_row else last_time
                
                # 解析时间
                try:
                    updated_at = datetime.fromisoformat(last_time.replace('Z', '+00:00')) if last_time else datetime.now()
                    created_at = datetime.fromisoformat(created_time.replace('Z', '+00:00')) if created_time else datetime.now()
                except:
                    updated_at = datetime.now()
                    created_at = datetime.now()
                
                # 生成会话名称（基于 thread_id）
                if thread_id == "default":
                    name = "默认会话"
                elif thread_id.startswith("session_"):
                    # 尝试提取序号
                    parts = thread_id.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        name = f"会话 {parts[1]}"
                    else:
                        name = f"会话 {thread_id[:8]}"
                else:
                    name = f"会话 {thread_id[:8]}"
                
                session = Session(
                    session_id=thread_id,
                    name=name,
                    model_name=model_name,
                    embedding_model=embedding_model,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=checkpoint_count * 2  # 估算：每个检查点约2条消息
                )
                session._source = "database"  # 标记来源
                
                sessions[thread_id] = session
            
        except Exception as e:
            print(f"⚠️  恢复会话失败: {e}")
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        从数据库删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否成功删除
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # 删除该会话的所有检查点
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            
            # 删除相关的 writes
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
            
            conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"❌ 删除会话失败: {e}")
            return False
    
    def get_session_history(self, session_id: str, limit: int = 100) -> List[Dict]:
        """
        获取会话历史记录
        
        Args:
            session_id: 会话ID
            limit: 返回的最大记录数
            
        Returns:
            List[Dict]: 消息列表
        """
        messages = []
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT checkpoint, metadata, parent_checkpoint_id
                FROM checkpoints 
                WHERE thread_id = ?
                ORDER BY rowid DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            
            for row in rows:
                checkpoint_blob = row[0]
                metadata_blob = row[1]
                
                try:
                    # 尝试解析 msgpack 格式
                    try:
                        import msgpack
                        checkpoint = msgpack.unpackb(checkpoint_blob, raw=False)
                    except:
                        # 回退到 JSON
                        checkpoint = json.loads(checkpoint_blob)
                    
                    # 解析 metadata
                    try:
                        metadata = json.loads(metadata_blob) if metadata_blob else {}
                    except:
                        metadata = {}
                    
                    # 提取消息
                    channel_values = checkpoint.get("channel_values", {})
                    msgs = channel_values.get("messages", [])
                    
                    for msg in msgs:
                        msg_data = None
                        
                        # 处理 msgpack ExtType (LangChain 消息对象)
                        if hasattr(msg, 'code') and hasattr(msg, 'data'):
                            # 这是 ExtType，尝试解析
                            try:
                                inner = msgpack.unpackb(msg.data, raw=False)
                                # ExtType 解析后是列表: [module, class, data_dict, ...]
                                if isinstance(inner, list) and len(inner) >= 3:
                                    msg_data = inner[2]  # 消息数据在索引 2
                                elif isinstance(inner, dict):
                                    msg_data = inner
                            except:
                                continue
                        elif isinstance(msg, dict):
                            msg_data = msg
                        
                        # 提取消息内容
                        if isinstance(msg_data, dict):
                            msg_type = msg_data.get("type", "")
                            content = msg_data.get("content", "")
                            
                            # 根据模块名推断类型
                            if not msg_type and hasattr(msg, 'data'):
                                try:
                                    inner = msgpack.unpackb(msg.data, raw=False)
                                    if isinstance(inner, list) and len(inner) >= 1:
                                        module = inner[0]
                                        if 'human' in module:
                                            msg_type = 'human'
                                        elif 'ai' in module or 'assistant' in module:
                                            msg_type = 'ai'
                                        elif 'system' in module:
                                            msg_type = 'system'
                                except:
                                    pass
                            
                            if content and len(content) > 5:  # 过滤空消息
                                messages.append({
                                    "type": msg_type,
                                    "content": content[:200] + "..." if len(content) > 200 else content,
                                    "timestamp": metadata.get("timestamp", ""),
                                    "step": metadata.get("step", -1)
                                })
                except Exception as e:
                    continue
            
            # 反转顺序，按时间正序排列
            messages.reverse()
            
        except Exception as e:
            print(f"❌ 获取会话历史失败: {e}")
        
        return messages
    
    def export_session(self, session_id: str, output_dir: str = "./exports") -> Optional[str]:
        """
        导出会话为 Markdown 文件
        
        Args:
            session_id: 会话ID
            output_dir: 输出目录
            
        Returns:
            Optional[str]: 导出的文件路径
        """
        try:
            messages = self.get_session_history(session_id, limit=1000)
            
            if not messages:
                return None
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{session_id}_{timestamp}.md"
            filepath = output_path / filename
            
            # 写入 Markdown
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 会话导出: {session_id}\n\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                
                for msg in messages:
                    msg_type = msg.get("type", "")
                    content = msg.get("content", "")
                    
                    if msg_type == "human":
                        f.write(f"**用户:** {content}\n\n")
                    elif msg_type == "ai":
                        f.write(f"**助手:** {content}\n\n")
                    elif msg_type == "system":
                        f.write(f"*[系统]* {content[:100]}...\n\n")
                    else:
                        f.write(f"*{msg_type}:* {content}\n\n")
                
                f.write("---\n\n")
                f.write(f"共 {len(messages)} 条消息\n")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 导出会话失败: {e}")
            return None
    
    def get_session_stats(self) -> Dict:
        """获取会话统计信息"""
        stats = {
            "total_sessions": 0,
            "total_messages": 0,
            "db_size_mb": 0
        }
        
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # 会话数
            cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
            result = cursor.fetchone()
            stats["total_sessions"] = result[0] if result else 0
            
            # 检查点数（估算消息数）
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            result = cursor.fetchone()
            stats["total_messages"] = (result[0] * 2) if result else 0  # 估算
            
            # 数据库大小
            if self.db_path.exists():
                stats["db_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            
        except Exception as e:
            print(f"⚠️  获取统计信息失败: {e}")
        
        return stats
    
    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
