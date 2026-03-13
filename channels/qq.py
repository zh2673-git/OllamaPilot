"""
QQ Bot 渠道实现

基于 go-cqhttp 的 HTTP API 和 WebSocket 接口。
支持私聊、群聊、图片消息、@触发等功能。

go-cqhttp 文档: https://docs.go-cqhttp.org/
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .base import Channel, ChannelMessage, ChannelAPIError


class QQChannel(Channel):
    """
    QQ Bot 渠道
    
    基于 go-cqhttp 实现，需要提前部署 go-cqhttp 服务。
    
    功能特性:
        - 私聊消息收发
        - 群聊消息收发（支持 @触发）
        - 图片消息支持
        - 长消息自动拆分（QQ限制2000字符）
    
    Example:
        >>> config = {
        ...     "api_url": "http://127.0.0.1:5700",
        ...     "ws_url": "ws://127.0.0.1:5701",
        ...     "bot_qq": "123456789",
        ...     "whitelist": ["987654321"],
        ...     "at_only_in_group": True
        ... }
        >>> channel = QQChannel(config, message_handler)
        >>> await channel.start()
    """
    
    name = "qq"
    description = "QQ Bot 渠道，基于 go-cqhttp"
    
    # QQ 消息长度限制
    MAX_MESSAGE_LENGTH = 2000
    
    def __init__(self, config: Dict[str, Any], message_handler):
        super().__init__(config, message_handler)
        self.api_url = config.get("api_url", "http://127.0.0.1:5700").rstrip("/")
        self.ws_url = config.get("ws_url", "ws://127.0.0.1:5701")
        self.bot_qq = str(config.get("bot_qq", ""))
        self.at_only_in_group = config.get("at_only_in_group", True)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动 QQ Bot"""
        self.session = aiohttp.ClientSession()
        self._running = True
        
        # 启动 WebSocket 监听
        self._create_task(self._ws_listener())
        
        print(f"✅ QQ Bot 已启动 (Bot QQ: {self.bot_qq})")
        print(f"   API: {self.api_url}")
        print(f"   WebSocket: {self.ws_url}")
    
    async def stop(self):
        """停止 QQ Bot"""
        self._running = False
        
        # 取消心跳任务
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭 WebSocket
        if self.ws and not self.ws.closed:
            await self.ws.close()
        
        # 关闭 Session
        if self.session:
            await self.session.close()
        
        # 取消所有后台任务
        await self._cancel_all_tasks()
        
        print("⏹️ QQ Bot 已停止")
    
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """
        发送消息
        
        Args:
            user_id: 用户QQ号
            content: 消息内容
            **kwargs:
                - group_id: 群号（发送群消息时）
                - reply_to: 回复的消息ID
        
        Returns:
            是否发送成功
        """
        if not content:
            return True
        
        # 长消息拆分
        messages = self._split_message(content)
        
        for msg in messages:
            try:
                if kwargs.get("group_id"):
                    # 发送群消息
                    success = await self._send_group_msg(
                        group_id=kwargs["group_id"],
                        message=msg,
                        reply_to=kwargs.get("reply_to")
                    )
                else:
                    # 发送私聊消息
                    success = await self._send_private_msg(
                        user_id=user_id,
                        message=msg,
                        reply_to=kwargs.get("reply_to")
                    )
                
                if not success:
                    return False
                
                # 避免发送过快
                if len(messages) > 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ 发送消息失败: {e}")
                return False
        
        return True
    
    async def _send_private_msg(self, user_id: str, message: str, reply_to: Optional[str] = None) -> bool:
        """发送私聊消息"""
        url = f"{self.api_url}/send_private_msg"
        
        # 构建消息段
        message_segments = [{"type": "text", "data": {"text": message}}]
        
        # 如果是回复消息
        if reply_to:
            message_segments.insert(0, {
                "type": "reply",
                "data": {"id": reply_to}
            })
        
        data = {
            "user_id": int(user_id),
            "message": message_segments,
            "auto_escape": False
        }
        
        async with self.session.post(url, json=data) as resp:
            result = await resp.json()
            if result.get("status") != "ok":
                print(f"⚠️ 发送私聊消息失败: {result.get('msg', '未知错误')}")
                return False
            return True
    
    async def _send_group_msg(self, group_id: str, message: str, reply_to: Optional[str] = None) -> bool:
        """发送群消息"""
        url = f"{self.api_url}/send_group_msg"
        
        # 构建消息段
        message_segments = [{"type": "text", "data": {"text": message}}]
        
        # 如果是回复消息
        if reply_to:
            message_segments.insert(0, {
                "type": "reply",
                "data": {"id": reply_to}
            })
        
        data = {
            "group_id": int(group_id),
            "message": message_segments,
            "auto_escape": False
        }
        
        async with self.session.post(url, json=data) as resp:
            result = await resp.json()
            if result.get("status") != "ok":
                print(f"⚠️ 发送群消息失败: {result.get('msg', '未知错误')}")
                return False
            return True
    
    def _split_message(self, content: str, max_length: int = None) -> List[str]:
        """
        拆分长消息
        
        QQ 单条消息限制 2000 字符，超过需要拆分。
        
        Args:
            content: 消息内容
            max_length: 最大长度，默认 MAX_MESSAGE_LENGTH
        
        Returns:
            拆分后的消息列表
        """
        if max_length is None:
            max_length = self.MAX_MESSAGE_LENGTH
        
        if len(content) <= max_length:
            return [content]
        
        # 按段落拆分，尽量保持完整段落
        paragraphs = content.split("\n")
        messages = []
        current_msg = ""
        
        for para in paragraphs:
            # 如果当前段落本身超过限制，需要强制拆分
            if len(para) > max_length:
                if current_msg:
                    messages.append(current_msg)
                    current_msg = ""
                
                # 强制拆分段落
                for i in range(0, len(para), max_length):
                    messages.append(para[i:i + max_length])
                continue
            
            # 检查添加当前段落后是否超过限制
            if len(current_msg) + len(para) + 1 > max_length:
                messages.append(current_msg)
                current_msg = para
            else:
                if current_msg:
                    current_msg += "\n"
                current_msg += para
        
        # 添加最后一条
        if current_msg:
            messages.append(current_msg)
        
        return messages
    
    async def _ws_listener(self):
        """WebSocket 消息监听"""
        while self._running:
            try:
                async with self.session.ws_connect(self.ws_url) as ws:
                    self.ws = ws
                    print("🔗 QQ WebSocket 已连接")
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                await self._handle_event(data)
                            except json.JSONDecodeError:
                                print(f"⚠️ 收到无效的 JSON: {msg.data[:100]}")
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"⚠️ WebSocket 错误: {ws.exception()}")
                            break
                        
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            print("⚠️ WebSocket 连接已关闭")
                            break
                            
            except aiohttp.ClientError as e:
                print(f"⚠️ WebSocket 连接失败: {e}")
            except Exception as e:
                print(f"⚠️ WebSocket 异常: {e}")
            
            if self._running:
                print("🔄 5秒后重新连接...")
                await asyncio.sleep(5)
    
    async def _handle_event(self, data: Dict[str, Any]):
        """处理 QQ 事件"""
        post_type = data.get("post_type")
        
        if post_type == "message":
            # 消息事件
            await self._handle_message_event(data)
        
        elif post_type == "meta_event":
            # 元事件（心跳等）
            pass
        
        elif post_type == "notice":
            # 通知事件
            pass
        
        elif post_type == "request":
            # 请求事件
            pass
    
    async def _handle_message_event(self, data: Dict[str, Any]):
        """处理消息事件"""
        message_type = data.get("message_type")
        
        if message_type == "private":
            await self._handle_private_message(data)
        elif message_type == "group":
            await self._handle_group_message(data)
    
    async def _handle_private_message(self, data: Dict[str, Any]):
        """处理私聊消息"""
        # 解析消息
        message = self._parse_message(data)
        
        print(f"📨 [QQ私聊] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        # 处理消息
        response = await self.handle_message(message)
        
        if response:
            # 发送回复
            await self.send_message(
                user_id=message.user_id,
                content=response,
                reply_to=message.message_id
            )
    
    async def _handle_group_message(self, data: Dict[str, Any]):
        """处理群聊消息"""
        # 解析消息
        message = self._parse_message(data)
        
        # 检查是否@机器人
        if self.at_only_in_group and not message.at_me:
            return
        
        print(f"📨 [QQ群:{message.group_id}] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        # 处理消息
        response = await self.handle_message(message)
        
        if response:
            # 发送回复
            await self.send_message(
                user_id=message.user_id,
                content=response,
                group_id=message.group_id,
                reply_to=message.message_id
            )
    
    def _parse_message(self, data: Dict[str, Any]) -> ChannelMessage:
        """解析 QQ 消息为标准格式"""
        message_type = data.get("message_type", "private")
        
        # 提取纯文本内容
        content, at_me = self._extract_text_and_check_at(data.get("message", []))
        
        # 提取图片
        images = self._extract_images(data.get("message", []))
        
        return ChannelMessage(
            message_id=str(data.get("message_id", "")),
            user_id=str(data.get("user_id", "")),
            user_name=data.get("sender", {}).get("nickname", ""),
            content=content,
            message_type=message_type,
            group_id=str(data.get("group_id", "")) if message_type == "group" else None,
            raw_data=data,
            timestamp=datetime.now(),
            images=images,
            at_me=at_me
        )
    
    def _extract_text_and_check_at(self, message_segments: List[Dict]) -> tuple:
        """
        从消息段中提取文本并检查是否@机器人
        
        Returns:
            (text, at_me)
        """
        texts = []
        at_me = False
        
        for seg in message_segments:
            seg_type = seg.get("type")
            seg_data = seg.get("data", {})
            
            if seg_type == "text":
                texts.append(seg_data.get("text", ""))
            
            elif seg_type == "at":
                # 检查是否@机器人
                if str(seg_data.get("qq", "")) == self.bot_qq:
                    at_me = True
        
        return " ".join(texts).strip(), at_me
    
    def _extract_images(self, message_segments: List[Dict]) -> List[str]:
        """从消息段中提取图片URL"""
        images = []
        
        for seg in message_segments:
            if seg.get("type") == "image":
                url = seg.get("data", {}).get("url", "")
                if url:
                    images.append(url)
        
        return images
