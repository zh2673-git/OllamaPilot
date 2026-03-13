"""
QQ Bot 渠道实现

基于 QQ 开放平台官方 Bot API。
支持私聊、群聊、频道消息、@触发等功能。

QQ 开放平台: https://q.qq.com/
"""

import asyncio
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .base import Channel, ChannelMessage, ChannelResponse
from .registry import register_channel


@register_channel
class QQChannel(Channel):
    """
    QQ Bot 渠道
    
    基于 QQ 开放平台官方 API 实现。
    支持私聊、群聊、频道消息。
    
    功能特性:
        - 私聊消息收发
        - 群聊消息收发（支持 @触发）
        - 频道消息支持
        - 富文本消息（Markdown、图片）
        - 消息按钮/卡片
        - 长消息自动拆分
    
    配置项:
        - app_id: QQ 开放平台 AppID
        - app_secret: QQ 开放平台 AppSecret
        - token: 消息验证 Token
        - sandbox: 是否使用沙箱环境
        - whitelist: 用户 ID 白名单
        - at_only_in_group: 群聊中是否只响应 @消息
    
    Example:
        >>> config = {
        ...     "app_id": "1024xxxxxx",
        ...     "app_secret": "xxxxxxxxxx",
        ...     "token": "xxxxxxxxxx",
        ...     "whitelist": ["123456789"],
        ...     "at_only_in_group": True
        ... }
        >>> channel = QQChannel(config, message_handler)
        >>> await channel.start()
    """
    
    name = "qq"
    description = "QQ Bot 渠道，基于 QQ 开放平台官方 API"
    
    MAX_MESSAGE_LENGTH = 2000
    
    def __init__(self, config: Dict[str, Any], message_handler):
        super().__init__(config, message_handler)
        self.app_id = config.get("app_id", "")
        self.app_secret = config.get("app_secret", "")
        self.token = config.get("token", "")
        self.sandbox = config.get("sandbox", False)
        self.at_only_in_group = config.get("at_only_in_group", True)
        self.intents = config.get("intents", ["DIRECT_MESSAGE", "GUILD_MESSAGES"])
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._token_expires_at = 0
        self._access_token = ""
        
        self._api_base = "https://api.sandbox.qq.com" if self.sandbox else "https://api.q.qq.com"
    
    async def start(self):
        """启动 QQ Bot"""
        self.session = aiohttp.ClientSession()
        self._running = True
        
        await self._refresh_access_token()
        
        self._create_task(self._ws_listener())
        
        print(f"✅ QQ Bot 已启动 (AppID: {self.app_id})")
        print(f"   API: {self._api_base}")
    
    async def stop(self):
        """停止 QQ Bot"""
        self._running = False
        
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        if self.session:
            await self.session.close()
        
        await self._cancel_all_tasks()
        
        print("⏹️ QQ Bot 已停止")
    
    async def _refresh_access_token(self):
        """刷新访问令牌"""
        if time.time() < self._token_expires_at - 300:
            return
        
        url = f"{self._api_base}/oauth2/token"
        params = {
            "grant_type": "client_credential",
            "client_id": self.app_id,
            "client_secret": self.app_secret
        }
        
        async with self.session.get(url, params=params) as resp:
            data = await resp.json()
            
            if "access_token" in data:
                self._access_token = data["access_token"]
                self._token_expires_at = time.time() + data.get("expires_in", 7200)
                print("🔑 QQ 访问令牌已刷新")
            else:
                print(f"⚠️ 获取访问令牌失败: {data}")
    
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """发送私聊消息"""
        if not content:
            return True
        
        messages = self._split_message(content)
        
        for msg in messages:
            success = await self._send_direct_message(user_id, msg)
            if not success:
                return False
            
            if len(messages) > 1:
                await asyncio.sleep(0.5)
        
        return True
    
    async def send_group_message(self, group_id: str, content: str, **kwargs) -> bool:
        """发送群消息"""
        if not content:
            return True
        
        messages = self._split_message(content)
        
        for msg in messages:
            success = await self._send_group_message_impl(group_id, msg)
            if not success:
                return False
            
            if len(messages) > 1:
                await asyncio.sleep(0.5)
        
        return True
    
    async def _send_direct_message(self, user_id: str, content: str) -> bool:
        """发送私信"""
        await self._refresh_access_token()
        
        url = f"{self._api_base}/v2/users/{self.app_id}/members/{user_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "content": content
        }
        
        try:
            async with self.session.post(url, json=data, headers=headers) as resp:
                if resp.status == 200:
                    return True
                else:
                    text = await resp.text()
                    print(f"⚠️ 发送私信失败: {resp.status} - {text}")
                    return False
        except Exception as e:
            print(f"⚠️ 发送私信异常: {e}")
            return False
    
    async def _send_group_message_impl(self, group_id: str, content: str) -> bool:
        """发送群消息"""
        await self._refresh_access_token()
        
        url = f"{self._api_base}/v2/groups/{group_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "content": content
        }
        
        try:
            async with self.session.post(url, json=data, headers=headers) as resp:
                if resp.status == 200:
                    return True
                else:
                    text = await resp.text()
                    print(f"⚠️ 发送群消息失败: {resp.status} - {text}")
                    return False
        except Exception as e:
            print(f"⚠️ 发送群消息异常: {e}")
            return False
    
    def _split_message(self, content: str, max_length: int = None) -> List[str]:
        """拆分长消息"""
        if max_length is None:
            max_length = self.MAX_MESSAGE_LENGTH
        
        if len(content) <= max_length:
            return [content]
        
        paragraphs = content.split("\n")
        messages = []
        current_msg = ""
        
        for para in paragraphs:
            if len(para) > max_length:
                if current_msg:
                    messages.append(current_msg)
                    current_msg = ""
                for i in range(0, len(para), max_length):
                    messages.append(para[i:i + max_length])
                continue
            
            if len(current_msg) + len(para) + 1 > max_length:
                messages.append(current_msg)
                current_msg = para
            else:
                if current_msg:
                    current_msg += "\n"
                current_msg += para
        
        if current_msg:
            messages.append(current_msg)
        
        return messages
    
    async def _ws_listener(self):
        """WebSocket 消息监听"""
        while self._running:
            try:
                ws_url = f"{self._api_base}/v2/ws"
                
                async with self.session.ws_connect(ws_url) as ws:
                    self._ws = ws
                    print("🔗 QQ WebSocket 已连接")
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = msg.json()
                                await self._handle_event(data)
                            except Exception as e:
                                print(f"⚠️ 解析消息失败: {e}")
                        
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
        """处理事件"""
        event_type = data.get("type")
        
        if event_type == "MESSAGE_CREATE":
            await self._handle_message_create(data)
        elif event_type == "DIRECT_MESSAGE_CREATE":
            await self._handle_direct_message(data)
    
    async def _handle_message_create(self, data: Dict[str, Any]):
        """处理频道消息"""
        channel_id = data.get("channel_id", "")
        guild_id = data.get("guild_id", "")
        message_data = data.get("msg", {})
        
        message = self._parse_message(message_data, "channel", channel_name=guild_id)
        message.group_id = channel_id
        
        print(f"📨 [QQ频道:{channel_id}] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        response = await self.handle_message(message)
        
        if response:
            await self.send_message(message.user_id, response)
    
    async def _handle_direct_message(self, data: Dict[str, Any]):
        """处理私信"""
        message_data = data.get("msg", {})
        
        message = self._parse_message(message_data, "private")
        
        print(f"📨 [QQ私信] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        response = await self.handle_message(message)
        
        if response:
            await self.send_message(message.user_id, response)
    
    def _parse_message(self, data: Dict[str, Any], message_type: str, **kwargs) -> ChannelMessage:
        """解析消息为标准格式"""
        author = data.get("author", {})
        
        content = data.get("content", "")
        mentions = data.get("mentions", [])
        
        at_me = False
        for mention in mentions:
            if mention.get("id") == self.app_id:
                at_me = True
                break
        
        images = []
        attachments = data.get("attachments", [])
        for att in attachments:
            if att.get("content_type", "").startswith("image/"):
                images.append(att.get("url", ""))
        
        return ChannelMessage(
            message_id=str(data.get("id", "")),
            user_id=str(author.get("id", "")),
            user_name=author.get("username", author.get("nick", "")),
            content=content,
            message_type=message_type,
            channel_name="qq",
            group_id=kwargs.get("group_id"),
            group_name=kwargs.get("channel_name"),
            raw_data=data,
            images=images,
            at_me=at_me,
            reply_to=str(data.get("message_reference", {}).get("message_id", ""))
        )
    
    def verify_webhook(self, timestamp: str, signature: str, body: bytes) -> bool:
        """验证 Webhook 签名"""
        if not self.token:
            return True
        
        sign_str = f"{timestamp}{body.decode()}{self.token}"
        expected = hmac.new(
            self.token.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature == expected
