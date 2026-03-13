"""
飞书 Bot 渠道实现

基于飞书开放平台 Bot API。
支持私聊、群聊、富文本消息等功能。

飞书开放平台文档: https://open.feishu.cn/document/home/index
"""

import asyncio
import base64
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .base import Channel, ChannelMessage, ChannelAPIError
from .registry import register_channel


@register_channel
class FeishuChannel(Channel):
    """
    飞书 Bot 渠道
    
    基于飞书开放平台实现，需要创建企业自建应用并开启机器人能力。
    
    功能特性:
        - 私聊消息收发
        - 群聊消息收发（支持 @触发）
        - 富文本消息
        - 卡片消息
    
    Example:
        >>> config = {
        ...     "app_id": "cli_xxxxxxxxxx",
        ...     "app_secret": "xxxxxxxxxx",
        ...     "encrypt_key": "",  # 可选
        ...     "verification_token": "",  # 可选
        ...     "whitelist": ["ou_xxxxxxxxxx"]
        ... }
        >>> channel = FeishuChannel(config, message_handler)
        >>> await channel.start()
    """
    
    name = "feishu"
    description = "飞书 Bot 渠道"
    
    # 飞书 API 基础地址
    BASE_URL = "https://open.feishu.cn/open-apis"
    
    def __init__(self, config: Dict[str, Any], message_handler):
        super().__init__(config, message_handler)
        self.app_id = config.get("app_id", "")
        self.app_secret = config.get("app_secret", "")
        self.encrypt_key = config.get("encrypt_key", "")
        self.verification_token = config.get("verification_token", "")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expire_time: Optional[datetime] = None
        self._webhook_server: Optional[Any] = None
    
    async def start(self):
        """启动飞书 Bot"""
        if not self.app_id or not self.app_secret:
            raise ChannelAPIError("飞书配置错误: 缺少 app_id 或 app_secret")
        
        self.session = aiohttp.ClientSession()
        self._running = True
        
        # 获取访问令牌
        await self._refresh_access_token()
        
        # 启动 WebSocket 或 HTTP 服务器接收事件
        # 这里使用 HTTP Webhook 方式
        self._create_task(self._start_webhook_server())
        
        print(f"✅ 飞书 Bot 已启动 (App ID: {self.app_id[:10]}...)")
    
    async def stop(self):
        """停止飞书 Bot"""
        self._running = False
        
        # 停止 Webhook 服务器
        if self._webhook_server:
            await self._webhook_server.stop()
        
        # 关闭 Session
        if self.session:
            await self.session.close()
        
        # 取消所有后台任务
        await self._cancel_all_tasks()
        
        print("⏹️ 飞书 Bot 已停止")
    
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """
        发送私聊消息
        
        Args:
            user_id: 用户 OpenID
            content: 消息内容
            **kwargs:
                - msg_type: 消息类型 (text/post/card)
        
        Returns:
            是否发送成功
        """
        if not content:
            return True
        
        try:
            await self._ensure_token()
            return await self._send_private_message(
                user_id=user_id,
                content=content,
                msg_type=kwargs.get("msg_type", "text")
            )
        except Exception as e:
            print(f"❌ 发送飞书消息失败: {e}")
            return False
    
    async def send_group_message(self, group_id: str, content: str, **kwargs) -> bool:
        """
        发送群消息
        
        Args:
            group_id: 群聊 ID (chat_id)
            content: 消息内容
            **kwargs:
                - msg_type: 消息类型 (text/post/card)
        
        Returns:
            是否发送成功
        """
        if not content:
            return True
        
        try:
            await self._ensure_token()
            return await self._send_group_message(
                chat_id=group_id,
                content=content,
                msg_type=kwargs.get("msg_type", "text")
            )
        except Exception as e:
            print(f"❌ 发送飞书群消息失败: {e}")
            return False
    
    async def _send_private_message(self, user_id: str, content: str, msg_type: str = "text") -> bool:
        """发送私聊消息"""
        url = f"{self.BASE_URL}/im/v1/messages"
        
        # 构建消息体
        if msg_type == "text":
            message_body = {
                "text": content
            }
        else:
            message_body = {"text": content}
        
        params = {"receive_id_type": "open_id"}
        data = {
            "receive_id": user_id,
            "msg_type": msg_type,
            "content": json.dumps(message_body, ensure_ascii=False)
        }
        
        headers = {"Authorization": f"Bearer {self._access_token}"}
        
        async with self.session.post(url, params=params, json=data, headers=headers) as resp:
            result = await resp.json()
            if result.get("code") != 0:
                print(f"⚠️ 发送飞书私聊消息失败: {result.get('msg', '未知错误')}")
                return False
            return True
    
    async def _send_group_message(self, chat_id: str, content: str, msg_type: str = "text") -> bool:
        """发送群消息"""
        url = f"{self.BASE_URL}/im/v1/messages"
        
        # 构建消息体
        if msg_type == "text":
            message_body = {
                "text": content
            }
        else:
            message_body = {"text": content}
        
        params = {"receive_id_type": "chat_id"}
        data = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": json.dumps(message_body, ensure_ascii=False)
        }
        
        headers = {"Authorization": f"Bearer {self._access_token}"}
        
        async with self.session.post(url, params=params, json=data, headers=headers) as resp:
            result = await resp.json()
            if result.get("code") != 0:
                print(f"⚠️ 发送飞书群消息失败: {result.get('msg', '未知错误')}")
                return False
            return True
    
    async def _refresh_access_token(self):
        """刷新访问令牌"""
        url = f"{self.BASE_URL}/auth/v3/tenant_access_token/internal"
        
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        async with self.session.post(url, json=data) as resp:
            result = await resp.json()
            if result.get("code") != 0:
                raise ChannelAPIError(f"获取飞书 access_token 失败: {result.get('msg')}")
            
            self._access_token = result.get("tenant_access_token")
            expire = result.get("expire", 7200)  # 默认2小时
            self._token_expire_time = datetime.now() + timedelta(seconds=expire - 300)  # 提前5分钟刷新
            
            print("🔑 飞书 access_token 已刷新")
    
    async def _ensure_token(self):
        """确保 access_token 有效"""
        if not self._access_token or datetime.now() >= self._token_expire_time:
            await self._refresh_access_token()
    
    async def _start_webhook_server(self):
        """启动 Webhook 服务器接收飞书事件"""
        from aiohttp import web
        
        async def handle_webhook(request):
            """处理飞书 Webhook 请求"""
            try:
                body = await request.text()
                data = json.loads(body)
                
                # 验证请求（如果有配置 verification_token）
                if self.verification_token:
                    token = request.headers.get("X-Lark-Token", "")
                    if token != self.verification_token:
                        return web.Response(status=401, text="Unauthorized")
                
                # 处理挑战请求（配置 Webhook 时）
                if data.get("type") == "url_verification":
                    challenge = data.get("challenge", "")
                    return web.json_response({"challenge": challenge})
                
                # 处理事件回调
                if data.get("header", {}).get("event_type") == "im.message.receive_v1":
                    event_data = data.get("event", {})
                    await self._handle_message_event(event_data)
                
                return web.Response(status=200, text="OK")
            
            except Exception as e:
                print(f"⚠️ 处理飞书 Webhook 失败: {e}")
                return web.Response(status=500, text="Internal Server Error")
        
        app = web.Application()
        app.router.add_post("/webhook/feishu", handle_webhook)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, "0.0.0.0", 8081)  # 默认端口 8081
        await site.start()
        
        print("🔗 飞书 Webhook 服务器已启动 (端口: 8081)")
        
        # 保持运行
        while self._running:
            await asyncio.sleep(1)
        
        await runner.cleanup()
    
    async def _handle_message_event(self, event_data: Dict[str, Any]):
        """处理消息事件"""
        message = event_data.get("message", {})
        sender = event_data.get("sender", {})
        
        # 解析消息
        msg_type = message.get("message_type", "")
        
        if msg_type == "p2p":
            # 私聊消息
            await self._handle_private_message(event_data)
        elif msg_type == "group":
            # 群聊消息
            await self._handle_group_message(event_data)
    
    async def _handle_private_message(self, event_data: Dict[str, Any]):
        """处理私聊消息"""
        message = event_data.get("message", {})
        sender = event_data.get("sender", {})
        
        # 构建标准消息格式
        channel_msg = ChannelMessage(
            message_id=message.get("message_id", ""),
            user_id=sender.get("sender_id", {}).get("open_id", ""),
            user_name=sender.get("sender_id", {}).get("user_id", "未知用户"),
            content=self._extract_text_content(message),
            message_type="private",
            raw_data=event_data,
            timestamp=datetime.now()
        )
        
        print(f"📨 [飞书私聊] {channel_msg.user_name}({channel_msg.user_id[:10]}...): {channel_msg.content[:50]}...")
        
        # 处理消息
        response = await self.handle_message(channel_msg)
        
        if response:
            # 发送回复
            await self.send_message(
                user_id=channel_msg.user_id,
                content=response
            )
    
    async def _handle_group_message(self, event_data: Dict[str, Any]):
        """处理群聊消息"""
        message = event_data.get("message", {})
        sender = event_data.get("sender", {})
        
        # 检查是否 @机器人
        mentions = message.get("mentions", [])
        at_me = any(m.get("key") == "@_user_1" for m in mentions)
        
        if self.config.get("at_only_in_group", True) and not at_me:
            return
        
        # 构建标准消息格式
        channel_msg = ChannelMessage(
            message_id=message.get("message_id", ""),
            user_id=sender.get("sender_id", {}).get("open_id", ""),
            user_name=sender.get("sender_id", {}).get("user_id", "未知用户"),
            content=self._extract_text_content(message),
            message_type="group",
            group_id=message.get("chat_id", ""),
            raw_data=event_data,
            timestamp=datetime.now(),
            at_me=at_me
        )
        
        print(f"📨 [飞书群:{channel_msg.group_id[:10]}...] {channel_msg.user_name}: {channel_msg.content[:50]}...")
        
        # 处理消息
        response = await self.handle_message(channel_msg)
        
        if response:
            # 发送回复
            await self.send_message(
                user_id=channel_msg.user_id,
                content=response,
                chat_id=channel_msg.group_id
            )
    
    def _extract_text_content(self, message: Dict[str, Any]) -> str:
        """从飞书消息中提取文本内容"""
        msg_type = message.get("content_type", "")
        content_str = message.get("content", "{}")
        
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            return content_str
        
        if msg_type == "text":
            return content.get("text", "")
        elif msg_type == "post":
            # 富文本消息，提取所有文本
            return self._extract_post_text(content)
        else:
            return f"[不支持的消息类型: {msg_type}]"
    
    def _extract_post_text(self, content: Dict) -> str:
        """从富文本消息中提取文本"""
        texts = []
        
        # 飞书富文本格式较复杂，这里简化处理
        # 实际使用时可能需要更复杂的解析
        def extract_text_from_element(element):
            if isinstance(element, str):
                return element
            elif isinstance(element, list):
                return "".join(extract_text_from_element(e) for e in element)
            elif isinstance(element, dict):
                if "text" in element:
                    return element["text"]
                return "".join(extract_text_from_element(v) for v in element.values())
            return ""
        
        return extract_text_from_element(content)


# 导入 timedelta 用于 token 过期计算
from datetime import timedelta
