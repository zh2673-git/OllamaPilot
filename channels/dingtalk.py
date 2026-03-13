"""
钉钉 Bot 渠道实现

基于钉钉开放平台企业内部机器人 API。
支持群聊、@触发、Markdown 消息等功能。

钉钉开放平台文档: https://open.dingtalk.com/document/
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from .base import Channel, ChannelMessage, ChannelAPIError
from .registry import register_channel


@register_channel
class DingTalkChannel(Channel):
    """
    钉钉 Bot 渠道

    基于钉钉开放平台实现，需要创建企业内部应用并添加机器人能力。

    功能特性:
        - 群聊消息收发（支持 @触发）
        - Markdown 消息
        - 卡片消息
        - 企业内部机器人

    Example:
        >>> config = {
        ...     "app_key": "xxxxxxxxxx",
        ...     "app_secret": "xxxxxxxxxx",
        ...     "robot_code": "",  # 可选
        ...     "whitelist": ["User123"]
        ... }
        >>> channel = DingTalkChannel(config, message_handler)
        >>> await channel.start()
    """

    name = "dingtalk"
    description = "钉钉 Bot 渠道"

    # 钉钉 API 基础地址
    BASE_URL = "https://oapi.dingtalk.com"

    def __init__(self, config: Dict[str, Any], message_handler):
        super().__init__(config, message_handler)
        self.app_key = config.get("app_key", "")
        self.app_secret = config.get("app_secret", "")
        self.robot_code = config.get("robot_code", "")

        self.session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expire_time: Optional[datetime] = None

    async def start(self):
        """启动钉钉 Bot"""
        if not self.app_key or not self.app_secret:
            raise ChannelAPIError("钉钉配置错误: 缺少 app_key 或 app_secret")

        self.session = aiohttp.ClientSession()
        self._running = True

        # 获取访问令牌
        await self._refresh_access_token()

        # 启动 Webhook 服务器
        self._create_task(self._start_webhook_server())

        print(f"✅ 钉钉 Bot 已启动 (App Key: {self.app_key[:10]}...)")

    async def stop(self):
        """停止钉钉 Bot"""
        self._running = False

        # 关闭 Session
        if self.session:
            await self.session.close()

        # 取消所有后台任务
        await self._cancel_all_tasks()

        print("⏹️ 钉钉 Bot 已停止")

    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """
        发送私聊消息
        
        Args:
            user_id: 用户 UserID
            content: 消息内容
            **kwargs:
                - msg_type: 消息类型 (text/markdown)
        
        Returns:
            是否发送成功
        """
        if not content:
            return True
        
        try:
            await self._ensure_token()
            return await self._send_to_user(user_id, content, kwargs.get("msg_type", "text"))
        except Exception as e:
            print(f"❌ 发送钉钉消息失败: {e}")
            return False
    
    async def send_group_message(self, group_id: str, content: str, **kwargs) -> bool:
        """
        发送群消息
        
        Args:
            group_id: 群聊 ID (open_conversation_id)
            content: 消息内容
            **kwargs:
                - msg_type: 消息类型 (text/markdown)
                - at_users: @用户列表
        
        Returns:
            是否发送成功
        """
        if not content:
            return True
        
        try:
            await self._ensure_token()
            return await self._send_group_message_impl(
                chat_id=group_id,
                content=content,
                msg_type=kwargs.get("msg_type", "text"),
                at_users=kwargs.get("at_users", [])
            )
        except Exception as e:
            print(f"❌ 发送钉钉群消息失败: {e}")
            return False
    
    async def _send_to_user(self, user_id: str, content: str, msg_type: str = "text") -> bool:
        """发送私信给用户"""
        url = f"{self.BASE_URL}/topapi/message/corpconversation/asyncsend_v2"
        
        params = {"access_token": self._access_token}
        
        if msg_type == "markdown":
            content_json = {"title": "消息", "text": content}
        else:
            content_json = {"content": content}
        
        data = {
            "agent_id": self.config.get("agent_id", ""),
            "userid_list": user_id,
            "msgtype": msg_type,
            msg_type: content_json
        }
        
        async with self.session.post(url, params=params, json=data) as resp:
            result = await resp.json()
            if result.get("errcode") != 0:
                print(f"⚠️ 发送钉钉私信失败: {result.get('errmsg', '未知错误')}")
                return False
            return True

    async def _send_group_message_impl(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        at_users: List[str] = None
    ) -> bool:
        """发送群消息"""
        url = f"{self.BASE_URL}/robot/send"

        # 构建消息体
        if msg_type == "markdown":
            message_body = {
                "msgtype": "markdown",
                "markdown": {
                    "title": content[:20] + "..." if len(content) > 20 else content,
                    "text": content
                }
            }
        else:
            message_body = {
                "msgtype": "text",
                "text": {
                    "content": content
                }
            }

        # 添加 @ 信息
        if at_users:
            message_body["at"] = {
                "atUserIds": at_users,
                "isAtAll": False
            }

        # 使用 webhook 方式发送
        # 注意：钉钉群机器人需要通过 webhook 发送，这里需要额外的 webhook_token
        # 实际使用时需要在配置中提供 webhook_token
        webhook_token = self.config.get("webhook_token", "")
        if not webhook_token:
            print("⚠️ 钉钉群机器人需要配置 webhook_token")
            return False

        webhook_url = f"{self.BASE_URL}/robot/send?access_token={webhook_token}"

        async with self.session.post(webhook_url, json=message_body) as resp:
            result = await resp.json()
            if result.get("errcode") != 0:
                print(f"⚠️ 发送钉钉群消息失败: {result.get('errmsg', '未知错误')}")
                return False
            return True

    async def _refresh_access_token(self):
        """刷新访问令牌"""
        url = f"{self.BASE_URL}/gettoken"

        params = {
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        async with self.session.get(url, params=params) as resp:
            result = await resp.json()
            if result.get("errcode") != 0:
                raise ChannelAPIError(f"获取钉钉 access_token 失败: {result.get('errmsg')}")

            self._access_token = result.get("access_token")
            expire = result.get("expires_in", 7200)  # 默认2小时
            self._token_expire_time = datetime.now() + timedelta(seconds=expire - 300)  # 提前5分钟刷新

            print("🔑 钉钉 access_token 已刷新")

    async def _ensure_token(self):
        """确保 access_token 有效"""
        if not self._access_token or datetime.now() >= self._token_expire_time:
            await self._refresh_access_token()

    async def _start_webhook_server(self):
        """启动 Webhook 服务器接收钉钉事件"""
        from aiohttp import web

        async def handle_webhook(request):
            """处理钉钉 Webhook 请求"""
            try:
                body = await request.text()
                data = json.loads(body)

                # 验证签名（如果配置了 app_secret）
                timestamp = request.headers.get("timestamp", "")
                sign = request.headers.get("sign", "")

                if self.app_secret and timestamp and sign:
                    expected_sign = self._calculate_sign(timestamp, self.app_secret)
                    if sign != expected_sign:
                        return web.Response(status=401, text="Invalid signature")

                # 处理挑战请求（配置回调 URL 时）
                if data.get("msgtype") == "verification":
                    challenge = data.get("challenge", "")
                    return web.json_response({"challenge": challenge})

                # 处理消息回调
                if data.get("msgtype") == "text":
                    await self._handle_message_event(data)

                return web.Response(status=200, text="OK")

            except Exception as e:
                print(f"⚠️ 处理钉钉 Webhook 失败: {e}")
                return web.Response(status=500, text="Internal Server Error")

        app = web.Application()
        app.router.add_post("/webhook/dingtalk", handle_webhook)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "0.0.0.0", 8082)  # 默认端口 8082
        await site.start()

        print("🔗 钉钉 Webhook 服务器已启动 (端口: 8082)")

        # 保持运行
        while self._running:
            await asyncio.sleep(1)

        await runner.cleanup()

    def _calculate_sign(self, timestamp: str, secret: str) -> str:
        """计算钉钉签名"""
        string_to_sign = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(
            secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return sign

    async def _handle_message_event(self, data: Dict[str, Any]):
        """处理消息事件"""
        # 钉钉机器人主要在群聊中使用
        await self._handle_group_message(data)

    async def _handle_group_message(self, data: Dict[str, Any]):
        """处理群聊消息"""
        text_content = data.get("text", {}).get("content", "")
        sender_staff_id = data.get("senderStaffId", "")
        sender_nick = data.get("senderNick", "未知用户")
        conversation_id = data.get("conversationId", "")
        msg_id = data.get("msgId", "")

        # 检查是否 @机器人
        at_users = data.get("atUsers", [])
        at_me = any(u.get("staffId") == self.robot_code for u in at_users)

        if self.config.get("at_only_in_group", True) and not at_me:
            return

        # 构建标准消息格式
        channel_msg = ChannelMessage(
            message_id=msg_id,
            user_id=sender_staff_id,
            user_name=sender_nick,
            content=text_content,
            message_type="group",
            group_id=conversation_id,
            raw_data=data,
            timestamp=datetime.now(),
            at_me=at_me
        )

        print(f"📨 [钉钉群:{channel_msg.group_id[:10]}...] {channel_msg.user_name}: {channel_msg.content[:50]}...")

        # 处理消息
        response = await self.handle_message(channel_msg)

        if response:
            # 发送回复
            await self.send_message(
                user_id=channel_msg.user_id,
                content=response,
                chat_id=channel_msg.group_id
            )
