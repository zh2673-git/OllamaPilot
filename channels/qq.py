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
from .renderers import get_renderer, QQPlainTextRenderer


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
        
        # 初始化 Markdown 渲染器（使用 QQ 纯文本优化版）
        self.renderer = QQPlainTextRenderer()
        
        # QQ Bot v2 Intents (位运算)
        # 1<<25: C2C_MESSAGE_CREATE (单聊消息)
        # 1<<26: GROUP_AT_MESSAGE_CREATE (群聊@消息)
        # 1<<30: DIRECT_MESSAGE_CREATE (频道私信)
        intents_config = config.get("intents", ["C2C_MESSAGE_CREATE", "GROUP_AT_MESSAGE_CREATE"])
        self.intents = self._calculate_intents(intents_config)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._token_expires_at = 0
        self._access_token = ""
        
        # QQ Bot API 地址
        # 沙箱: https://sandbox.api.sgroup.qq.com
        # 正式: https://api.sgroup.qq.com
        if self.sandbox:
            self._api_base = "https://sandbox.api.sgroup.qq.com"
            self._ws_base = "wss://sandbox.api.sgroup.qq.com"
        else:
            self._api_base = "https://api.sgroup.qq.com"
            self._ws_base = "wss://api.sgroup.qq.com"
    
    def _calculate_intents(self, intents_list: List[str]) -> int:
        """计算 intents 值"""
        intent_map = {
            "GUILDS": 1 << 0,
            "GUILD_MEMBERS": 1 << 1,
            "GUILD_MESSAGES": 1 << 9,
            "GUILD_MESSAGE_REACTIONS": 1 << 10,
            "DIRECT_MESSAGE": 1 << 12,
            "C2C_MESSAGE_CREATE": 1 << 25,  # 单聊消息
            "GROUP_AT_MESSAGE_CREATE": 1 << 26,  # 群聊@消息
            "DIRECT_MESSAGE_CREATE": 1 << 30,  # 频道私信
        }
        result = 0
        for intent in intents_list:
            if intent in intent_map:
                result |= intent_map[intent]
        return result
    
    async def start(self):
        """启动 QQ Bot"""
        print("🔄 正在初始化 QQ Bot...")
        self.session = aiohttp.ClientSession()
        self._running = True
        
        print("🔄 正在刷新 access_token...")
        try:
            await self._refresh_access_token()
            print("✅ access_token 刷新完成")
        except Exception as e:
            print(f"⚠️ 刷新 access_token 出错: {e}")
            import traceback
            traceback.print_exc()
        
        print("🔄 正在启动 WebSocket 监听...")
        try:
            self._create_task(self._ws_listener())
            print("✅ WebSocket 监听已启动")
        except Exception as e:
            print(f"⚠️ 启动 WebSocket 出错: {e}")
            import traceback
            traceback.print_exc()
        
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
    
    def _generate_access_token(self) -> str:
        """生成 QQ Bot AccessToken
        
        QQ Bot 使用 AppID + Token 生成 JWT 格式的 AccessToken
        格式: base64(appid.token).base64(timestamp).HMACSHA256
        """
        import time
        import hmac
        import hashlib
        import base64
        
        # 时间戳（秒）
        timestamp = int(time.time())
        
        # 第一部分: appid.token
        header = f"{self.app_id}.{self.token}"
        header_b64 = base64.b64encode(header.encode()).decode()
        
        # 第二部分: 时间戳
        timestamp_b64 = base64.b64encode(str(timestamp).encode()).decode()
        
        # 第三部分: HMACSHA256 签名
        message = f"{header_b64}.{timestamp_b64}"
        signature = hmac.new(
            self.app_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode()
        
        # 组合成 JWT
        access_token = f"{header_b64}.{timestamp_b64}.{signature_b64}"
        return access_token
    
    async def _refresh_access_token(self):
        """刷新访问令牌 - 从 QQ 官方 API 获取"""
        if time.time() < self._token_expires_at - 300:
            return
        
        print("🔑 正在获取 QQ Bot AccessToken...")
        
        try:
            url = "https://bots.qq.com/app/getAppAccessToken"
            data = {
                "appId": self.app_id,
                "clientSecret": self.app_secret,
            }
            
            async with self.session.post(url, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    self._access_token = result.get("access_token")
                    expires_in = int(result.get("expires_in", 3600))
                    self._token_expires_at = time.time() + expires_in
                    print(f"🔑 QQ AccessToken 已获取，有效期 {expires_in} 秒")
                else:
                    text = await resp.text()
                    print(f"⚠️ 获取 AccessToken 失败: {resp.status} - {text}")
        except Exception as e:
            print(f"⚠️ 获取 AccessToken 失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_headers(self) -> Dict[str, str]:
        """获取认证头"""
        return {
            "Authorization": f"QQBot {self._access_token}",
            "Content-Type": "application/json"
        }
    
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
    
    async def _send_c2c_message(self, user_id: str, content: str, msg_id: str = None) -> bool:
        """发送单聊消息 (C2C)"""
        await self._refresh_access_token()

        # QQ Bot v2 API: /v2/users/{user_id}/messages
        url = f"{self._api_base}/v2/users/{user_id}/messages"
        headers = self._get_headers()

        # 使用 Markdown 渲染器转换内容（优化为 QQ 友好的纯文本格式）
        rendered = self.renderer.render(content)
        message_content = rendered.content if rendered.content else content

        # 构建消息数据
        data = {
            "content": message_content,
            "msg_type": 0,  # 文本消息
        }

        if msg_id:
            data["msg_id"] = msg_id  # 被动消息需要回复的消息ID

        try:
            async with self.session.post(url, json=data, headers=headers) as resp:
                if resp.status == 200:
                    print(f"✅ 单聊消息已发送给 {user_id}")
                    return True
                else:
                    text = await resp.text()
                    print(f"⚠️ 发送单聊消息失败: {resp.status} - {text}")
                    # 处理内容违规错误(40034)
                    if resp.status == 400 and "40034" in text:
                        print(f"   [ContentFilter] 消息内容被QQ平台拦截，尝试发送简化提示...")
                        fallback_msg = "抱歉，回复内容包含敏感信息被平台拦截。请尝试询问其他话题。"
                        fallback_data = {
                            "content": fallback_msg,
                            "msg_type": 0,
                        }
                        if msg_id:
                            fallback_data["msg_id"] = msg_id
                        async with self.session.post(url, json=fallback_data, headers=headers) as fallback_resp:
                            if fallback_resp.status == 200:
                                print(f"✅ 已发送简化提示给用户 {user_id}")
                                return True
                    return False
        except Exception as e:
            print(f"⚠️ 发送单聊消息异常: {e}")
            return False

    async def _send_group_message(self, group_id: str, content: str, msg_id: str = None) -> bool:
        """发送群消息"""
        await self._refresh_access_token()

        # QQ Bot v2 API: /v2/groups/{group_id}/messages
        url = f"{self._api_base}/v2/groups/{group_id}/messages"
        headers = self._get_headers()

        # 使用 Markdown 渲染器转换内容（优化为 QQ 友好的纯文本格式）
        rendered = self.renderer.render(content)
        message_content = rendered.content if rendered.content else content

        # 构建消息数据
        data = {
            "content": message_content,
            "msg_type": 0,
        }

        if msg_id:
            data["msg_id"] = msg_id

        try:
            async with self.session.post(url, json=data, headers=headers) as resp:
                if resp.status == 200:
                    print(f"✅ 群消息已发送到 {group_id}")
                    return True
                else:
                    text = await resp.text()
                    print(f"⚠️ 发送群消息失败: {resp.status} - {text}")
                    # 处理内容违规错误(40034)
                    if resp.status == 400 and "40034" in text:
                        print(f"   [ContentFilter] 消息内容被QQ平台拦截，尝试发送简化提示...")
                        fallback_msg = "抱歉，回复内容包含敏感信息被平台拦截。请尝试询问其他话题。"
                        fallback_data = {
                            "content": fallback_msg,
                            "msg_type": 0,
                        }
                        if msg_id:
                            fallback_data["msg_id"] = msg_id
                        async with self.session.post(url, json=fallback_data, headers=headers) as fallback_resp:
                            if fallback_resp.status == 200:
                                print(f"✅ 已发送简化提示到群 {group_id}")
                                return True
                    return False
        except Exception as e:
            print(f"⚠️ 发送群消息异常: {e}")
            return False

    async def _send_direct_message(self, user_id: str, content: str) -> bool:
        """发送私信 (频道私信)"""
        await self._refresh_access_token()

        url = f"{self._api_base}/v2/users/{self.app_id}/members/{user_id}/messages"
        headers = self._get_headers()
        data = {
            "content": content,
            "msg_type": 0,
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
        """发送群消息 (旧版实现)"""
        return await self._send_group_message(group_id, content)
    
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
    
    async def _get_gateway_url(self) -> str:
        """获取 WebSocket 网关地址"""
        url = f"{self._api_base}/gateway"
        headers = self._get_headers()
        
        async with self.session.get(url, headers=headers) as resp:
            data = await resp.json()
            if resp.status == 200 and "url" in data:
                return data["url"]
            else:
                print(f"⚠️ 获取网关地址失败: {data}")
                return None
    
    async def _ws_listener(self):
        """WebSocket 消息监听"""
        # 先获取网关地址
        print("📡 正在获取 QQ WebSocket 网关地址...")
        ws_url = await self._get_gateway_url()
        
        if not ws_url:
            print("⚠️ 无法获取网关地址，5秒后重试...")
            await asyncio.sleep(5)
            return
        
        print(f"📡 正在连接 QQ WebSocket: {ws_url}")
        
        while self._running:
            try:
                async with self.session.ws_connect(ws_url) as ws:
                    self._ws = ws
                    print("🔗 QQ WebSocket 已连接，等待消息...")
                    
                    # 发送鉴权消息
                    # QQ Bot v2 鉴权格式: "QQBot {access_token}"
                    auth_token = f"QQBot {self._access_token}"
                    auth_msg = {
                        "op": 2,  # Identify
                        "d": {
                            "token": auth_token,
                            "intents": self.intents,
                            "shard": [0, 1],
                            "properties": {
                                "$os": "windows",
                                "$browser": "ollamapilot",
                                "$device": "ollamapilot"
                            }
                        }
                    }
                    print(f"📡 发送鉴权消息: token={auth_token[:30]}..., intents={self.intents}")
                    await ws.send_json(auth_msg)
                    print(f"📡 已发送鉴权消息")
                    
                    # 心跳任务
                    heartbeat_interval = None
                    last_sequence = None
                    
                    async def send_heartbeat():
                        """发送心跳"""
                        while self._running and heartbeat_interval:
                            await asyncio.sleep(heartbeat_interval / 1000)
                            if not self._running:
                                break
                            heartbeat_msg = {
                                "op": 1,
                                "d": last_sequence
                            }
                            await ws.send_json(heartbeat_msg)
                            print("💓 发送心跳")
                    
                    # 启动心跳任务
                    heartbeat_task = None
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = msg.json()
                                op = data.get("op")
                                
                                # 处理 Hello 消息 (op: 10)
                                if op == 10:
                                    heartbeat_interval = data.get("d", {}).get("heartbeat_interval")
                                    print(f"� 收到 Hello，心跳间隔: {heartbeat_interval}ms")
                                    # 启动心跳
                                    if heartbeat_task:
                                        heartbeat_task.cancel()
                                    heartbeat_task = asyncio.create_task(send_heartbeat())
                                
                                # 处理心跳确认 (op: 11)
                                elif op == 11:
                                    print("💓 心跳确认")
                                
                                # 处理事件 (op: 0)
                                elif op == 0:
                                    last_sequence = data.get("s")
                                    event_type = data.get("t")
                                    
                                    # 处理 READY 事件
                                    if event_type == "READY":
                                        d = data.get("d", {})
                                        user = d.get("user", {})
                                        print(f"✅ QQ Bot 已就绪: {user.get('username')} ({user.get('id')})")
                                    else:
                                        await self._handle_event(data)
                                
                                # 处理错误
                                elif op == 9:
                                    print(f"⚠️ 鉴权失败: {data}")
                                    break
                                
                                else:
                                    print(f"📨 收到消息: {data}")
                                    
                            except Exception as e:
                                print(f"⚠️ 解析消息失败: {e}")
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"⚠️ WebSocket 错误: {ws.exception()}")
                            break
                        
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            print("⚠️ WebSocket 连接已关闭")
                            break
                    
                    # 取消心跳任务
                    if heartbeat_task:
                        heartbeat_task.cancel()
                            
            except aiohttp.ClientError as e:
                print(f"⚠️ WebSocket 连接失败: {e}")
                import traceback
                traceback.print_exc()
            except Exception as e:
                print(f"⚠️ WebSocket 异常: {e}")
            
            if self._running:
                print("🔄 5秒后重新连接...")
                await asyncio.sleep(5)
    
    async def _handle_event(self, data: Dict[str, Any]):
        """处理事件"""
        event_type = data.get("t")  # QQ Bot 使用 "t" 字段表示事件类型
        
        print(f"📨 收到事件: {event_type}")
        
        if event_type == "C2C_MESSAGE_CREATE":
            await self._handle_c2c_message(data)
        elif event_type == "GROUP_AT_MESSAGE_CREATE":
            await self._handle_group_at_message(data)
        elif event_type == "DIRECT_MESSAGE_CREATE":
            await self._handle_direct_message(data)
        elif event_type == "MESSAGE_CREATE":
            await self._handle_message_create(data)
    
    async def _handle_c2c_message(self, data: Dict[str, Any]):
        """处理单聊消息 (C2C_MESSAGE_CREATE)"""
        d = data.get("d", {})
        author = d.get("author", {})
        user_id = author.get("id", "")
        user_name = author.get("username", "")
        content = d.get("content", "")
        msg_id = d.get("id", "")

        print(f"📨 [QQ单聊] {user_name}({user_id}): {content[:50]}...")

        # 创建消息对象
        message = ChannelMessage(
            message_id=msg_id,
            user_id=user_id,
            user_name=user_name,
            content=content,
            message_type="private",
            channel_name="qq",
            raw_data=d
        )

        # 检查白名单
        if not self.check_permission(user_id):
            print(f"⛔ 用户 {user_id} 不在白名单中")
            return

        # 处理消息
        print(f"🤖 正在处理消息...")
        try:
            response = await self.message_handler(message)
            print(f"✅ 消息处理完成，准备发送回复")
            
            # 从 ChannelResponse 中提取内容
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            print(f"📤 回复内容长度: {len(content)} 字符")
            
            if content:
                success = await self._send_c2c_message(user_id, content, msg_id)
                if success:
                    print(f"✅ 回复已发送")
                else:
                    print(f"❌ 回复发送失败")
            else:
                print(f"⚠️ 回复内容为空，不发送")
        except Exception as e:
            print(f"⚠️ 处理消息失败: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_group_at_message(self, data: Dict[str, Any]):
        """处理群聊@消息 (GROUP_AT_MESSAGE_CREATE)"""
        d = data.get("d", {})
        author = d.get("author", {})
        user_id = author.get("id", "")
        user_name = author.get("username", "")
        content = d.get("content", "")
        msg_id = d.get("id", "")
        group_id = d.get("group_id", "")

        print(f"📨 [QQ群:{group_id}] {user_name}({user_id}): {content[:50]}...")

        # 检查白名单
        if not self.check_permission(user_id):
            print(f"⛔ 用户 {user_id} 不在白名单中")
            return

        # 创建消息对象
        message = ChannelMessage(
            message_id=msg_id,
            user_id=user_id,
            user_name=user_name,
            content=content,
            message_type="group",
            channel_name="qq",
            group_id=group_id,
            raw_data=d
        )

        # 处理消息
        try:
            response = await self.message_handler(message)
            
            # 从 ChannelResponse 中提取内容
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            if content:
                await self._send_group_message(group_id, content, msg_id)
        except Exception as e:
            print(f"⚠️ 处理群消息失败: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_message_create(self, data: Dict[str, Any]):
        """处理频道消息"""
        channel_id = data.get("channel_id", "")
        guild_id = data.get("guild_id", "")
        message_data = data.get("d", {})  # QQ Bot v2 使用 "d" 字段
        
        message = self._parse_message(message_data, "channel", channel_name=guild_id)
        message.group_id = channel_id
        
        print(f"📨 [QQ频道:{channel_id}] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        try:
            response = await self.message_handler(message)
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            if content:
                await self.send_message(message.user_id, content)
        except Exception as e:
            print(f"⚠️ 处理频道消息失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_direct_message(self, data: Dict[str, Any]):
        """处理私信"""
        message_data = data.get("d", {})  # QQ Bot v2 使用 "d" 字段
        
        message = self._parse_message(message_data, "private")
        
        print(f"📨 [QQ私信] {message.user_name}({message.user_id}): {message.content[:50]}...")
        
        try:
            response = await self.message_handler(message)
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            if content:
                await self.send_message(message.user_id, content)
        except Exception as e:
            print(f"⚠️ 处理私信失败: {e}")
            import traceback
            traceback.print_exc()
    
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
