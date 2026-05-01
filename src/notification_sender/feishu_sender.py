# -*- coding: utf-8 -*-
"""
飞书 发送提醒服务

支持两种模式：
1. App Bot API（优先）：使用 feishu_app_id + feishu_app_secret + feishu_chat_id
2. Webhook（兼容）：使用 feishu_webhook_url
"""
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from src.config import Config
from src.formatters import (
    MIN_MAX_BYTES,
    PAGE_MARKER_SAFE_BYTES,
    chunk_content_by_max_bytes,
    format_feishu_markdown,
)


logger = logging.getLogger(__name__)

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


class FeishuSender:

    def __init__(self, config: Config):
        """
        初始化飞书配置

        Args:
            config: 配置对象
        """
        self._feishu_url = getattr(config, 'feishu_webhook_url', None)
        self._feishu_secret = (getattr(config, 'feishu_webhook_secret', None) or '').strip()
        self._feishu_keyword = (getattr(config, 'feishu_webhook_keyword', None) or '').strip()
        self._feishu_max_bytes = getattr(config, 'feishu_max_bytes', 20000)
        self._webhook_verify_ssl = getattr(config, 'webhook_verify_ssl', True)

        # App Bot API 凭证
        self._feishu_app_id = getattr(config, 'feishu_app_id', None) or None
        self._feishu_app_secret = getattr(config, 'feishu_app_secret', None) or None
        self._feishu_chat_id = getattr(config, 'feishu_chat_id', None) or None

        # 缓存 tenant_access_token
        self._tenant_token: Optional[str] = None
        self._token_expires_at: float = 0

    def _get_tenant_token(self) -> Optional[str]:
        """获取 tenant_access_token（带缓存）"""
        # 如果 token 还在有效期，直接返回
        if self._tenant_token and time.time() < self._token_expires_at - 60:
            return self._tenant_token

        if not self._feishu_app_id or not self._feishu_app_secret:
            return None

        try:
            resp = requests.post(
                f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self._feishu_app_id,
                    "app_secret": self._feishu_app_secret,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get('code') == 0:
                self._tenant_token = data['tenant_access_token']
                # token 有效期通常是 2 小时
                self._token_expires_at = time.time() + 7200
                logger.info("✅ 飞书 tenant_access_token 获取成功")
                return self._tenant_token
            else:
                logger.error(f"飞书获取 token 失败 [code={data.get('code')}]: {data.get('msg')}")
                return None
        except Exception as e:
            logger.error(f"飞书获取 token 请求失败: {e}")
            return None

    def _send_via_app_bot(self, content: str) -> bool:
        """通过 App Bot API 发送消息（优先使用）"""
        token = self._get_tenant_token()
        if not token:
            logger.warning("飞书 App Bot token 获取失败，尝试回退到 webhook")
            return False

        if not self._feishu_chat_id:
            logger.error("飞书 feishu_chat_id 未配置，无法使用 App Bot API")
            return False

        # 准备卡片消息
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "📈 A股智能分析报告"
                }
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": content
                    }
                }
            ]
        }

        try:
            resp = requests.post(
                f"{FEISHU_API_BASE}/im/v1/messages?receive_id_type=chat_id",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "receive_id": self._feishu_chat_id,
                    "msg_type": "interactive",
                    "content": json.dumps(card),
                },
                timeout=30,
            )
            result = resp.json()
            if result.get('code') == 0:
                logger.info("✅ 飞书 App Bot 消息发送成功")
                return True
            else:
                logger.error(f"飞书 App Bot 返回错误 [code={result.get('code')}]: {result.get('msg')}")
                return False
        except Exception as e:
            logger.error(f"飞书 App Bot 请求失败: {e}")
            return False

    def _get_keyword_prefix(self) -> str:
        """Return the keyword prefix required by Feishu webhook security settings."""
        if not self._feishu_keyword:
            return ""
        return f"{self._feishu_keyword}\n"

    def _apply_keyword_prefix(self, content: str) -> str:
        """Prepend the optional keyword so each webhook request passes keyword checks."""
        prefix = self._get_keyword_prefix()
        if not prefix:
            return content
        return f"{prefix}{content}" if content else self._feishu_keyword

    def _build_security_fields(self) -> Dict[str, str]:
        """Build optional signing fields required by Feishu custom robot security."""
        if not self._feishu_secret:
            return {}

        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self._feishu_secret}"
        sign = base64.b64encode(
            hmac.new(
                string_to_sign.encode('utf-8'),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode('utf-8')
        return {
            "timestamp": timestamp,
            "sign": sign,
        }

    def send_to_feishu(self, content: str) -> bool:
        """
        推送消息到飞书机器人

        优先使用 App Bot API（需要配置 feishu_app_id + feishu_app_secret + feishu_chat_id），
        若未配置则回退到 Webhook 方式。

        App Bot API 使用飞书卡片消息格式，支持 Markdown 渲染。

        Args:
            content: 消息内容（Markdown 格式）

        Returns:
            是否发送成功
        """
        # 格式化内容
        formatted_content = format_feishu_markdown(content)

        # 调试日志
        logger.info(f"[飞书调试] app_id={'有' if self._feishu_app_id else '无'} app_secret={'有' if self._feishu_app_secret else '无'} chat_id={'有' if self._feishu_chat_id else '无'} webhook_url={'有' if self._feishu_url else '无'}")

        # 优先尝试 App Bot API
        if self._feishu_app_id and self._feishu_app_secret and self._feishu_chat_id:
            logger.info("使用飞书 App Bot API 发送消息...")
            if self._send_via_app_bot(formatted_content):
                return True
            logger.warning("App Bot API 失败，尝试 webhook...")

        # 回退到 Webhook 方式
        if not self._feishu_url:
            logger.warning("飞书 Webhook 未配置，跳过推送")
            return False

        max_bytes = self._feishu_max_bytes
        keyword_overhead = len(self._get_keyword_prefix().encode('utf-8'))
        effective_max_bytes = max_bytes - keyword_overhead

        if effective_max_bytes <= 0:
            logger.error("飞书关键词过长，超过单条消息允许的最大字节数，无法发送")
            return False

        content_bytes = len(formatted_content.encode('utf-8')) + keyword_overhead
        if content_bytes > max_bytes:
            min_chunk_bytes = MIN_MAX_BYTES + PAGE_MARKER_SAFE_BYTES
            if effective_max_bytes < min_chunk_bytes:
                logger.error(
                    "飞书关键词过长，剩余分片预算(%s字节)不足以安全分页发送，至少需要 %s 字节",
                    effective_max_bytes,
                    min_chunk_bytes,
                )
                return False
            logger.info(f"飞书消息内容超长({content_bytes}字节/{len(formatted_content)}字符)，将分批发送")
            return self._send_feishu_chunked(formatted_content, effective_max_bytes)

        try:
            return self._send_feishu_message(formatted_content)
        except Exception as e:
            logger.error(f"发送飞书消息失败: {e}")
            return False

    def _send_feishu_chunked(self, content: str, max_bytes: int) -> bool:
        """
        分批发送长消息到飞书（Webhook 模式）

        按股票分析块（以 --- 或 ### 分隔）智能分割，确保每批不超过限制

        Args:
            content: 完整消息内容
            max_bytes: 单条消息最大字节数

        Returns:
            是否全部发送成功
        """
        try:
            chunks = chunk_content_by_max_bytes(content, max_bytes, add_page_marker=True)
        except ValueError as e:
            logger.error("飞书消息分片失败，单片预算不足以安全分页（关键词过长或 max_bytes 过小）: %s", e)
            return False

        # 分批发送
        total_chunks = len(chunks)
        success_count = 0
        for i, chunk in enumerate(chunks, 1):
            page_marker = f"\n📄 第 {i}/{total_chunks} 页\n" if total_chunks > 1 else ""
            full_chunk = f"{page_marker}{chunk}"
            logger.info(f"发送飞书消息第 {i}/{total_chunks} 批（{len(full_chunk)} 字符）")
            if self._send_feishu_message(full_chunk):
                success_count += 1
            else:
                logger.error(f"飞书消息第 {i}/{total_chunks} 批发送失败")

        if success_count == total_chunks:
            logger.info(f"飞书消息全部发送成功（共 {total_chunks} 批）")
            return True
        else:
            logger.warning(f"飞书消息部分发送成功（{success_count}/{total_chunks}）")
            return success_count > 0

    def _send_feishu_message(self, content: str) -> bool:
        """发送单条飞书消息（Webhook 模式）"""
        prepared_content = self._apply_keyword_prefix(content)
        security_fields = self._build_security_fields()

        def _post_payload(payload: Dict[str, Any]) -> bool:
            request_payload = dict(payload)
            request_payload.update(security_fields)
            logger.debug(f"飞书请求 URL: {self._feishu_url}")
            logger.debug(f"飞书请求 payload 长度: {len(prepared_content)} 字符")

            response = requests.post(
                self._feishu_url,
                json=request_payload,
                timeout=30,
                verify=self._webhook_verify_ssl
            )

            logger.debug(f"飞书响应状态码: {response.status_code}")
            logger.debug(f"飞书响应内容: {response.text}")

            if response.status_code == 200:
                result = response.json()
                code = result.get('code') if 'code' in result else result.get('StatusCode')
                if code == 0:
                    logger.info("飞书消息发送成功")
                    return True
                else:
                    error_msg = result.get('msg') or result.get('StatusMessage', '未知错误')
                    error_code = result.get('code') or result.get('StatusCode', 'N/A')
                    logger.error(f"飞书返回错误 [code={error_code}]: {error_msg}")
                    logger.error(f"完整响应: {result}")
                    return False
            else:
                logger.error(f"飞书请求失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return False

        # 1) 优先使用交互卡片（支持 Markdown 渲染）
        card_payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "📈 股票智能分析报告"
                    }
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": prepared_content
                        }
                    }
                ]
            }
        }

        if _post_payload(card_payload):
            return True

        # 2) 回退为普通文本消息
        text_payload = {
            "msg_type": "text",
            "content": {
                "text": prepared_content
            }
        }

        return _post_payload(text_payload)
