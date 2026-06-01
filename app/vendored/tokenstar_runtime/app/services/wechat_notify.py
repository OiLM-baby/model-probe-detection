"""企业微信通知模块。

支持发送 markdown 汇总；传入报告文件时会先上传为企业微信临时素材，
再把附件和摘要一起发到机器人 webhook。
"""

import logging
import os
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger("tokenstar")


class WechatNotifier:
    """企业微信通知渠道。"""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = _valid_webhook_url(webhook_url or os.environ.get("WECHAT_WEBHOOK", ""))

    def send_summary(self, content: str,
                     attachment_path: str | None = None,
                     subject: str = "") -> bool:
        return send_markdown(content, attachment_path, self.webhook_url)

def send_markdown(content: str, file_path: str | None = None, webhook_url: str | None = None) -> bool:
    webhook_url = _valid_webhook_url(webhook_url or os.environ.get("WECHAT_WEBHOOK", ""))
    if not webhook_url:
        logger.warning("WECHAT_WEBHOOK 未配置，跳过发送")
        return False
    try:
        if file_path:
            media_id = _upload_file(file_path, webhook_url)
            if media_id:
                file_resp = requests.post(
                    webhook_url,
                    json={"msgtype": "file", "file": {"media_id": media_id}},
                    timeout=20,
                )
                file_resp.raise_for_status()
                file_data = file_resp.json()
                if file_data.get("errcode") != 0:
                    logger.warning("企业微信附件消息推送失败: %s", file_data)
        response = requests.post(
            webhook_url,
            json={"msgtype": "markdown", "markdown": {"content": content}},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("errcode") != 0:
            logger.warning("企业微信 markdown 消息推送失败: %s", data)
            return False
        return True
    except Exception:
        logger.exception("企业微信通知发送失败")
        return False


def _upload_file(file_path: str, webhook_url: str):
    if not os.path.exists(file_path):
        logger.warning("企业微信附件不存在: %s", file_path)
        return None
    match = re.search(r"key=([^&]+)", webhook_url)
    if not match:
        logger.warning("无法从 WECHAT_WEBHOOK 提取 key，跳过附件上传")
        return None
    upload_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={match.group(1)}&type=file"
    try:
        with open(file_path, "rb") as handle:
            response = requests.post(
                upload_url,
                files={"media": (os.path.basename(file_path), handle)},
                timeout=30,
            )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("企业微信附件上传失败")
        return None
    if data.get("errcode") != 0:
        logger.warning("企业微信附件上传失败: %s", data)
        return None
    return data.get("media_id")


def _valid_webhook_url(value: str | None) -> str:
    url = (value or "").strip()
    if not url or url.startswith("$"):
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url
