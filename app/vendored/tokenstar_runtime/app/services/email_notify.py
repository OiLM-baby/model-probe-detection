"""邮件通知模块。

把 TokenStar 汇总结果和 HTML 报告附件发送到配置的 SMTP 邮箱。
通知失败不影响主流程，只记录日志。
"""

import logging
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.models import EmailConfig

logger = logging.getLogger("tokenstar")


def _configured(value: str | None) -> bool:
    text = (value or "").strip()
    return bool(text) and not text.startswith("$")


class EmailNotifier:
    """邮件通知渠道。"""

    def __init__(self, config: EmailConfig):
        self.config = config

    def send_summary(self, content: str,
                     attachment_path: str | None = None,
                     subject: str = "") -> bool:
        subj = subject or "TokenStar 测试汇总"
        return send_email(self.config, subj, content, attachment_path)

def send_email(config: EmailConfig, subject: str, content: str, attachment_path: str | None = None) -> bool:
    if (
        not _configured(config.smtp_server)
        or not _configured(config.username)
        or not _configured(config.password)
        or not config.receivers
    ):
        logger.warning("邮箱配置不完整，跳过发送")
        return False

    try:
        message = MIMEMultipart()
        message["From"] = config.sender or config.username
        message["To"] = ", ".join(config.receivers)
        message["Subject"] = subject
        message.attach(MIMEText(content, "html", "utf-8"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as handle:
                attachment = MIMEApplication(handle.read())
            attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment_path))
            message.attach(attachment)

        with smtplib.SMTP_SSL(config.smtp_server, config.smtp_port) as server:
            server.login(config.username, config.password)
            server.send_message(message)
        return True
    except Exception:
        logger.exception("邮件发送失败")
        return False
