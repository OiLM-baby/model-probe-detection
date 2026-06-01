"""日志工具。

创建项目统一 logger，把运行日志按日期写入 logs/ 目录，
供测试执行、通知发送和异常记录复用。
"""

import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.utils.timezone import BEIJING_TZ, beijing_log_date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class BeijingFormatter(logging.Formatter):
    converter = staticmethod(lambda timestamp: datetime.fromtimestamp(timestamp, BEIJING_TZ).timetuple())


def setup_logger(
    env: str = "prod",
    run_id: str | None = None,
    log_dir: str = "logs",
    level: str = "INFO",
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
):
    logger = logging.getLogger("tokenstar")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = BeijingFormatter(
        f"[%(asctime)s] [%(levelname)s] [env={env}] [run_id={run_id or '-'}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    target_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
    target_dir = os.path.join(target_dir, env)
    os.makedirs(target_dir, exist_ok=True)
    log_file = os.path.join(target_dir, f"tokenstar_{beijing_log_date()}.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
