"""Timezone helpers for TokenStar API/report timestamps."""

from datetime import datetime
from zoneinfo import ZoneInfo

BEIJING_TZ = ZoneInfo("Asia/Shanghai")
BEIJING_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def beijing_now_str() -> str:
    return datetime.now(BEIJING_TZ).strftime(BEIJING_TIME_FORMAT)


def beijing_from_timestamp(ts: float, fmt: str = BEIJING_TIME_FORMAT) -> str:
    return datetime.fromtimestamp(ts, tz=BEIJING_TZ).strftime(fmt)


def beijing_log_date() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d")
