"""API 层固定透出的中转范围。"""

VISIBLE_GROUP = "clawos"


def visible_group_where(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"{prefix}group_name=?"
