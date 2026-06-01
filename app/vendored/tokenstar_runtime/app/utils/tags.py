"""标签工具：自动打标 + 合并标签列表。"""

from functools import lru_cache


def _caps_key(capabilities: dict | None) -> tuple[tuple[str, object], ...]:
    """将 capabilities dict 转为可哈希的 tuple，供 lru_cache 使用。"""
    if not capabilities:
        return ()
    return tuple(sorted(capabilities.items()))


@lru_cache(maxsize=256)
def _auto_tags_cached(provider_name: str, model: str, group: str,
                      suite: str, timeout: int,
                      caps_key: tuple[tuple[str, object], ...]) -> tuple[str, ...]:
    """auto_tags 的可缓存版本（返回 tuple 以支持 lru_cache）。"""
    tags: list[str] = []
    name_lower = provider_name.lower()
    if "internal" in name_lower or "selfhost" in name_lower or "本地" in provider_name:
        tags.append("selfhosted")
    if "proxy" in name_lower or "中转" in provider_name or "gateway" in name_lower:
        tags.append("proxy")
    if "test" in name_lower or "测试" in provider_name:
        tags.append("test")
    if suite:
        if suite in ("political_sensitivity", "all"):
            tags.append("political")
        if suite in ("audit", "model_audit", "cache_audit", "concurrency_audit"):
            tags.append("audit")
        if suite in ("daily", "availability"):
            tags.append("monitor")
    if timeout > 120:
        tags.append("slow")
    if model:
        from app.utils.model_family import detect_family
        vendor, family = detect_family(model)
        if vendor != "unknown":
            tags.append(f"vendor:{vendor}")
            tags.append(f"family:{family}")
    if group and group != provider_name:
        tags.append(f"group:{group}")
    caps: dict[str, object] = dict(caps_key) if caps_key else {}
    if caps.get("vision"):
        tags.append("vision")
    if caps.get("files"):
        tags.append("files")
    return tuple(tags)


def auto_tags(provider_name: str, model: str, group: str,
              suite: str = "", timeout: int = 0,
              capabilities: dict | None = None) -> list[str]:
    """根据中转名/模型/分组/套件/能力自动生成标签（带缓存）。"""
    ck = _caps_key(capabilities)
    return list(_auto_tags_cached(provider_name, model, group, suite, timeout, ck))


def merge_tags(base: list[str], extra: list[str]) -> list[str]:
    """合并两组标签，去重排序。"""
    return sorted(set(base) | set(extra))
