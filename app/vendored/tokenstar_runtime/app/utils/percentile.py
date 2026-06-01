"""百分位等通用统计工具。"""



def percentile(values: list[float], p: int) -> float | None:
    """第 p 百分位，使用线性插值（与 numpy 默认一致）。"""
    if not values:
        return None
    ordered = sorted(values)
    k = (p / 100) * (len(ordered) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(ordered):
        return ordered[f] + c * (ordered[f + 1] - ordered[f])
    return ordered[f]
