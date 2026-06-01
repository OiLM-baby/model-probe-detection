"""基线工具：加载基线阈值、对比结果、检测偏差。"""

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Baseline:
    """一个套件的基线阈值。"""

    suite: str = ""
    p95_latency_max_ms: float | None = None
    avg_latency_max_ms: float | None = None
    hard_pass_rate_min_pct: float | None = None
    soft_score_min: float | None = None
    max_cost_per_request: float | None = None
    deviation_allowed_pct: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], suite: str = "") -> "Baseline":
        lat = data.get("latency", {}) or {}
        pr = data.get("pass_rate", {}) or {}
        cost = data.get("cost", {}) or {}
        dev = data.get("deviation", {}) or {}
        return cls(
            suite=suite,
            p95_latency_max_ms=lat.get("p95_max_ms"),
            avg_latency_max_ms=lat.get("avg_max_ms"),
            hard_pass_rate_min_pct=pr.get("hard_min_pct"),
            soft_score_min=pr.get("soft_min_score"),
            max_cost_per_request=cost.get("max_per_request_cny"),
            deviation_allowed_pct=dev.get("allowed_pct"),
        )

    def merge(self, override: "Baseline") -> "Baseline":
        """用 override 中的非 None 值覆盖当前基线。"""
        return Baseline(
            suite=override.suite or self.suite,
            p95_latency_max_ms=override.p95_latency_max_ms if override.p95_latency_max_ms is not None else self.p95_latency_max_ms,
            avg_latency_max_ms=override.avg_latency_max_ms if override.avg_latency_max_ms is not None else self.avg_latency_max_ms,
            hard_pass_rate_min_pct=override.hard_pass_rate_min_pct if override.hard_pass_rate_min_pct is not None else self.hard_pass_rate_min_pct,
            soft_score_min=override.soft_score_min if override.soft_score_min is not None else self.soft_score_min,
            max_cost_per_request=override.max_cost_per_request if override.max_cost_per_request is not None else self.max_cost_per_request,
            deviation_allowed_pct=override.deviation_allowed_pct if override.deviation_allowed_pct is not None else self.deviation_allowed_pct,
        )

    def to_dict(self, source: str = "") -> dict[str, Any]:
        """导出给报告 payload 展示的基线配置。"""
        return {
            "source": source,
            "suite": self.suite,
            "p95_latency_max_ms": self.p95_latency_max_ms,
            "avg_latency_max_ms": self.avg_latency_max_ms,
            "hard_pass_rate_min_pct": self.hard_pass_rate_min_pct,
            "soft_score_min": self.soft_score_min,
            "max_cost_per_request": self.max_cost_per_request,
            "deviation_allowed_pct": self.deviation_allowed_pct,
        }


@dataclass
class BaselineDeviations:
    """对比偏差结果。"""

    provider: str = ""
    model: str = ""
    deviations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_deviations(self) -> bool:
        return len(self.deviations) > 0


def load_baselines(config_path: str, suite: str) -> Baseline:
    """从 YAML 配置加载基线，合并 defaults + suite 特定配置。"""
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception:
        return Baseline(suite=suite)

    defaults = Baseline.from_dict(raw.get("defaults", {}) or {}, suite="defaults")
    suites_raw = raw.get("suites", {}) or {}
    suite_raw = suites_raw.get(suite, {}) or {}
    suite_baseline = Baseline.from_dict(suite_raw, suite=suite)
    return defaults.merge(suite_baseline)


def compare_to_baseline(
    summary: Any,  # ProviderSummary
    baseline: Baseline,
) -> BaselineDeviations:
    """将单个 provider 的汇总结果与基线对比，返回偏差列表。"""
    devs: list[dict[str, Any]] = []
    provider = getattr(summary, "provider", "")
    model = getattr(summary, "model", "")

    # 硬指标通过率
    if baseline.hard_pass_rate_min_pct is not None:
        total = getattr(summary, "hard_total", 0) or 0
        passed = getattr(summary, "hard_passed", 0) or 0
        if total > 0:
            rate = passed / total * 100
            if rate < baseline.hard_pass_rate_min_pct:
                devs.append({
                    "metric": "hard_pass_rate",
                    "label": "硬指标通过率",
                    "current": round(rate, 1),
                    "baseline": baseline.hard_pass_rate_min_pct,
                    "unit": "%",
                    "direction": "below",
                })

    # 软指标平均分
    if baseline.soft_score_min is not None:
        score = getattr(summary, "soft_avg_score", None)
        if score is not None and score < baseline.soft_score_min:
            devs.append({
                "metric": "soft_avg_score",
                "label": "软指标平均分",
                "current": round(score, 1),
                "baseline": baseline.soft_score_min,
                "unit": "分",
                "direction": "below",
            })

    # P95 延迟
    if baseline.p95_latency_max_ms is not None:
        p95 = getattr(summary, "p95_latency_ms", None)
        if p95 is not None and p95 > baseline.p95_latency_max_ms:
            devs.append({
                "metric": "p95_latency",
                "label": "P95 延迟",
                "current": round(p95, 1),
                "baseline": baseline.p95_latency_max_ms,
                "unit": "ms",
                "direction": "above",
            })

    # 平均延迟
    if baseline.avg_latency_max_ms is not None:
        avg = getattr(summary, "avg_latency_ms", None)
        if avg is not None and avg > baseline.avg_latency_max_ms:
            devs.append({
                "metric": "avg_latency",
                "label": "平均延迟",
                "current": round(avg, 1),
                "baseline": baseline.avg_latency_max_ms,
                "unit": "ms",
                "direction": "above",
            })

    # 成本
    if baseline.max_cost_per_request is not None:
        cost = getattr(summary, "estimated_cost", None)
        count = getattr(summary, "request_count", 0) or 1
        if cost is not None and (cost / count) > baseline.max_cost_per_request:
            devs.append({
                "metric": "cost_per_request",
                "label": "单次请求成本",
                "current": round(cost / count, 4),
                "baseline": baseline.max_cost_per_request,
                "unit": "元",
                "direction": "above",
            })

    return BaselineDeviations(provider=provider, model=model, deviations=devs)


def build_baseline_comparisons(
    summaries: list[Any],
    baseline: Baseline,
) -> list[dict[str, Any]]:
    """对所有 provider 汇总执行基线对比，返回偏差列表供报告使用。"""
    results: list[dict[str, Any]] = []
    for s in summaries:
        dev = compare_to_baseline(s, baseline)
        results.append({
            "provider": dev.provider,
            "model": dev.model,
            "has_deviations": dev.has_deviations,
            "deviations": dev.deviations,
        })
    return results
