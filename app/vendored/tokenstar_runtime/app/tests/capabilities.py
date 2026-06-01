"""能力探测测试 —— 全部 INFO 类型，仅占位。

第一版：未声明 capabilities 就跳过，声明了就提示"真实探针尚未接入"。
"""

from app.core.models import ProviderConfig, TestResult

STATUS_INFO = "信息"


def _probe_skip_or_todo(provider: ProviderConfig, cap_key: str, test_name: str) -> list[TestResult]:
    caps = provider.capabilities or {}
    if not caps.get(cap_key):
        return [TestResult(
            provider.name, provider.model, test_name,
            STATUS_INFO, score=None,
            message=f"{cap_key} 能力未声明，跳过探测",
            detail={"capability": cap_key, "declared": False},
        )]
    return [TestResult(
        provider.name, provider.model, test_name,
        STATUS_INFO, score=None,
        message=f"{cap_key} 能力已声明，真实探针尚未接入",
        detail={"capability": cap_key, "declared": True, "note": "probe_not_implemented"},
    )]


def run_vision_probe(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    return _probe_skip_or_todo(provider, "vision", "vision_probe")


def run_file_probe(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    return _probe_skip_or_todo(provider, "files", "file_probe")


def run_audio_probe(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    return _probe_skip_or_todo(provider, "audio", "audio_probe")


def run_embedding_probe(provider: ProviderConfig, thresholds: dict) -> list[TestResult]:
    return _probe_skip_or_todo(provider, "embeddings", "embedding_probe")
