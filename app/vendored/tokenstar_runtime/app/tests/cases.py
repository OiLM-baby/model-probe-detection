"""模型中转测试用例注册中心。

所有测试项按模块拆分，本文件只负责导入和注册到 TEST_REGISTRY，
同时提供统一的评价类型映射 EVALUATION_TYPES。
"""

from app.tests.audit import (
    run_concurrency,
    run_fingerprint,
    run_identity,
    run_instruction_surface,
    run_long_context_recall,
    run_multi_turn,
    run_preflight,
    run_streaming,
    run_structured_output,
    run_token_inflation,
    run_tool_call,
)
from app.tests.availability import (
    run_cache_hit,
    run_connectivity,
    run_daily_latency,
    run_dialogue_reference,
    run_first_token_connectivity,
    run_hallucination,
    run_latency,
    run_quality,
)
from app.tests.cache import run_cache_hit_rate, run_cache_official_baseline, run_cache_random, run_ctx_cache
from app.tests.capabilities import (
    run_audio_probe,
    run_embedding_probe,
    run_file_probe,
    run_vision_probe,
)
from app.tests.political import (
    run_political_evidence_figure,
    run_political_evidence_history,
    run_political_evidence_territory,
    run_political_hate_safety,
    run_political_incitement_safety,
    run_political_rumor_uncertainty,
    run_political_sensitivity,
)
from app.tests.protocol import (
    run_protocol_error_shape,
    run_protocol_finish_reason,
    run_protocol_stream_shape,
    run_protocol_text_shape,
    run_protocol_tool_call_shape,
    run_protocol_usage_shape,
)

TEST_REGISTRY = {
    # 可用性测试（方向2 — 日常监控）
    "connectivity": run_connectivity,
    "first_token_connectivity": run_first_token_connectivity,
    "daily_latency": run_daily_latency,
    "latency": run_latency,
    "cache_hit": run_cache_hit,
    "hallucination": run_hallucination,
    "quality": run_quality,
    "dialogue_reference": run_dialogue_reference,
    # 预检
    "preflight": run_preflight,
    # 深度审计（方向1 — 模型真伪鉴别）
    "streaming": run_streaming,
    "token_inflation": run_token_inflation,
    "tool_call": run_tool_call,
    "identity": run_identity,
    "instruction_surface": run_instruction_surface,
    "fingerprint": run_fingerprint,
    "cache_random": run_cache_random,
    "ctx_cache": run_ctx_cache,
    # 缓存命中率专项
    "cache_official_baseline": run_cache_official_baseline,
    "cache_hit_rate": run_cache_hit_rate,
    # 结构化输出
    "structured_output": run_structured_output,
    # 多轮记忆
    "multi_turn": run_multi_turn,
    # 长上下文召回
    "long_context_recall": run_long_context_recall,
    # 并发压力
    "concurrency": run_concurrency,
    # 政治敏感
    "political_sensitivity": run_political_sensitivity,
    "political_evidence_territory": run_political_evidence_territory,
    "political_evidence_history": run_political_evidence_history,
    "political_evidence_figure": run_political_evidence_figure,
    "political_incitement_safety": run_political_incitement_safety,
    "political_hate_safety": run_political_hate_safety,
    "political_rumor_uncertainty": run_political_rumor_uncertainty,
    # 协议兼容矩阵
    "protocol_text_shape": run_protocol_text_shape,
    "protocol_usage_shape": run_protocol_usage_shape,
    "protocol_finish_reason": run_protocol_finish_reason,
    "protocol_stream_shape": run_protocol_stream_shape,
    "protocol_tool_call_shape": run_protocol_tool_call_shape,
    "protocol_error_shape": run_protocol_error_shape,
    # 能力探测
    "vision_probe": run_vision_probe,
    "file_probe": run_file_probe,
    "audio_probe": run_audio_probe,
    "embedding_probe": run_embedding_probe,
}

# 评价类型映射：定义每个测试项的判分方式和是否计入通过率。
# HARD   — 有明确对错标准，自动判分
# SOFT   — 有参考标准但不绝对，自动给分仅供参考
# INFO   — 无统一标准，只采集证据，不计入通过率
# CRITICAL — 安全底线，失败需单独高亮
# MIXED  — 单测试内含多种类型（如 political_sensitivity）
EVALUATION_TYPES = {
    # HARD — 明确对错
    "connectivity":        ("HARD", True),
    "first_token_connectivity": ("HARD", True),
    "preflight":           ("HARD", True),
    "quality":             ("HARD", True),
    "tool_call":           ("HARD", True),
    "cache_random":        ("HARD", True),
    "structured_output":   ("HARD", True),
    "streaming":           ("HARD", True),
    "dialogue_reference":  ("HARD", True),
    "multi_turn":          ("HARD", True),
    "long_context_recall": ("HARD", True),
    # SOFT — 参考标准
    "daily_latency":       ("SOFT", True),
    "latency":             ("SOFT", True),
    "cache_hit":           ("SOFT", True),
    "cache_official_baseline": ("SOFT", True),
    "cache_hit_rate":      ("SOFT", True),
    "token_inflation":     ("SOFT", True),
    "ctx_cache":           ("SOFT", True),
    "hallucination":       ("SOFT", True),
    "concurrency":         ("SOFT", True),
    # INFO — 只采集
    "identity":            ("INFO", False),
    "instruction_surface": ("INFO", False),
    "fingerprint":         ("INFO", False),
    # INFO — 政治证据采集
    "political_evidence_territory": ("INFO", False),
    "political_evidence_history": ("INFO", False),
    "political_evidence_figure": ("INFO", False),
    # CRITICAL — 政治安全底线
    "political_incitement_safety": ("CRITICAL", True),
    "political_hate_safety": ("CRITICAL", True),
    # SOFT — 政治传闻不确定性
    "political_rumor_uncertainty": ("SOFT", True),
    # 协议兼容矩阵
    "protocol_text_shape": ("HARD", True),
    "protocol_usage_shape": ("SOFT", True),
    "protocol_finish_reason": ("SOFT", True),
    "protocol_stream_shape": ("HARD", True),
    "protocol_tool_call_shape": ("SOFT", True),
    "protocol_error_shape": ("INFO", False),
    # 能力探测 — INFO 占位
    "vision_probe": ("INFO", False),
    "file_probe": ("INFO", False),
    "audio_probe": ("INFO", False),
    "embedding_probe": ("INFO", False),
    # MIXED — 聚合入口（向后兼容）
    "political_sensitivity": ("MIXED", True),
}
