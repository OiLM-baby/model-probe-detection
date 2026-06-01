"""核心数据模型。

定义中转配置、单条测试结果、单个中转汇总结果的数据结构，
供 runner、测试用例和报告生成模块共同使用。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeConfig:
    """一次运行的基础上下文。

    env 决定使用哪个环境的网关地址，suite 决定跑哪组测试项，
    run_id 用于关联日志、报告、历史记录和通知。
    """

    env: str
    run_id: str
    suite: str


@dataclass
class LoggingConfig:
    """日志输出配置。"""

    level: str = "INFO"
    directory: str = "logs"
    max_bytes: int = 100 * 1024 * 1024
    backup_count: int = 5


@dataclass
class ReportConfig:
    """HTML 报告输出目录配置。"""

    directory: str = "reports"


@dataclass
class WechatConfig:
    """企业微信通知配置。"""

    enabled: bool = False
    webhook_url: str = ""


@dataclass
class EmailConfig:
    """邮件通知配置。"""

    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 465
    username: str = ""
    password: str = ""
    sender: str = ""
    receivers: list[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """一个待测试的中转/模型组合。

    provider 配置可以在 YAML 中写一个中转多个 models；加载后会展开成多个
    ProviderConfig，每个对象只对应一个 model，方便 runner 并发执行和报告统计。
    """

    name: str
    format: str
    base_url: str
    api_key: str
    model: str
    enabled: bool = True
    timeout: int = 180
    tests: list[str] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)
    group: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)  # vision/files/audio/embeddings 口子
    tags: list[str] = field(default_factory=list)  # 自动/手动打标


@dataclass
class HistoryConfig:
    """历史报告保存与清理配置。"""

    enabled: bool = False
    keep: int = 30
    older_than_days: int | None = None
    db_path: str = ""


@dataclass
class AppConfig:
    """应用启动后的完整配置对象。

    config_loader 会把 YAML、环境变量、vault、默认值统一整理成这个对象，
    后续入口、runner、报告、通知都只依赖 AppConfig。
    """

    runtime: RuntimeConfig
    logging: LoggingConfig
    report: ReportConfig
    wechat: WechatConfig
    email: EmailConfig
    thresholds: dict[str, Any]
    suites: dict[str, list[str]]
    providers: list[ProviderConfig]
    pricing: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    quota: dict[str, Any] = field(default_factory=dict)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """单条测试项结果。

    一次 provider/model 运行会产生多条 TestResult，例如 daily 套件通常产生
    connectivity 和 latency 两条。报告里的“测试项总数”就是这些结果条数的总和。
    """

    __test__ = False

    provider: str
    model: str
    test_name: str
    status: str
    latency_ms: int | None = None
    score: float | None = None
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    evaluation_type: str = "SOFT"
    counts_toward_pass_rate: bool = True


@dataclass
class ProviderSummary:
    """单个 provider/model 的汇总结果。

    runner 把同一个 provider/model 的多条 TestResult 聚合成 ProviderSummary。
    报告总览再把所有 ProviderSummary 相加，得到中转数、通过数、失败数等指标。
    """

    provider: str
    model: str
    total: int
    passed: int
    failed: int
    warned: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    results: list[TestResult]
    group: str = ""
    # 评价类型统计
    scorable_total: int = 0
    scorable_passed: int = 0
    scorable_failed: int = 0
    scorable_warned: int = 0
    hard_total: int = 0
    hard_passed: int = 0
    soft_avg_score: float | None = None
    critical_failed: int = 0
    info_count: int = 0
    # 成本/用量
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost: float | None = None
    cost_currency: str = "CNY"
    price_matched: bool = False
    # quota（provider 级展示，不汇总）
    quota_balance: float | None = None
    quota_currency: str = ""
    estimated_remaining: float | None = None
