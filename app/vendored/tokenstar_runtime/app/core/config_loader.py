"""统一配置加载模块。

读取 YAML 后统一整理成 AppConfig，支持 test/prod 环境、默认 provider 配置、
环境级 base_url，支持 base_config 继承，以及旧版 providers.yaml 的兼容加载。
"""

import functools
import os
import re
import time
from copy import deepcopy
from typing import Any

import yaml

from app.core.models import (
    AppConfig,
    EmailConfig,
    HistoryConfig,
    LoggingConfig,
    ProviderConfig,
    ReportConfig,
    RuntimeConfig,
    WechatConfig,
)

DEFAULT_ENV = "prod"
DEFAULT_SUITE = "availability"


@functools.cache
def get_known_tests() -> tuple[str, ...]:
    """返回所有已知测试项名称（从 TEST_REGISTRY 派生，线程安全）。"""
    from app.tests.cases import TEST_REGISTRY  # 延迟导入，避免循环依赖
    return tuple(sorted(TEST_REGISTRY.keys()))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """深度合并 overlay 到 base，列表直接替换。"""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_path(path: str) -> str:
    """统一解析配置路径：展开 ~ 和环境变量，转为绝对路径。"""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def load_config(path: str, _stack: tuple | None = None) -> dict[str, Any]:
    """读取 YAML 配置并处理 base_config 继承。

    base_config 可以让生产/本地配置共享同一份基础配置；这里会检测循环引用，
    并用当前文件覆盖 base 文件中的同名字段。
    """
    path = _resolve_path(path)
    _stack = (_stack or ()) + (path,)
    if path in _stack[:-1]:
        raise ValueError(f"base_config 循环引用: {' -> '.join(_stack)}")
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取配置 {path}: {exc}") from exc
    base_path = data.pop("base_config", None)
    if base_path:
        base_path = os.path.expanduser(os.path.expandvars(base_path))
        if not os.path.isabs(base_path):
            config_relative = os.path.join(os.path.dirname(path), base_path)
            project_relative = os.path.join(os.path.dirname(os.path.dirname(path)), base_path)
            base_path = config_relative if os.path.exists(config_relative) else project_relative
        data = _deep_merge(load_config(base_path, _stack), data)
    data.setdefault("providers", [])
    data.setdefault("thresholds", {})
    data.setdefault("wechat", {})
    return data


def load_app_config(
    path: str,
    env: str | None = None,
    suite: str | None = None,
    run_id: str | None = None,
) -> AppConfig:
    """加载并规范化完整应用配置。

    入口层只传 config/env/suite/run_id；本函数负责按优先级合并：
    CLI 参数 > 环境变量 > YAML 默认值，并加载 vault、通知、历史、provider 等配置。
    """
    raw = load_config(path)
    config_abs = _resolve_path(path)
    project_root = os.path.dirname(os.path.dirname(config_abs))
    _load_env_file(raw.get("env_file"), project_root)
    _load_vault_secret_key(raw)
    selected_env = _resolve_env(raw, env)
    selected_suite = _resolve_suite(raw, suite)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    suites = _load_suites(raw)
    vault_result = _load_from_vault(project_root, raw)
    if vault_result:
        vault_providers, vault_wechat, vault_email = vault_result
    else:
        vault_providers, vault_wechat, vault_email = None, None, None

    return AppConfig(
        runtime=RuntimeConfig(env=selected_env, run_id=run_id, suite=selected_suite),
        logging=_load_logging(raw, project_root),
        report=_load_report(raw, project_root),
        wechat=vault_wechat if (vault_wechat is not None and vault_wechat.webhook_url) else _load_wechat(raw),
        email=vault_email if (vault_email is not None and vault_email.password) else _load_email(raw),
        thresholds=raw.get("thresholds", {}),
        suites=suites,
        providers=vault_providers if vault_providers is not None else load_providers(raw, selected_env),
        pricing=raw.get("pricing") or {},
        judge=_load_judge(raw, selected_env),
        quota=raw.get("quota") or {},
        history=_load_history(raw, project_root),
        raw=raw,
    )


def _expand_env(value: Any) -> Any:
    """展开配置中的环境变量引用。

    如果值形如 $TOKEN 且环境变量不存在，返回空字符串，避免把字面量密钥
    当成真实配置继续使用。
    """
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if expanded == value and re.fullmatch(r"\$(\w+)|\$\{(\w+)\}", value.strip()):
            return ""
        return expanded
    return value


def _load_vault_secret_key(config: dict[str, Any]) -> None:
    """把 YAML 中的 vault.secret_key 放入环境变量，供 vault 解密使用。"""
    if os.environ.get("TOKENSTAR_SECRET_KEY"):
        return
    secret_key = _expand_env((config.get("vault") or {}).get("secret_key", ""))
    if secret_key:
        os.environ["TOKENSTAR_SECRET_KEY"] = str(secret_key)


def load_providers(config: dict[str, Any], env: str | None = None) -> list[ProviderConfig]:
    """把 YAML provider 配置展开成可执行的 ProviderConfig 列表。

    一个 provider 可以配置多个 models；这里会展开成多个 provider/model 组合。
    若 models: auto，会调用 /v1/models 探测可用模型。
    """
    env = env or _resolve_env(config, None)
    defaults = config.get("defaults", {})
    env_config = (config.get("environments", {}) or {}).get(env, {})
    env_base_url = _expand_env(env_config.get("base_url", ""))
    providers = []
    for item in config.get("providers", []):
        if not item.get("enabled", True):
            continue
        base_url = _expand_env(item.get("base_url") or env_base_url)
        if not base_url:
            raise ValueError(f"provider {item.get('name')} 缺少 base_url，请在 provider 或 environments.{env}.base_url 配置")
        models = _provider_models(item)
        if models is None:
            models = probe_models(base_url, _expand_env(item.get("api_key", "")))
            if not models:
                raise ValueError(f"provider {item.get('name', '?')}: models: auto 探针 /v1/models 失败，请手动指定模型")
        if not models:
            raise ValueError(f"provider {item.get('name', '?')} 缺少 model 或 models 配置，设置 models: auto 可自动发现模型")
        for model in models:
            provider_name = item["name"] if len(models) == 1 else f"{item['name']}__{model}"
            providers.append(
                ProviderConfig(
                    name=provider_name,
                    format=str(item.get("format", defaults.get("format", "openai"))).lower(),
                    base_url=base_url.rstrip("/"),
                    api_key=str(_expand_env(item.get("api_key", "")) or ""),
                    model=model,
                    enabled=item.get("enabled", True),
                    timeout=int(item.get("timeout", defaults.get("timeout", 180))),
                    tests=item.get("tests") or defaults.get("tests") or get_known_tests(),
                    extra_headers=item.get("extra_headers") or defaults.get("extra_headers") or {},
                    group=item["name"],
                    capabilities=item.get("capabilities") or defaults.get("capabilities") or {},
                    tags=list(item.get("tags") or defaults.get("tags") or []),
                )
            )
    return providers


def _resolve_env(config: dict[str, Any], requested_env: str | None) -> str:
    """解析最终运行环境。优先级：CLI 参数 > TOKENSTAR_ENV > YAML 默认值。"""
    if requested_env:
        return requested_env
    return os.environ.get("TOKENSTAR_ENV") or config.get("app", {}).get("default_env") or config.get("env") or DEFAULT_ENV


def _resolve_suite(config: dict[str, Any], requested_suite: str | None) -> str:
    """解析最终测试套件。优先级：CLI 参数 > TOKENSTAR_SUITE > YAML 默认值。"""
    if requested_suite:
        return requested_suite
    return os.environ.get("TOKENSTAR_SUITE") or config.get("app", {}).get("default_suite") or DEFAULT_SUITE


def _load_suites(config: dict[str, Any]) -> dict[str, list[str]]:
    """读取 suite 映射；缺省 availability 时用 defaults.tests 兜底。"""
    suites = config.get("suites") or {}
    if DEFAULT_SUITE not in suites:
        suites[DEFAULT_SUITE] = config.get("defaults", {}).get("tests") or get_known_tests()
    return {name: list(tests or []) for name, tests in suites.items()}


def _load_logging(config: dict[str, Any], project_root: str) -> LoggingConfig:
    item = config.get("logging", {})
    directory = str(item.get("directory", "logs"))
    if not os.path.isabs(directory):
        directory = os.path.join(project_root, directory)
    return LoggingConfig(
        level=str(item.get("level", "INFO")),
        directory=directory,
        max_bytes=int(item.get("max_bytes", 100 * 1024 * 1024)),
        backup_count=int(item.get("backup_count", 5)),
    )


def _load_report(config: dict[str, Any], project_root: str) -> ReportConfig:
    item = config.get("report", {})
    directory = str(item.get("directory", "reports"))
    if not os.path.isabs(directory):
        directory = os.path.join(project_root, directory)
    return ReportConfig(directory=directory)


def _load_wechat(config: dict[str, Any]) -> WechatConfig:
    item = config.get("wechat", {})
    return WechatConfig(
        enabled=bool(item.get("enabled", False)),
        webhook_url=str(_expand_env(item.get("webhook_url", "")) or ""),
    )


def _load_email(config: dict[str, Any]) -> EmailConfig:
    item = config.get("email", {})
    receivers_raw = item.get("receivers") or []
    if isinstance(receivers_raw, str):
        receivers_raw = _expand_env(receivers_raw)
        receivers = [part.strip() for part in receivers_raw.split(",") if part.strip()]
    else:
        receivers = [str(_expand_env(part)).strip() for part in receivers_raw if str(part).strip()]
    username = str(_expand_env(item.get("username", "")) or "")
    sender = str(_expand_env(item.get("sender", "")) or "") or username
    return EmailConfig(
        enabled=bool(item.get("enabled", False)),
        smtp_server=str(_expand_env(item.get("smtp_server", "")) or ""),
        smtp_port=int(item.get("smtp_port", 465)),
        username=username,
        password=str(_expand_env(item.get("password", "")) or ""),
        sender=sender,
        receivers=list(receivers),
    )


def _provider_models(item: dict[str, Any]) -> list[str] | None:
    """返回模型列表，或 None 表示需要自动探针，或 [] 表示未配置。"""
    models_val = item.get("models")
    if models_val == "auto":
        return None  # 触发 probe_models()
    if models_val:
        return list(models_val)
    if item.get("model"):
        return [item["model"]]
    return []  # 未配置 model/models，报错


def probe_models(base_url: str, api_key: str, timeout: int = 10) -> list[str]:
    """调用 /v1/models 接口拉取可用模型列表。"""
    import logging

    logger = logging.getLogger("tokenstar")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    url = f"{base}/v1/models"
    try:
        import requests
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        if resp.ok:
            data = resp.json()
            models = [m["id"] for m in data.get("data", []) if m.get("id")]
            logger.info("自动探针 %s: 发现 %d 个模型", base_url, len(models))
            return models
        logger.warning("自动探针 %s 返回 HTTP %s", base_url, resp.status_code)
        return []
    except Exception as exc:
        logger.warning("自动探针 %s 失败: %s", base_url, exc)
        return []


def _load_judge(raw: dict[str, Any], env: str) -> dict[str, Any]:
    """加载裁判配置并构建 judge provider。

    支持两种方式指定裁判模型：
    1. 独立配置：judge.base_url + judge.api_key（优先）
    2. 引用已有 provider：judge.provider_name → 查找 providers 列表
    """
    judge = raw.get("judge") or {}
    if not judge.get("enabled"):
        return judge

    model = judge.get("model", "")
    if not model:
        return judge

    defaults = raw.get("defaults", {})
    env_config = (raw.get("environments", {}) or {}).get(env, {})
    env_base_url = _expand_env(env_config.get("base_url", ""))

    # 方式1：独立配置
    direct_url = _expand_env(judge.get("base_url", ""))
    direct_key = _expand_env(judge.get("api_key", ""))
    if direct_url and direct_key:
        judge["_provider"] = ProviderConfig(
            name="judge-standalone",
            format=str(judge.get("format", "openai")).lower(),
            base_url=direct_url.rstrip("/"),
            api_key=str(direct_key),
            model=model,
            timeout=int(judge.get("timeout", defaults.get("timeout", 180))),
        )
        return judge

    # 方式2：引用已有 provider
    provider_name = judge.get("provider_name", "")
    if not provider_name:
        return judge

    providers_raw = raw.get("providers") or []
    provider_data = None
    for p in providers_raw:
        if p.get("name") == provider_name and p.get("enabled", True):
            provider_data = p
            break

    if not provider_data:
        return judge

    base_url = _expand_env(provider_data.get("base_url") or env_base_url)
    judge["_provider"] = ProviderConfig(
        name=f"judge-{provider_name}",
        format=str(judge.get("format", provider_data.get("format", "openai"))).lower(),
        base_url=base_url.rstrip("/") if base_url else "",
        api_key=str(_expand_env(provider_data.get("api_key", "")) or ""),
        model=model,
        timeout=int(judge.get("timeout", provider_data.get("timeout", defaults.get("timeout", 180)))),
        group=provider_name,
    )
    return judge


def _load_from_vault(project_root: str, raw: dict[str, Any]) -> tuple | None:
    """如果 vault 可用，从加密数据库加载中转配置。返回 (providers, wechat, email) 或 None。"""
    import logging
    logger = logging.getLogger("tokenstar")

    vault_db = os.path.join(project_root, "data", "vault.db")
    if not os.path.exists(vault_db):
        logger.info("vault 未启用：%s 不存在，使用 YAML 配置", vault_db)
        return None

    from app.services.crypto_vault import _find_key_file
    from app.services.crypto_vault import load_providers as vault_load

    if not os.environ.get("TOKENSTAR_SECRET_KEY") and not _find_key_file():
        logger.info("vault 未启用：TOKENSTAR_SECRET_KEY 未设置且未找到 .tokenstar_key 文件，使用 YAML 配置")
        return None
    vault_providers, vault_wechat, vault_email = vault_load(vault_db)

    if not vault_providers:
        logger.warning("vault 已启用但未加载到中转配置，不再回退 YAML；请检查 vault.db 或禁用 vault")
        return [], vault_wechat, vault_email

    defaults = raw.get("defaults", {})
    default_tests = list(defaults.get("tests") or get_known_tests())

    providers: list[ProviderConfig] = []
    for vp in vault_providers:
        if not vp.enabled:
            continue
        models = vp.models if vp.models else []
        if not models:
            models = probe_models(vp.base_url, vp.api_key)
            if not models:
                logger.warning("provider %s: models: auto 探针失败，跳过", vp.name)
                continue
        for model in models:
            provider_name = vp.name if len(models) == 1 else f"{vp.name}__{model}"
            providers.append(ProviderConfig(
                name=provider_name,
                format=vp.format,
                base_url=vp.base_url.rstrip("/"),
                api_key=vp.api_key,
                model=model,
                enabled=vp.enabled,
                timeout=vp.timeout,
                tests=default_tests,
                extra_headers=dict(vp.extra_headers),
                group=vp.group or vp.name,
                capabilities=dict(vp.capabilities),
                tags=list(vp.tags),
            ))

    wechat = WechatConfig(
        enabled=vault_wechat.enabled,
        webhook_url=vault_wechat.webhook_url,
    )

    email = EmailConfig(
        enabled=vault_email.enabled,
        smtp_server=vault_email.smtp_server,
        smtp_port=vault_email.smtp_port,
        username=vault_email.username,
        password=vault_email.password,
        sender=vault_email.sender,
        receivers=list(vault_email.receivers) if vault_email.receivers else [],
    )

    logger.info("从 vault 加载了 %d 个中转（展开为 %d 个 provider）", len(vault_providers), len(providers))
    return providers, wechat, email


def _load_history(raw: dict[str, Any], project_root: str) -> HistoryConfig:
    item = raw.get("history") or {}
    db_path = str(item.get("db_path", "") or "")
    if db_path and not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)
    older_than_days = item.get("older_than_days")
    return HistoryConfig(
        enabled=bool(item.get("enabled", False)),
        keep=int(item.get("keep", 30)),
        older_than_days=int(older_than_days) if older_than_days is not None else None,
        db_path=db_path,
    )


def _load_env_file(path: str | None, project_root: str = ""):
    if not path:
        return
    expanded_path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(expanded_path) and project_root:
        expanded_path = os.path.join(project_root, expanded_path)
    if not os.path.exists(expanded_path):
        return
    with open(expanded_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)
