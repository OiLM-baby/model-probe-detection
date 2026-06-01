#!/usr/bin/env python3
"""TokenStar 命令行入口 — Composition Root。

装配 EventBus / Store / Notifier / Renderer / Subscribers，编排测试流程。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from app.core.config_loader import get_known_tests, load_app_config
from app.core.runner import run_provider
from app.events.bus import EventBus
from app.events.types import (
    PROVIDER_CRASHED,
    PROVIDER_DONE,
    PROVIDER_STARTED,
    REPORT_COMPLETED,
    REPORT_CRASHED,
    REPORT_STARTED,
    ProgressEvent,
)
from app.report.debouncer import DebouncedRenderer
from app.report.generator import (
    build_email_summary,
    build_payload,
    build_totals,
    build_wechat_summary,
)
from app.report.renderer import HtmlRenderer
from app.services.composite import CompositeNotifier
from app.services.email_notify import EmailNotifier
from app.services.wechat_notify import WechatNotifier
from app.storage.sqlite_store import SqliteReportStore, suite_type_for, template_name_for
from app.subscribers.alert_subscriber import AlertSubscriber
from app.subscribers.logger_subscriber import LoggerSubscriber
from app.subscribers.render_subscriber import RenderSubscriber
from app.subscribers.store_subscriber import StoreSubscriber
from app.utils import logger as logger_module
from app.utils.baseline import build_baseline_comparisons, load_baselines
from app.utils.tags import auto_tags
from app.utils.timezone import beijing_from_timestamp


def _run_parallel_providers(providers, args, app_config, suite_tests, bus, logger):
    """并发执行 provider 列表，并保持最终 summaries 顺序与配置顺序一致。

    事件发布按实际完成顺序发生，便于页面实时刷新；函数返回值按原 provider
    顺序排列，便于报告稳定展示。
    """
    ordered = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_provider, provider, app_config.thresholds, suite_tests,
                pricing_config=app_config.pricing,
                judge_config=app_config.judge,
                quota_config=app_config.quota,
                bus=bus,
            ): provider for provider in providers
        }
        completed = as_completed(future_map)
        try:
            from tqdm import tqdm
            completed = tqdm(completed, total=len(future_map), desc="执行测试")
        except ImportError:
            pass
        for future in completed:
            provider = future_map[future]
            try:
                summary = future.result()
            except Exception:
                logger.exception("provider 执行崩溃: %s", provider.name)
                summary = None
            ordered[provider.name] = summary
            if summary is not None:
                _publish_provider_events(bus, provider, summary, stage="done")
            else:
                _publish_provider_events(bus, provider, None, stage="crashed", error="worker future returned None")
    return [summary for provider in providers if (summary := ordered.get(provider.name)) is not None]


def parse_args(argv):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="TokenStar 模型中转可用性测试")
    parser.add_argument("--config", default="config/providers.yaml", help="中转配置 YAML")
    parser.add_argument("--env", choices=["test", "prod"], help="运行环境，默认读取配置 app.default_env")
    parser.add_argument("--suite", default=None, help="测试功能套件，默认 availability")
    parser.add_argument("--run-id", default=None, help="本次运行 ID，默认自动生成")
    parser.add_argument("--provider", action="append", help="只运行指定中转名，可传多次")
    parser.add_argument("--send-wechat", action="store_true", help="发送企业微信汇总")
    parser.add_argument("--send-email", action="store_true", help="发送邮箱汇总")
    parser.add_argument("--no-attachment", action="store_true", help="企业微信只发摘要，不上传报告附件")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数（默认 1 串行）")
    parser.add_argument("--resume", action="store_true", help="从上次报告中恢复：跳过已成功的中转，只测试未完成的")
    parser.add_argument("--list-tests", action="store_true", help="列出所有可用测试项并退出")
    parser.add_argument("--list-providers", action="store_true", help="列出所有已启用中转名和模型并退出")
    parser.add_argument("--list-reports", action="store_true", help="列出历史报告记录并退出")
    parser.add_argument("--list-reports-limit", type=int, default=30, help="--list-reports 显示条数（默认 30）")
    parser.add_argument("--probe-models", action="store_true", help="调用 /v1/models 接口拉取可用模型列表")
    parser.add_argument("--init-vault", action="store_true", help="生成加密密钥并初始化 vault 数据库")
    parser.add_argument("--serve", action="store_true", help="启动 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8080, help="API 监听端口（默认 8080）")
    return parser.parse_args(argv)


def main(argv=None):
    """CLI / API 服务入口。

    普通运行模式会加载配置、组装事件总线/存储/通知/渲染器，执行 suite，
    最后生成 HTML、写历史库并发送通知；--serve 模式只启动 FastAPI 服务。
    """
    args = parse_args(argv or sys.argv[1:])

    if args.serve:
        import uvicorn

        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.db")
        from app.api.deps import DEFAULT_DB_PATH

        if os.path.exists(DEFAULT_DB_PATH):
            db_path = DEFAULT_DB_PATH
        store = SqliteReportStore(db_path)
        store.init()
        from app.api import create_app

        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        app_config = load_app_config(config_path, args.env, args.suite)
        app = create_app(db_path, app_config=app_config)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.init_vault:
        from app.services.crypto_vault import VAULT_DB, generate_key, init_vault
        key = generate_key()
        init_vault()
        key_file = os.path.join(os.path.expanduser("~"), ".config", "tokenstar", "master.key")
        os.makedirs(os.path.dirname(key_file), mode=0o700, exist_ok=True)
        with open(key_file, "w", encoding="ascii") as fh:
            fh.write(key)
        os.chmod(key_file, 0o600)
        print(f"Vault 已初始化: {VAULT_DB}")
        print(f"密钥已写入: {key_file}  (权限 0600)")
        print("请将以下环境变量加入你的 shell 配置文件：")
        print(f"  export TOKENSTAR_SECRET_KEY={key}")
        return

    if not os.path.isabs(args.config):
        args.config = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    app_config = load_app_config(args.config, args.env, args.suite, args.run_id)

    suite_tests = app_config.suites.get(app_config.runtime.suite)
    if not suite_tests:
        raise SystemExit(f"未知或空测试套件: {app_config.runtime.suite}")

    providers = app_config.providers
    if args.provider:
        wanted = set(args.provider)
        providers = [item for item in providers if item.name in wanted]

    if args.list_tests:
        print(f"套件 {app_config.runtime.suite} 的测试项：")
        for test_name in suite_tests:
            print(f"  - {test_name}")
        return

    if args.list_providers:
        print("已启用的中转：")
        for p in providers:
            print(f"  - {p.name}  model={p.model}  format={p.format}  base_url={p.base_url}")
        return

    if args.list_reports:
        _list_reports(app_config, args)
        return

    if args.probe_models:
        from app.core.config_loader import probe_models
        for p in providers:
            print(f"\n{p.name} ({p.base_url}):")
            models = probe_models(p.base_url, p.api_key)
            if models:
                for m in models:
                    print(f"  - {m}")
                print(f"\n  → 共 {len(models)} 个模型，可直接复制到 providers.local.yaml:")
                print("  models:")
                for m in models:
                    print(f"    - {m}")
            else:
                print("  (未获取到模型列表，请手动指定 model 或 models)")
        return

    import signal

    _validate_startup(app_config, args)
    logger = logger_module.setup_logger(
        env=app_config.runtime.env,
        run_id=app_config.runtime.run_id,
        log_dir=app_config.logging.directory,
        level=app_config.logging.level,
        max_bytes=app_config.logging.max_bytes,
        backup_count=app_config.logging.backup_count,
    )

    if not providers:
        raise SystemExit("没有可执行的中转配置，请检查 providers 配置或 --provider 参数")

    # ── 断点续跑 ───────────────────────────────────────────
    if args.resume:
        providers = _resume_providers(app_config, providers, args)

    # ── 组装模块（Composition Root）────────────────────────
    # 主入口只负责把各模块接起来：事件总线负责解耦执行、落库、渲染和告警。
    history_config = app_config.history
    db_path = (history_config.db_path
               or os.path.join(os.path.dirname(app_config.report.directory), "data", "history.db"))

    bus = EventBus()
    store = SqliteReportStore(db_path)
    store.init()

    # 通知渠道
    notifier = CompositeNotifier()
    _wechat = None
    _email = None
    if args.send_wechat or app_config.wechat.enabled:
        _wechat = WechatNotifier(app_config.wechat.webhook_url)
        if _wechat.webhook_url:
            notifier.add(_wechat)
        else:
            logger.warning("企业微信 webhook_url 未配置，已跳过")
    if args.send_email or app_config.email.enabled:
        email_cfg = app_config.email
        if email_cfg.smtp_server and email_cfg.username and email_cfg.password and email_cfg.receivers:
            _email = EmailNotifier(email_cfg)
            notifier.add(_email)
        else:
            logger.warning("邮箱 SMTP 配置不完整，已跳过")

    # 订阅者
    store_sub = StoreSubscriber(store)
    store_sub.attach(bus)
    LoggerSubscriber().attach(bus)
    AlertSubscriber(store).attach(bus)

    # 渲染器
    html_renderer = HtmlRenderer()
    debounced_html = DebouncedRenderer(html_renderer, debounce_secs=10.0)
    RenderSubscriber(debounced_html, report_dir=app_config.report.directory).attach(bus)

    # ── 崩溃兜底 ───────────────────────────────────────────
    # 进程被信号中断时尽量落一份 partial payload，方便 UI 和历史库知道这次跑挂了。
    start_time = time.time()
    summaries: list = []

    def _crash_handler(reason: str):
        try:
            partial_payload = build_payload(
                summaries, start_time, time.time(),
                env=app_config.runtime.env,
                run_id=app_config.runtime.run_id,
                suite=app_config.runtime.suite,
            )
            payload_json = json.dumps(partial_payload, ensure_ascii=False)
        except Exception:
            payload_json = ""
        bus.publish(ProgressEvent(
            kind=REPORT_CRASHED,
            payload={"run_id": app_config.runtime.run_id, "reason": reason, "payload_json": payload_json},
        ))

    def _signal_handler(signum, frame):
        _crash_handler(f"signal {signum}")
        sys.exit(130)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── 文件锁（防多实例并发写 SQLite）─────────────────────
    # SQLite 对多进程写比较敏感；这里用旁路 lock 文件避免 cron/API 重叠运行。
    _lock_fd = None
    _lock_path = f"{db_path}.lock"
    try:
        _lock_fd = open(_lock_path, "w")
        fcntl = __import__("fcntl")
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as _lock_err:
        if _lock_fd:
            _lock_fd.close()
        raise SystemExit("另一个 TokenStar 实例正在运行，请稍后重试。") from _lock_err

    # ── 启动报告 ───────────────────────────────────────────
    stype = suite_type_for(app_config.runtime.suite)
    tpl = template_name_for(stype)
    cleanup_count = store.cleanup_zombie_reports()
    if cleanup_count:
        logger.info("已清理 %d 个僵尸 report", cleanup_count)

    bus.publish(ProgressEvent(
        kind=REPORT_STARTED,
        payload={
            "env": app_config.runtime.env,
            "suite": app_config.runtime.suite,
            "run_id": app_config.runtime.run_id,
            "planned": len(providers),
            "suite_type": stype,
            "template_name": tpl,
        },
    ))

    logger.info(
        "开始执行 TokenStar: env=%s suite=%s providers=%s",
        app_config.runtime.env, app_config.runtime.suite, len(providers),
    )
    start_time = time.time()

    # ── 执行测试 ───────────────────────────────────────────
    if args.workers > 1:
        # 提交前先发 PROVIDER_STARTED
        for provider in providers:
            _publish_provider_events(bus, provider, None, stage="started")

        summaries.extend(_run_parallel_providers(providers, args, app_config, suite_tests, bus, logger))
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(providers, desc="执行测试")
        except ImportError:
            iterator = providers
        for provider in iterator:
            _publish_provider_events(bus, provider, None, stage="started")
            try:
                summary = run_provider(
                    provider, app_config.thresholds, suite_tests,
                    pricing_config=app_config.pricing,
                    judge_config=app_config.judge,
                    quota_config=app_config.quota,
                    bus=bus,
                )
                _publish_provider_events(bus, provider, summary, stage="done")
                summaries.append(summary)
            except Exception as exc:
                logger.exception("provider 执行崩溃: %s", provider.name)
                _publish_provider_events(bus, provider, None, stage="crashed", error=str(exc))

    end_time = time.time()

    # ── 历史对比 ───────────────────────────────────────────
    # 历史对比使用同 env + suite 的上一份报告，展示本次与上次的变化。
    comparison = None
    if history_config.enabled and summaries:
        previous = store.load_previous_run(app_config.runtime.env, app_config.runtime.suite)
        if previous:
            current_payload = build_payload(
                summaries, start_time, end_time,
                env=app_config.runtime.env,
                run_id=app_config.runtime.run_id,
                suite=app_config.runtime.suite,
            )
            comparison = store.build_comparison(current_payload, previous)
            logger.info("历史对比已生成: 上次 run_id=%s", previous.get("run_id", "-"))
        else:
            logger.info("未找到历史记录，跳过对比")

    # ── 基线对比 ───────────────────────────────────────────
    # 基线对比使用 config/baselines.yaml 中的阈值，偏差会写入 alerts 表。
    baseline_comparisons = None
    historical_baseline = None
    baseline_config = None
    baselines_config = os.environ.get("TOKENSTAR_BASELINES_CONFIG") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "baselines.yaml"
    )
    if os.path.exists(baselines_config):
        baseline = load_baselines(baselines_config, app_config.runtime.suite)
        baseline_config = baseline.to_dict(source="config/baselines.yaml")
        baseline_comparisons = build_baseline_comparisons(summaries, baseline)
        historical_baseline = store.get_historical_baseline(
            app_config.runtime.env, app_config.runtime.suite,
        )
        dev_count = sum(1 for bc in baseline_comparisons if bc["has_deviations"])
        if dev_count:
            logger.info("基线对比完成: %d 个中转存在偏差", dev_count)
            for bc in baseline_comparisons:
                if not bc["has_deviations"]:
                    continue
                for d in bc["deviations"]:
                    store.write_alert(
                        report_id=store_sub.report_id,
                        error_kind="baseline_deviation",
                        error_msg=(
                            f"[{d['label']}] 当前={d['current']}{d['unit']} "
                            f"基线={d['baseline']}{d['unit']} "
                            f"方向={d['direction']}"
                        ),
                        test_name=f"baseline:{d['metric']}",
                        provider=bc["provider"],
                        model=bc.get("model", ""),
                    )
            logger.info("基线偏差告警已写入，共 %d 条", sum(len(bc["deviations"]) for bc in baseline_comparisons if bc["has_deviations"]))

    # ── 生成报告 ───────────────────────────────────────────
    # payload 是统一数据源：HTML、SQLite 历史、通知摘要都从它或 summaries 派生。
    report_dir = app_config.report.directory
    archive_dir = os.path.join(report_dir, app_config.runtime.env, app_config.runtime.suite, "archive")
    timestamp = beijing_from_timestamp(end_time, "%Y%m%d_%H%M%S")
    report_id = app_config.runtime.run_id or timestamp

    payload = build_payload(
        summaries, start_time, end_time,
        env=app_config.runtime.env,
        run_id=app_config.runtime.run_id,
        suite=app_config.runtime.suite,
        comparison=comparison,
        baseline_comparisons=baseline_comparisons,
        historical_baseline=historical_baseline,
        baseline_config=baseline_config,
    )

    html_path = os.path.join(archive_dir, f"tokenstar_report_{report_id}.html")

    html_renderer.render(payload, html_path)

    payload_json = json.dumps(payload, ensure_ascii=False)
    json_path = os.path.join(archive_dir, f"tokenstar_report_{report_id}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(payload_json)

    # ── 收尾 ───────────────────────────────────────────────
    bus.publish(ProgressEvent(
        kind=REPORT_COMPLETED,
        payload={"run_id": app_config.runtime.run_id, "payload_json": payload_json},
    ))

    if history_config.enabled:
        pruned = store.prune_history(
            keep=history_config.keep,
            older_than_days=history_config.older_than_days,
        )
        if pruned:
            logger.info("历史记录已清理 %d 条旧数据", pruned)

    totals = build_totals(summaries)
    print(f"报告已生成: {html_path}")
    print(
        f"中转数={totals['provider_count']} 测试项={totals['total']} "
        f"通过={totals['passed']} 失败={totals['failed']} 警告={totals['warned']}"
    )
    if totals["hard_pass_rate"] is not None:
        print(f"硬指标通过率={totals['hard_pass_rate']}% ({totals['hard_passed']}/{totals['hard_total']})")
    if totals["soft_health"] is not None:
        print(f"软指标健康度={totals['soft_health']}分")
    if totals["critical_failed"] is not None:
        print(f"安全底线失败={totals['critical_failed']}")
    if totals["info_count"] is not None:
        print(f"信息采集项={totals['info_count']}")
    print(f"整体状态: {totals['overall']}")

    # ── 通知 ───────────────────────────────────────────────
    if args.send_wechat or app_config.wechat.enabled:
        content = build_wechat_summary(
            summaries, start_time, end_time,
            env=app_config.runtime.env,
            run_id=app_config.runtime.run_id,
            suite=app_config.runtime.suite,
        )
        attachment = None if args.no_attachment else html_path
        if _wechat and _wechat.send_summary(content, attachment):
            logger.info("企业微信汇总已发送")
        elif _wechat:
            logger.warning("企业微信汇总发送失败，请检查 webhook 配置")

    if args.send_email or app_config.email.enabled:
        content = build_email_summary(
            summaries, start_time, end_time,
            env=app_config.runtime.env,
            run_id=app_config.runtime.run_id,
            suite=app_config.runtime.suite,
        )
        subject = f"TokenStar {app_config.runtime.env}/{app_config.runtime.suite} 测试汇总 {app_config.runtime.run_id}"
        if _email and _email.send_summary(content, None if args.no_attachment else html_path, subject=subject):
            logger.info("邮箱汇总已发送")
        elif _email:
            logger.warning("邮箱汇总发送失败，请检查 SMTP 配置")

    # 释放文件锁
    if _lock_fd is not None:
        fcntl = __import__("fcntl")
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
        _lock_fd.close()


def _publish_provider_events(bus, provider, summary=None, stage="started", error=""):
    """发布 provider 级事件。"""
    tags_json = json.dumps(
        auto_tags(provider.name, provider.model, provider.group),
        ensure_ascii=False,
    )

    if stage == "started":
        bus.publish(ProgressEvent(
            kind=PROVIDER_STARTED,
            payload={
                "provider": provider.name,
                "model": provider.model,
                "group_name": provider.group,
                "provider_format": provider.format,
                "base_url": provider.base_url,
                "tags_json": tags_json,
            },
        ))
    elif stage == "done" and summary is not None:
        bus.publish(ProgressEvent(
            kind=PROVIDER_DONE,
            payload={"summary": asdict(summary), "results": [asdict(r) for r in summary.results]},
        ))
    elif stage == "crashed":
        bus.publish(ProgressEvent(
            kind=PROVIDER_CRASHED,
            payload={
                "provider": provider.name,
                "model": provider.model,
                "group_name": provider.group,
                "error": error,
            },
        ))


def _resume_providers(app_config, providers, args):
    """断点续跑：从上次报告中获取已完成的中转，跳过它们。"""
    history_config = app_config.history
    db_path = (history_config.db_path
               or os.path.join(os.path.dirname(app_config.report.directory), "data", "history.db"))
    store = SqliteReportStore(db_path)
    store.init()
    previous = store.load_previous_run(app_config.runtime.env, app_config.runtime.suite)
    if not previous:
        print("未找到上次报告，无法恢复。将运行全部中转。")
        return providers
    completed_names: set[str] = set()
    for p in previous.get("providers") or []:
        name = p.get("provider", "")
        if name and p.get("status") == "completed":
            completed_names.add(name)
    if not completed_names:
        print("上次报告中没有已成功的中转。将运行全部中转。")
        return providers
    remaining = [p for p in providers if p.name not in completed_names]
    skipped = len(providers) - len(remaining)
    print(f"断点续跑: 从 run_id={previous.get('run_id', '-')} 恢复")
    print(f"  已跳过 {skipped} 个已完成中转: {', '.join(sorted(completed_names))}")
    print(f"  剩余 {len(remaining)} 个待测试中转")
    if not remaining:
        print("所有中转均已完成，无需再跑。")
        raise SystemExit(0)
    return remaining


def _list_reports(app_config, args):
    """列出历史报告记录。"""
    history_config = app_config.history
    db_path = (history_config.db_path
               or os.path.join(os.path.dirname(app_config.report.directory), "data", "history.db"))
    store = SqliteReportStore(db_path)
    store.init()
    env_filter = app_config.runtime.env or ""
    suite_type = None
    if app_config.runtime.suite:
        from app.storage.sqlite_store import suite_type_for
        suite_type = suite_type_for(app_config.runtime.suite)
    reports = store.list_recent_reports(env=env_filter, suite_type=suite_type, limit=args.list_reports_limit)
    if not reports:
        print("暂无历史报告记录。")
        return
    print(f"{'ID':<6} {'Run ID':<22} {'环境':<8} {'套件':<18} {'类型':<12} {'状态':<12} {'通过率':<8} {'创建时间'}")
    print("-" * 110)
    for r in reports:
        run_id = (r.get("run_id") or "")[:20]
        env = (r.get("env") or "")[:6]
        suite = (r.get("suite") or "")[:16]
        stype = (r.get("suite_type") or "")[:10]
        status = (r.get("status") or "")[:10]
        total = r.get("total_tests") or 0
        passed = r.get("passed") or 0
        rate = f"{passed}/{total}" if total else "-"
        created = (r.get("created_at") or "")[:19]
        print(f"{r['id']:<6} {run_id:<22} {env:<8} {suite:<18} {stype:<12} {status:<12} {rate:<8} {created}")


def _validate_startup(app_config, args):
    if args.workers < 1:
        raise SystemExit("--workers 必须大于等于 1")

    suite_tests = app_config.suites.get(app_config.runtime.suite)
    if not suite_tests:
        raise SystemExit(f"未知或空测试套件: {app_config.runtime.suite}")

    known_tests = set(get_known_tests())
    unknown_tests = sorted({test for tests in app_config.suites.values() for test in tests if test not in known_tests})
    if unknown_tests:
        raise SystemExit(f"配置包含未知测试项: {', '.join(unknown_tests)}")

    valid_formats = {"openai", "anthropic", "responses"}
    for provider in app_config.providers:
        if provider.format not in valid_formats:
            raise SystemExit(f"provider {provider.name} format 不支持: {provider.format}")
        if not provider.api_key or "$" in provider.api_key:
            raise SystemExit(f"provider {provider.name} api_key 未配置或环境变量未展开")
        if not provider.model:
            raise SystemExit(f"provider {provider.name} model 未配置")


if __name__ == "__main__":
    main()
