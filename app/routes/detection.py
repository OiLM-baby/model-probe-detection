"""模型检测 — 复用 TokenStar 完整检测套件。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.storage import json_store

router = APIRouter(tags=["detection"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENSTAR_RUNTIME_ROOT = (PROJECT_ROOT / "app" / "vendored" / "tokenstar_runtime").resolve()
TOKENSTAR_RUNNER = TOKENSTAR_RUNTIME_ROOT / "run.py"
TOKENSTAR_CONFIG = PROJECT_ROOT / "config" / "tokenstar" / "providers.yaml"
TOKENSTAR_BASELINES_CONFIG = PROJECT_ROOT / "config" / "tokenstar" / "baselines.yaml"
TOKENSTAR_LOG_DIR = PROJECT_ROOT / "log" / "tokenstar"
TOKENSTAR_DATA_DIR = PROJECT_ROOT / "data" / "tokenstar"
TOKENSTAR_REPORT_DIR = TOKENSTAR_DATA_DIR / "reports"
TOKENSTAR_HISTORY_DB = TOKENSTAR_DATA_DIR / "history" / "history.db"
TOKENSTAR_PYTHON = TOKENSTAR_RUNTIME_ROOT / ".venv" / "bin" / "python"
CACHE_CONFIDENCE_TESTS = ("cache_hit_rate", "cache_hit")

STATUS_TO_UI = {
    "成功": "PASS",
    "失败": "FAIL",
    "警告": "WARN",
    "信息": "INFO",
    "pass": "PASS",
    "fail": "FAIL",
    "warn": "WARN",
    "info": "INFO",
}

EVAL_TYPE_LABEL = {
    "HARD": "硬性",
    "SOFT": "软性",
    "INFO": "信息",
    "CRITICAL": "安全",
    "MIXED": "混合",
}


class DetectionRunRequest(BaseModel):
    config_id: str
    suite: str = "availability"
    models: list[str] = []


class TokenStarRunError(RuntimeError):
    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.payload = payload


@router.get("/api/detection/suites")
def list_suites() -> list[dict[str, Any]]:
    suites = _load_tokenstar_suites()
    return [
        {
            "value": key,
            "label": f"{_suite_label(key)} ({len(items)}项)",
            "tests": items,
        }
        for key, items in suites.items()
    ]


@router.get("/api/detection/tests")
def list_tests() -> dict[str, Any]:
    return {
        "suites": _load_tokenstar_suites(),
        "tests": _load_tokenstar_tests(),
    }


@router.post("/api/detection/run")
def start_detection(body: DetectionRunRequest) -> dict[str, Any]:
    suites = _load_tokenstar_suites()
    if body.suite not in suites:
        raise HTTPException(400, f"未知套件: {body.suite}")
    if not body.models:
        raise HTTPException(400, "至少选择一个模型")

    cfg_data = json_store.get_config(body.config_id)
    if not cfg_data:
        raise HTTPException(404, "Provider 配置不存在")

    cfg = _cfg_obj(cfg_data)
    run = json_store.create_detection_run(body.config_id, body.suite, body.models, suites[body.suite])
    try:
        payload = _run_tokenstar(cfg, body.models, body.suite)
        _persist_tokenstar_payload(run["id"], payload)
    except TokenStarRunError as exc:
        if exc.payload:
            _persist_tokenstar_payload(run["id"], exc.payload, status="error", error=str(exc))
        else:
            json_store.update_detection_run(run["id"], {
                "status": "error",
                "summary": {"error": str(exc), "planned_models": body.models},
                "error": str(exc),
                "finished_at": json_store.now_text(),
            })
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:
        json_store.update_detection_run(run["id"], {
            "status": "error",
            "summary": {"error": str(exc), "planned_models": body.models},
            "error": str(exc),
            "finished_at": json_store.now_text(),
        })
        raise HTTPException(500, str(exc)) from exc

    return {"ok": True, **_format_run_detail(run["id"]), "raw_report": payload}


@router.get("/api/detection/runs")
def list_runs() -> list[dict[str, Any]]:
    suites = _load_tokenstar_suites()
    return [
        {
            "id": row["id"],
            "config_id": row.get("config_id") or "",
            "suite": _suite_label(row.get("suite_key") or ""),
            "suite_key": row.get("suite_key") or "",
            "status": row.get("status") or "",
            "summary": row.get("summary") or {},
            "test_count": len(suites.get(row.get("suite_key") or "", [])),
            "started_at": row.get("started_at") or "",
            "finished_at": row.get("finished_at") or "",
        }
        for row in json_store.list_detection_runs()
    ]


@router.get("/api/detection/runs/{run_id}")
def get_run_detail(run_id: str) -> dict[str, Any]:
    return _format_run_detail(run_id)


@router.get("/api/detection/runs/{run_id}/report")
def download_run_report(run_id: str) -> Response:
    detail = _format_run_detail(run_id)
    content = _build_markdown_from_db(detail)
    filename = f"TokenStar_{_safe_filename(detail.get('suite') or '模型检测')}_报告_{run_id}.md"
    quoted = filename.encode("utf-8").decode("latin-1", errors="ignore")
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{quoted}"; filename*=UTF-8\'\'{_url_quote(filename)}'},
    )


def _format_run_detail(run_id: str) -> dict[str, Any]:
    run = json_store.get_detection_run(run_id)
    if not run:
        raise HTTPException(404, "记录不存在")
    cfg = json_store.get_config(run.get("config_id") or "")
    return {
        "run_id": run["id"],
        "id": run["id"],
        "config_id": run.get("config_id") or "",
        "config": {
            "label": cfg.get("label") if cfg else "",
            "base_url": cfg.get("base_url") if cfg else "",
            "api_key_masked": _mask_key(cfg.get("api_key") or "") if cfg else "",
            "api_format": cfg.get("api_format") if cfg else "",
        },
        "suite": _suite_label(run.get("suite_key") or ""),
        "suite_key": run.get("suite_key") or "",
        "status": run.get("status") or "",
        "summary": run.get("summary") or {},
        "error": run.get("error") or "",
        "raw_payload": run.get("raw_payload"),
        "started_at": run.get("started_at") or "",
        "finished_at": run.get("finished_at") or "",
        "results": run.get("results") or [],
    }


def _url_quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value)


def _load_tokenstar_suites() -> dict[str, list[str]]:
    if not TOKENSTAR_CONFIG.exists():
        return {}
    data = yaml.safe_load(TOKENSTAR_CONFIG.read_text(encoding="utf-8")) or {}
    return {key: list(value or []) for key, value in (data.get("suites") or {}).items()}


def _load_tokenstar_tests() -> list[str]:
    proc = subprocess.run(
        [str(_python_bin()), "-c", "from app.tests.cases import TEST_REGISTRY; print('\\n'.join(sorted(TEST_REGISTRY)))"],
        cwd=str(TOKENSTAR_RUNTIME_ROOT),
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return sorted({item for tests in _load_tokenstar_suites().values() for item in tests})
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _suite_label(key: str) -> str:
    labels = {
        "availability": "可用性测试",
        "daily": "日常巡检",
        "daily_full": "完整日常巡检",
        "first_token_connectivity": "首 Token 连通巡检",
        "connectivity_matrix": "联通性矩阵",
        "audit": "深度审计",
        "cache_audit": "缓存专项",
        "benchmark": "基准测试",
        "model_audit": "轻量模型审计",
        "concurrency_audit": "并发压力",
        "all": "全量测试",
        "all_no_political": "全量测试(不含政治)",
        "protocol_audit": "协议兼容矩阵",
        "capability_probe": "能力探测",
        "political_sensitivity": "政治敏感合规",
    }
    return labels.get(key, key)


def _cfg_obj(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=data.get("id") or "",
        label=data.get("label") or "",
        base_url=data.get("base_url") or "",
        api_key=data.get("api_key") or "",
        api_format=data.get("api_format") or "openai",
    )


def _run_tokenstar(cfg: Any, models: list[str], suite: str) -> dict[str, Any]:
    if not TOKENSTAR_RUNNER.exists():
        raise RuntimeError(f"TokenStar 未找到: {TOKENSTAR_RUNNER}")

    run_id = f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as fh:
        tmp_config = Path(fh.name)
        yaml.safe_dump(
            {
                "base_config": str(TOKENSTAR_CONFIG),
                "app": {"default_env": "prod", "default_suite": suite},
                "runtime": {"run_id": run_id},
                "environments": {"prod": {"base_url": cfg.base_url or ""}},
                "providers": [
                    {
                        "name": cfg.label or "api_test_ui",
                        "enabled": True,
                        "format": cfg.api_format or "openai",
                        "api_key": "$API_TEST_UI_DETECTION_KEY",
                        "models": models,
                        "timeout": 900,
                    }
                ],
                "history": {"enabled": False, "db_path": str(TOKENSTAR_HISTORY_DB)},
                "wechat": {"enabled": False},
                "logging": {"directory": str(TOKENSTAR_LOG_DIR)},
                "report": {"directory": str(TOKENSTAR_REPORT_DIR)},
            },
            fh,
            allow_unicode=True,
            sort_keys=False,
        )

    env = os.environ.copy()
    env["TOKENSTAR_DISABLE_VAULT"] = "1"
    env["TOKENSTAR_BASELINES_CONFIG"] = str(TOKENSTAR_BASELINES_CONFIG)
    env["API_TEST_UI_DETECTION_KEY"] = cfg.api_key or ""
    try:
        proc = subprocess.run(
            [
                str(_python_bin()),
                "run.py",
                "--config",
                str(tmp_config),
                "--suite",
                suite,
                "--run-id",
                run_id,
                "--workers",
                str(min(max(len(models), 1), 4)),
            ],
            cwd=str(TOKENSTAR_RUNTIME_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=3600,
        )
        report_path = TOKENSTAR_REPORT_DIR / "prod" / suite / "archive" / f"tokenstar_report_{run_id}.json"
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
            partial_payload = _read_json_report(report_path)
            raise TokenStarRunError(f"TokenStar 执行失败(code={proc.returncode}):\n{tail}", partial_payload)
        if not report_path.exists():
            raise RuntimeError("TokenStar 已完成但未找到 JSON 报告")
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        try:
            tmp_config.unlink()
        except OSError:
            pass


def _python_bin() -> Path:
    if TOKENSTAR_PYTHON.exists():
        return TOKENSTAR_PYTHON
    return Path(sys.executable)


def _read_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _persist_tokenstar_payload(run_id: str, payload: dict[str, Any], status: str = "done", error: str | None = None) -> None:
    summary = payload.get("summary") or {}
    rows = []
    for provider in payload.get("providers") or []:
        model = provider.get("model") or ""
        for item in provider.get("results") or []:
            result_status = STATUS_TO_UI.get(item.get("status"), item.get("status") or "")
            test_name = f"{item.get('test_name', '')} — {model}"
            rows.append({
                "test_name": test_name[:200],
                "status": result_status,
                "eval_type": EVAL_TYPE_LABEL.get(item.get("evaluation_type") or "", item.get("evaluation_type") or ""),
                "score": item.get("score"),
                "latency_ms": item.get("latency_ms"),
                "message": item.get("message") or "",
                "detail": item.get("detail") or {},
            })
    json_store.update_detection_run(run_id, {
        "status": status,
        "raw_payload": payload,
        "error": error or "",
        "results": rows,
        "summary": {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "warned": summary.get("warned", 0),
            "score": summary.get("hard_pass_rate", summary.get("success_rate", 0)),
            "hard_pass_rate": summary.get("hard_pass_rate"),
            "soft_health": summary.get("soft_health"),
            "critical_failed": summary.get("critical_failed"),
            "info_count": summary.get("info_count"),
            "overall": summary.get("overall"),
            "tokenstar_run_id": payload.get("run_id"),
            "error": error,
        },
        "finished_at": json_store.now_text(),
    })


def _build_markdown_report(run: Any, cfg: Any, payload: dict[str, Any]) -> str:
    providers = payload.get("providers") or []
    summary = payload.get("summary") or {}
    suite = payload.get("suite") or run.suite or ""
    run_id = payload.get("run_id") or (run.summary or {}).get("tokenstar_run_id") or run.id
    title = _report_title(suite, cfg.label or cfg.base_url or "Provider")
    tests = _load_tokenstar_suites().get(suite, [])
    started = run.started_at.strftime("%Y-%m-%d %H:%M:%S") if run.started_at else "-"
    finished = run.finished_at.strftime("%Y-%m-%d %H:%M:%S") if run.finished_at else "-"
    duration = _duration_seconds(run.started_at, run.finished_at)
    lines = [
        f"# {title}",
        "",
        "## 运行信息",
        "",
        f"- Run ID: `{run_id}`",
        f"- Base URL: `{cfg.base_url or '-'}`",
        f"- API Key: `{_masked_api_key(cfg)}`",
        f"- 测试套件: `{suite or '-'}`",
        f"- 模型数量: {len(providers)}",
        f"- 测试项数量: {len(tests) or _max_provider_total(providers)}",
        f"- 开始时间: {started}",
        f"- 结束时间: {finished}",
        f"- 运行时长: {_fmt_duration(duration)}",
        "",
    ]
    lines.extend(_confidence_section(summary, providers))
    lines.extend(_official_anthropic_section(payload, providers))
    lines.extend(_overall_summary_section(providers))
    lines.extend(_template_result_summary_section(providers))
    lines.extend(_template_matrix_section(providers))
    lines.extend(_key_issues_section(providers))
    lines.extend(_template_details_section(providers))
    return "\n".join(lines).rstrip() + "\n"


def _build_markdown_from_db(detail: dict[str, Any]) -> str:
    summary = detail.get("summary") or {}
    suite_key = detail.get("suite_key") or ""
    raw_payload = detail.get("raw_payload") or {}
    if raw_payload.get("providers"):
        payload = dict(raw_payload)
        payload.setdefault("summary", summary)
        payload.setdefault("suite", suite_key)
        cfg_info = detail.get("config") or {}
        fake_run = type("Run", (), {
            "id": detail.get("run_id"),
            "suite": suite_key,
            "started_at": _parse_dt(detail.get("started_at")),
            "finished_at": _parse_dt(detail.get("finished_at")),
            "summary": summary,
        })()
        fake_cfg = type("Cfg", (), {
            "label": cfg_info.get("label") or detail.get("config_id") or "Provider",
            "base_url": cfg_info.get("base_url") or "-",
            "api_key": "",
            "api_key_masked": cfg_info.get("api_key_masked") or "-",
            "api_format": cfg_info.get("api_format") or "",
        })()
        return _build_markdown_report(fake_run, fake_cfg, payload)

    rows = detail.get("results") or []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        test_name, model = _split_test_model(row.get("test_name") or "")
        item = {**row, "test_name": test_name}
        by_model.setdefault(model or "unknown", []).append(item)
    providers = []
    for model, items in by_model.items():
        passed = sum(1 for item in items if item.get("status") == "PASS")
        failed = sum(1 for item in items if item.get("status") == "FAIL")
        warned = sum(1 for item in items if item.get("status") == "WARN")
        latencies = [item.get("latency_ms") for item in items if item.get("latency_ms")]
        providers.append({
            "model": model,
            "provider": model,
            "results": [_db_result_to_payload(item) for item in items],
            "total": len(items),
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p95_latency_ms": _percentile(latencies, 0.95) if latencies else None,
        })
    payload = {
        "run_id": summary.get("tokenstar_run_id") or detail.get("run_id"),
        "suite": suite_key,
        "summary": summary,
        "providers": providers,
    }
    cfg_info = detail.get("config") or {}
    fake_run = type("Run", (), {
        "id": detail.get("run_id"),
        "suite": suite_key,
        "started_at": _parse_dt(detail.get("started_at")),
        "finished_at": _parse_dt(detail.get("finished_at")),
        "summary": summary,
    })()
    fake_cfg = type("Cfg", (), {
        "label": cfg_info.get("label") or detail.get("config_id") or "Provider",
        "base_url": cfg_info.get("base_url") or "-",
        "api_key": "",
        "api_key_masked": cfg_info.get("api_key_masked") or "-",
        "api_format": cfg_info.get("api_format") or "",
    })()
    return _build_markdown_report(fake_run, fake_cfg, payload)


def _confidence_section(summary: dict[str, Any], providers: list[dict[str, Any]]) -> list[str]:
    failed = int(summary.get("failed") or _sum_provider_count(providers, "failed") or 0)
    warned = int(summary.get("warned") or _sum_provider_count(providers, "warned") or 0)
    hard_rate = summary.get("hard_pass_rate", summary.get("score"))
    long_failed = _count_status(providers, "long_context_recall", "fail")
    overall = summary.get("overall") or _overall_from_summary(summary)
    if failed == 0 and warned == 0:
        usability = f"高置信。全部测试项通过，硬指标通过率 {_fmt_rate(hard_rate)}。"
    elif failed == 0:
        usability = f"中高置信。无失败项，存在 {warned} 个警告项，硬指标通过率 {_fmt_rate(hard_rate)}。"
    else:
        usability = f"需关注。存在 {failed} 个失败项、{warned} 个警告项，硬指标通过率 {_fmt_rate(hard_rate)}。"
    risk_parts = []
    if failed:
        risk_parts.append(f"{failed} 个失败项")
    if warned:
        risk_parts.append(f"{warned} 个警告项")
    if long_failed:
        risk_parts.append(f"长上下文召回失败 {long_failed} 项")
    risk_text = "；".join(risk_parts) if risk_parts else "未发现明显高风险项。"
    suggestion = "优先处理失败项，再复测警告项。" if failed or warned else "当前结果可作为可用基线，建议保留周期性巡检。"
    return [
        "## 最终置信度判断",
        "",
        "| 结论项 | 判断 |",
        "|---|---|",
        f"| 整体可用性 | {usability}整体状态：{overall}。 |",
        f"| Anthropic Prompt Cache | {_prompt_cache_confidence(providers, 'Anthropic')} |",
        f"| OpenAI Prompt Cache | {_prompt_cache_confidence(providers, 'OpenAI')} |",
        f"| 风险点 | {risk_text} |",
        f"| 最终建议 | {suggestion} |",
        "",
    ]


def _prompt_cache_confidence(providers: list[dict[str, Any]], dialect: str) -> str:
    matched = [provider for provider in providers if _provider_dialect(provider) == dialect]
    if not matched:
        return f"暂无 {dialect} 口径数据。"

    hit_count = 0
    request_count = 0
    ratio_values: list[float] = []
    status_counts = {"pass": 0, "fail": 0, "warn": 0, "info": 0}
    for provider in matched:
        item = _prompt_cache_result(provider)
        if not item:
            continue
        status = _status_key(item.get("status"))
        if status in status_counts:
            status_counts[status] += 1
        detail = item.get("detail") or {}
        numbers = _cache_hit_numbers(detail)
        if numbers:
            hits, total = numbers
            hit_count += hits
            request_count += total
        ratio = _cache_ratio_value(detail)
        if ratio is not None:
            ratio_values.append(ratio)

    cached_tokens = sum(_provider_usage(provider)["cached_tokens"] for provider in matched)
    if request_count:
        token_text = "缓存 Token 为 0" if cached_tokens == 0 else f"缓存 Token {_fmt_int(cached_tokens)}"
        ratio_text = _cache_ratio_range(ratio_values)
        metric_text = f"{dialect} 口径缓存命中 {hit_count}/{request_count}"
        if ratio_text:
            metric_text += f"，缓存比例约 {ratio_text}"
        metric_text += f"，{token_text}。"
        if hit_count == request_count and cached_tokens > 0:
            return f"高置信。{metric_text}"
        if hit_count == 0 and cached_tokens == 0:
            return f"高置信不支持或未透出。{metric_text}"
        return f"需关注。{metric_text}"

    observed = sum(status_counts.values())
    if observed:
        return (
            f"已有 {observed} 个 {dialect} Prompt Cache 结果，"
            f"成功 {status_counts['pass']}、失败 {status_counts['fail']}、警告 {status_counts['warn']}、信息 {status_counts['info']}。"
        )
    if cached_tokens:
        return f"存在缓存 Token {_fmt_int(cached_tokens)}，但暂无可计算命中率的 Prompt Cache 结果。"
    return f"暂无 {dialect} Prompt Cache 专项数据。"


def _prompt_cache_result(provider: dict[str, Any]) -> dict[str, Any] | None:
    for name in CACHE_CONFIDENCE_TESTS:
        item = _result_by_name(provider, name)
        if item:
            return item
    return None


def _cache_hit_numbers(detail: dict[str, Any]) -> tuple[int, int] | None:
    hit = _first_present(detail, "hits", "cache_hits", "hit_count")
    total = _first_present(detail, "total", "requests", "request_count")
    if hit is None and total is None:
        probes = detail.get("probes") or []
        if probes:
            hit = sum(1 for item in probes if item.get("field_hit") or item.get("cache_hit"))
            total = len(probes)
    if hit is None and total is None:
        return None
    return _safe_int(hit), _safe_int(total)


def _cache_ratio_value(detail: dict[str, Any]) -> float | None:
    value = _first_present(detail, "cache_token_rate", "token_cache_rate", "cache_ratio")
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio * 100 if ratio <= 1 else ratio


def _cache_ratio_range(values: list[float]) -> str:
    if not values:
        return ""
    low = round(min(values), 2)
    high = round(max(values), 2)
    if low == high:
        return f"{low}%"
    return f"{low}%-{high}%"


def _first_present(value: dict[str, Any], *keys: str):
    for key in keys:
        if key in value and value.get(key) is not None:
            return value.get(key)
    return None


def _safe_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _official_anthropic_section(payload: dict[str, Any], providers: list[dict[str, Any]]) -> list[str]:
    comparisons = payload.get("baseline_comparisons") or payload.get("official_anthropic_comparison") or []
    lines = [
        "## 官方 Anthropic Token 对比",
        "",
        "> 仅对比 Anthropic 口径：官方 `api.anthropic.com/v1/messages` vs 中转 Anthropic。",
        "",
    ]
    if comparisons:
        lines.extend([
            "| 模型 | 官方命中 | 中转命中 | 官方前缀Token | 中转前缀Token | 官方Input分母 | 中转Input分母 | 官方缓存比例 | 中转缓存比例 | 差值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in comparisons:
            lines.append(
                f"| {item.get('model') or '-'} "
                f"| {_fmt_value(item.get('official_hits'))} "
                f"| {_fmt_value(item.get('relay_hits') or item.get('proxy_hits'))} "
                f"| {_fmt_int(item.get('official_prefix_tokens'))} "
                f"| {_fmt_int(item.get('relay_prefix_tokens') or item.get('proxy_prefix_tokens'))} "
                f"| {_fmt_int(item.get('official_input_tokens'))} "
                f"| {_fmt_int(item.get('relay_input_tokens') or item.get('proxy_input_tokens'))} "
                f"| {_fmt_pct_value(item.get('official_cache_ratio'))} "
                f"| {_fmt_pct_value(item.get('relay_cache_ratio') or item.get('proxy_cache_ratio'))} "
                f"| {_fmt_value(item.get('delta') or item.get('diff'))} |"
            )
    else:
        rows = []
        for provider in providers:
            if _provider_dialect(provider) != "Anthropic":
                continue
            baseline = _result_by_name(provider, "cache_official_baseline")
            rate = _result_by_name(provider, "cache_hit_rate")
            if baseline or rate:
                detail = (rate or {}).get("detail") or {}
                rows.append((provider, baseline, rate, detail))
        if not rows:
            lines.append("- 暂无官方 Anthropic 对比数据。")
        else:
            lines.extend([
                "| 模型 | 官方命中 | 中转命中 | 官方前缀Token | 中转前缀Token | 官方Input分母 | 中转Input分母 | 官方缓存比例 | 中转缓存比例 | 差值 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for provider, baseline, rate, detail in rows:
                lines.append(
                    f"| {provider.get('model') or provider.get('provider') or '-'} "
                    f"| {_cache_hit_text((baseline or {}).get('detail') or {})} "
                    f"| {_cache_hit_text(detail)} "
                    f"| {_fmt_int(_nested_get(detail, 'official.prefix_tokens'))} "
                    f"| {_fmt_int(_nested_get(detail, 'relay.prefix_tokens'))} "
                    f"| {_fmt_int(_nested_get(detail, 'official.input_tokens'))} "
                    f"| {_fmt_int(_nested_get(detail, 'relay.input_tokens'))} "
                    f"| {_fmt_pct_value(_nested_get(detail, 'official.cache_ratio'))} "
                    f"| {_fmt_pct_value(_nested_get(detail, 'relay.cache_ratio'))} "
                    f"| {_fmt_value(_nested_get(detail, 'delta'))} |"
                )
    lines.append("")
    return lines


def _overall_summary_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 整体汇总",
        "",
        "| 口径 | 模型 | 联通性 | 首 Token | 平均吐字速度 | 缓存 | 基准对照 | 长上下文召回 | 工具调用 | 结果概览 |",
        "|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for provider in providers:
        latency_detail = _result_detail(provider, "latency") or _result_detail(provider, "daily_latency") or _result_detail(provider, "first_token_connectivity")
        lines.append(
            f"| {_provider_dialect(provider)} "
            f"| {provider.get('model') or provider.get('provider') or '-'} "
            f"| {_result_message(provider, 'connectivity')} "
            f"| {_fmt_ms_unit(latency_detail.get('avg_first_token_ms') or latency_detail.get('first_token_ms'))} "
            f"| {_fmt_cps(latency_detail.get('avg_chars_per_second') or latency_detail.get('chars_per_second'))} "
            f"| {_cache_summary(provider)} "
            f"| {_result_message(provider, 'cache_official_baseline')} "
            f"| {_result_message(provider, 'long_context_recall')} "
            f"| {_result_message(provider, 'tool_call')} "
            f"| {_provider_count(provider, 'passed')}/{_provider_count(provider, 'total')}，失败{_provider_count(provider, 'failed')}，警告{_provider_count(provider, 'warned')} |"
        )
    lines.append("")
    return lines


def _template_result_summary_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 结果汇总",
        "",
        "| 口径 | 模型 | 通过/总数 | 失败 | 警告 | 平均延迟 | P95延迟 | 输入Token | 输出Token | 缓存Token |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in providers:
        usage = _provider_usage(provider)
        lines.append(
            f"| {_provider_dialect(provider)} "
            f"| {provider.get('model') or provider.get('provider') or '-'} "
            f"| {_provider_count(provider, 'passed')}/{_provider_count(provider, 'total')} "
            f"| {_provider_count(provider, 'failed')} "
            f"| {_provider_count(provider, 'warned')} "
            f"| {_fmt_ms_unit(provider.get('avg_latency_ms') or provider.get('latency'))} "
            f"| {_fmt_ms_unit(provider.get('p95_latency_ms'))} "
            f"| {_fmt_int(usage['input_tokens'])} "
            f"| {_fmt_int(usage['output_tokens'])} "
            f"| {_fmt_int(usage['cached_tokens'])} |"
        )
    lines.append("")
    return lines


def _template_matrix_section(providers: list[dict[str, Any]]) -> list[str]:
    tests = []
    seen = set()
    for provider in providers:
        for item in provider.get("results") or []:
            name = item.get("test_name") or "unknown"
            if name not in seen:
                seen.add(name)
                tests.append(name)
    lines = ["## 测试矩阵", ""]
    headers = " | ".join([str(p.get("model") or p.get("provider") or "-") for p in providers])
    lines.append(f"| 测试项 | 名称 | {headers} |")
    lines.append("|---|---|" + "---|" * len(providers))
    for test in tests:
        cells = []
        for provider in providers:
            item = _result_by_name(provider, test)
            cells.append(_matrix_cell(item))
        lines.append(f"| {test} | {_test_label(test)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _key_issues_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = ["## 关键问题", ""]
    issues = []
    for provider in providers:
        model = provider.get("model") or provider.get("provider") or "-"
        for item in provider.get("results") or []:
            if _status_key(item.get("status")) not in {"fail", "warn"}:
                continue
            name = item.get("test_name") or "unknown"
            issues.append(
                f"- `{model}` `{name}` {_test_label(name)}: {_status_cn(item.get('status'))}，{_escape_md(item.get('message') or '-')}"
            )
    if issues:
        lines.extend(issues)
    else:
        lines.append("- 暂无失败或警告项。")
    lines.append("")
    return lines


def _template_details_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = ["## 详细结果", ""]
    for provider in providers:
        model = provider.get("model") or provider.get("provider") or "-"
        lines.extend([f"### {model}", ""])
        for item in provider.get("results") or []:
            name = item.get("test_name") or "unknown"
            detail = item.get("detail") or {}
            lines.extend([
                f"#### {name} - {_test_label(name)}",
                "",
                f"- 状态: {_status_mark(item.get('status'))}",
                f"- 分数: {_fmt_value(item.get('score'))}",
                f"- 延迟: {_fmt_ms_unit(item.get('latency_ms'))}",
                f"- 结果: {_escape_md(item.get('message') or '-')}",
                "",
            ])
            if detail:
                lines.extend(["```json", json.dumps(detail, ensure_ascii=False, indent=2), "```", ""])
    return lines


def _summary_section(summary: dict[str, Any], providers: list[dict[str, Any]]) -> list[str]:
    usage = _sum_usage(providers)
    return [
        "## 总览",
        "",
        f"- 测试结果条数：{summary.get('total', _sum_provider_count(providers, 'total'))}",
        f"- 成功：{summary.get('passed', _sum_provider_count(providers, 'passed'))}",
        f"- 失败：{summary.get('failed', _sum_provider_count(providers, 'failed'))}",
        f"- 警告：{summary.get('warned', _sum_provider_count(providers, 'warned'))}",
        f"- 硬指标通过率：{_fmt_rate(summary.get('hard_pass_rate', summary.get('score')))}",
        f"- 软指标健康度：{_fmt_value(summary.get('soft_health'))}",
        f"- 安全底线失败：{_fmt_value(summary.get('critical_failed'))}",
        f"- 信息采集项：{_fmt_value(summary.get('info_count'))}",
        f"- 请求数：{usage['requests']}",
        f"- 输入 Token：{usage['input_tokens']}",
        f"- 输出 Token：{usage['output_tokens']}",
        f"- 缓存 Token：{usage['cached_tokens']}",
        f"- 整体状态：{summary.get('overall') or _overall_from_summary(summary)}",
        "",
    ]


def _model_summary_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 模型汇总",
        "",
        "| 模型 | 测试结果条数 | 成功 | 失败 | 警告 | 硬指标 | 软指标健康度 | 平均延迟 | P95 延迟 | 请求数 | 输入 Token | 输出 Token | 缓存 Token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in providers:
        usage = _provider_usage(provider)
        lines.append(
            f"| `{provider.get('model') or provider.get('provider') or '-'}` "
            f"| {_provider_count(provider, 'total')} "
            f"| {_provider_count(provider, 'passed')} "
            f"| {_provider_count(provider, 'failed')} "
            f"| {_provider_count(provider, 'warned')} "
            f"| {_fmt_hard(provider)} "
            f"| {_fmt_value(provider.get('soft_avg_score') or provider.get('soft_health'))} "
            f"| {_fmt_ms(provider.get('avg_latency_ms') or provider.get('latency'))} "
            f"| {_fmt_ms(provider.get('p95_latency_ms'))} "
            f"| {usage['requests']} "
            f"| {usage['input_tokens']} "
            f"| {usage['output_tokens']} "
            f"| {usage['cached_tokens']} |"
        )
    lines.append("")
    return lines


def _matrix_section(providers: list[dict[str, Any]]) -> list[str]:
    tests = []
    seen = set()
    for provider in providers:
        for item in provider.get("results") or []:
            name = item.get("test_name") or "unknown"
            if name not in seen:
                seen.add(name)
                tests.append(name)
    lines = ["## 测试项矩阵", ""]
    headers = " | ".join([f"`{p.get('model') or p.get('provider') or '-'}`" for p in providers])
    lines.append(f"| 测试项 | {headers} |")
    lines.append("|---|" + "---|" * len(providers))
    for test in tests:
        cells = []
        for provider in providers:
            item = next((r for r in provider.get("results") or [] if r.get("test_name") == test), None)
            cells.append(_status_cn((item or {}).get("status")))
        lines.append(f"| `{test}` {_test_label(test)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _suite_focus_section(suite: str, providers: list[dict[str, Any]]) -> list[str]:
    template = _report_template_type(suite)
    lines = [f"## {template}重点分析", ""]
    if template == "协议兼容报告":
        focus = ["protocol_text_shape", "protocol_usage_shape", "protocol_finish_reason", "protocol_stream_shape", "protocol_tool_call_shape", "protocol_error_shape"]
    elif template == "缓存专项报告":
        focus = ["cache_hit", "cache_random", "ctx_cache", "cache_hit_rate"]
    elif template == "能力探测报告":
        focus = ["vision_probe", "file_probe", "audio_probe", "embedding_probe"]
    elif template == "政治敏感合规报告":
        focus = ["political_evidence_territory", "political_evidence_history", "political_evidence_figure", "political_incitement_safety", "political_hate_safety", "political_rumor_uncertainty"]
    elif template in {"深度模型审计报告", "轻量模型审计报告"}:
        focus = ["identity", "tool_call", "streaming", "structured_output", "multi_turn", "long_context_recall", "fingerprint"]
    elif template == "并发压力报告":
        focus = ["concurrency"]
    else:
        focus = ["connectivity", "latency", "quality", "hallucination", "dialogue_reference"]
    rows = []
    for name in focus:
        matched = [item for provider in providers for item in provider.get("results") or [] if item.get("test_name") == name]
        if not matched:
            continue
        rows.append((name, matched))
    if not rows:
        lines.append("- 暂无该模板的重点测试项数据。")
        lines.append("")
        return lines
    lines.append("| 测试项 | 成功 | 失败 | 警告 | 信息 | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, matched in rows:
        lines.append(
            f"| `{name}` {_test_label(name)} "
            f"| {sum(1 for item in matched if _status_key(item.get('status')) == 'pass')} "
            f"| {sum(1 for item in matched if _status_key(item.get('status')) == 'fail')} "
            f"| {sum(1 for item in matched if _status_key(item.get('status')) == 'warn')} "
            f"| {sum(1 for item in matched if _status_key(item.get('status')) == 'info')} "
            f"| {_escape_md((matched[0].get('message') or '')[:120])} |"
        )
    lines.append("")
    return lines


def _details_section(providers: list[dict[str, Any]]) -> list[str]:
    lines = ["## 详细结果", ""]
    for provider in providers:
        model = provider.get("model") or provider.get("provider") or "-"
        lines.extend([f"## 模型：`{model}`", ""])
        for item in provider.get("results") or []:
            name = item.get("test_name") or "unknown"
            detail = item.get("detail") or {}
            lines.extend([
                f"### {name} - {_test_label(name)}",
                "",
                f"- 状态：{_status_cn(item.get('status'))}",
                f"- 分数：{_fmt_value(item.get('score'))}",
                f"- 延迟：{_fmt_ms(item.get('latency_ms'))}",
                f"- 评价类型：{item.get('evaluation_type') or detail.get('evaluation_type') or '-'}",
                f"- 消息：{_escape_md(item.get('message') or '-')}",
                "",
                "```json",
                json.dumps(detail, ensure_ascii=False, indent=2),
                "```",
                "",
            ])
    return lines


def _report_template_type(suite: str) -> str:
    if suite in {"all", "all_no_political"}:
        return "全量测试汇总报告"
    if suite in {"availability", "daily", "daily_full", "first_token_connectivity", "connectivity_matrix"}:
        return "可用性报告"
    if suite == "audit":
        return "深度模型审计报告"
    if suite == "model_audit":
        return "轻量模型审计报告"
    if suite in {"cache_audit", "benchmark"}:
        return "缓存专项报告"
    if suite == "concurrency_audit":
        return "并发压力报告"
    if suite == "protocol_audit":
        return "协议兼容报告"
    if suite == "capability_probe":
        return "能力探测报告"
    if suite == "political_sensitivity":
        return "政治敏感合规报告"
    return "模型检测报告"


def _report_title(suite: str, provider: str) -> str:
    return f"TokenStar {provider} {_report_template_type(suite)}"


def _mask_key(key: str) -> str:
    if not key:
        return "-"
    if len(key) <= 10:
        return "***"
    return f"{key[:3]}***{key[-5:]}"


def _masked_api_key(cfg) -> str:
    masked = getattr(cfg, "api_key_masked", "")
    if masked:
        return masked
    return _mask_key(getattr(cfg, "api_key", "") or "")


def _model_list(providers: list[dict[str, Any]]) -> str:
    models = [f"`{item.get('model') or item.get('provider') or '-'}`" for item in providers]
    return "、".join(models) if models else "-"


def _duration_seconds(start, end) -> str:
    if not start or not end:
        return "-"
    return str(round((end - start).total_seconds(), 2))


def _fmt_duration(value: str) -> str:
    if value in {"", "-"}:
        return "-"
    try:
        return f"{float(value):,.2f}s"
    except (TypeError, ValueError):
        return f"{value}s"


def _max_provider_total(providers: list[dict[str, Any]]) -> int:
    totals = [_provider_count(provider, "total") for provider in providers]
    return max(totals) if totals else 0


def _provider_count(provider: dict[str, Any], key: str) -> int:
    if provider.get(key) is not None:
        return int(provider.get(key) or 0)
    results = provider.get("results") or []
    if key == "total":
        return len(results)
    target = {"passed": "pass", "failed": "fail", "warned": "warn"}.get(key)
    return sum(1 for item in results if _status_key(item.get("status")) == target)


def _sum_provider_count(providers: list[dict[str, Any]], key: str) -> int:
    return sum(_provider_count(provider, key) for provider in providers)


def _status_key(status: str | None) -> str:
    value = str(status or "").lower()
    return {"成功": "pass", "失败": "fail", "警告": "warn", "信息": "info", "pass": "pass", "fail": "fail", "warn": "warn", "info": "info", "PASS": "pass", "FAIL": "fail", "WARN": "warn", "INFO": "info"}.get(value, value)


def _status_cn(status: str | None) -> str:
    return {"pass": "成功", "fail": "失败", "warn": "警告", "info": "信息"}.get(_status_key(status), status or "-")


def _fmt_rate(value) -> str:
    if value is None:
        return "-"
    num = float(value)
    if num <= 1:
        num *= 100
    return f"{round(num, 2)}%"


def _fmt_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return str(round(value, 2))
    return str(value)


def _fmt_int(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ms(value) -> str:
    if value is None:
        return "-"
    return str(round(float(value), 2))


def _fmt_ms_unit(value) -> str:
    if value is None:
        return "-"
    return f"{round(float(value))}ms"


def _fmt_cps(value) -> str:
    if value is None:
        return "-"
    return f"{round(float(value), 2)}字/秒"


def _fmt_pct_value(value) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num <= 1:
        num *= 100
    return f"{round(num, 2)}%"


def _fmt_hard(provider: dict[str, Any]) -> str:
    passed = provider.get("hard_passed")
    total = provider.get("hard_total")
    if passed is None or total is None:
        return "-"
    return f"{passed}/{total}"


def _overall_from_summary(summary: dict[str, Any]) -> str:
    if summary.get("failed"):
        return "失败"
    if summary.get("warned"):
        return "警告"
    return "成功"


def _provider_usage(provider: dict[str, Any]) -> dict[str, int]:
    usage = {
        "requests": int(provider.get("request_count") or 0),
        "input_tokens": int(provider.get("input_tokens") or 0),
        "output_tokens": int(provider.get("output_tokens") or 0),
        "cached_tokens": int(provider.get("cached_tokens") or 0),
    }
    if any(usage.values()):
        return usage
    for item in provider.get("results") or []:
        detail = item.get("detail") or {}
        item_usage = detail.get("usage") or {}
        records = detail.get("usage_records") or []
        usage["requests"] += max(1 if item_usage else 0, len(records))
        usage["input_tokens"] += int(item_usage.get("input_tokens") or 0)
        usage["output_tokens"] += int(item_usage.get("output_tokens") or 0)
        usage["cached_tokens"] += int(item_usage.get("cached_tokens") or item_usage.get("cache_read_tokens") or 0)
    return usage


def _sum_usage(providers: list[dict[str, Any]]) -> dict[str, int]:
    total = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    for provider in providers:
        usage = _provider_usage(provider)
        for key in total:
            total[key] += usage[key]
    return total


def _provider_dialect(provider: dict[str, Any]) -> str:
    value = str(provider.get("format") or provider.get("api_format") or provider.get("protocol") or "").lower()
    if value == "anthropic":
        return "Anthropic"
    if value == "responses":
        return "Responses"
    if value == "openai":
        return "OpenAI"
    text = f"{provider.get('provider') or ''} {provider.get('group') or ''}".lower()
    if "anthropic" in text:
        return "Anthropic"
    if "responses" in text:
        return "Responses"
    if "openai" in text:
        return "OpenAI"
    return "-"


def _result_by_name(provider: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((item for item in provider.get("results") or [] if item.get("test_name") == name), None)


def _result_detail(provider: dict[str, Any], name: str) -> dict[str, Any]:
    return (_result_by_name(provider, name) or {}).get("detail") or {}


def _result_message(provider: dict[str, Any], name: str) -> str:
    item = _result_by_name(provider, name)
    if not item:
        return "-"
    return _escape_md(item.get("message") or _status_cn(item.get("status")))


def _count_test(providers: list[dict[str, Any]], name: str) -> int:
    return sum(1 for provider in providers if _result_by_name(provider, name))


def _count_status(providers: list[dict[str, Any]], name: str, status: str) -> int:
    return sum(1 for provider in providers if _status_key((_result_by_name(provider, name) or {}).get("status")) == status)


def _cache_summary(provider: dict[str, Any]) -> str:
    rate = _result_by_name(provider, "cache_hit_rate")
    hit = _result_by_name(provider, "cache_hit")
    item = rate or hit
    if not item:
        return "-"
    detail = item.get("detail") or {}
    message = item.get("message") or _status_cn(item.get("status"))
    hit_text = _cache_hit_text(detail)
    token_ratio = _fmt_pct_value(detail.get("cache_token_rate") or detail.get("token_cache_rate") or detail.get("cache_ratio"))
    cached = detail.get("cached_tokens") or detail.get("cache_read_tokens")
    if hit_text != "-" or token_ratio != "-" or cached:
        return f"{_escape_md(message)}；命中 {hit_text}；Token {token_ratio}；{_fmt_int(cached)}"
    return _escape_md(message)


def _cache_hit_text(detail: dict[str, Any]) -> str:
    hit = detail.get("hits") or detail.get("cache_hits") or detail.get("hit_count")
    total = detail.get("total") or detail.get("requests") or detail.get("request_count")
    if hit is None and total is None:
        probes = detail.get("probes") or []
        if probes:
            hit = sum(1 for item in probes if item.get("field_hit") or item.get("cache_hit"))
            total = len(probes)
    if hit is None and total is None:
        return "-"
    return f"{hit or 0}/{total or 0}"


def _nested_get(value: dict[str, Any], path: str):
    current = value
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _matrix_cell(item: dict[str, Any] | None) -> str:
    if not item:
        return "-"
    score = item.get("score")
    suffix = f" ({_fmt_value(score)})" if score is not None else ""
    return f"{_status_mark(item.get('status'))}{suffix}"


def _status_mark(status: str | None) -> str:
    key = _status_key(status)
    if key == "pass":
        return "✓ 成功"
    if key == "fail":
        return "✕ 失败"
    if key == "warn":
        return "⚠ 警告"
    if key == "info":
        return "ℹ 信息"
    return status or "-"


def _test_label(name: str) -> str:
    labels = {
        "preflight": "预检",
        "connectivity": "连通性",
        "first_token_connectivity": "首 Token 连通",
        "daily_latency": "日常延迟",
        "latency": "延迟",
        "cache_hit": "模型命中缓存",
        "hallucination": "幻觉风险",
        "quality": "基础质量",
        "dialogue_reference": "参考对话",
        "streaming": "流式完整性",
        "token_inflation": "Token膨胀",
        "tool_call": "工具调用",
        "identity": "身份识别",
        "instruction_surface": "指令面探测",
        "fingerprint": "行为指纹",
        "cache_random": "随机缓存",
        "ctx_cache": "长上下文缓存",
        "cache_official_baseline": "官方缓存字段基准",
        "structured_output": "结构化输出",
        "multi_turn": "多轮记忆",
        "long_context_recall": "长上下文召回",
        "cache_hit_rate": "缓存命中率",
        "concurrency": "并发压力",
        "protocol_text_shape": "文本协议结构",
        "protocol_usage_shape": "Usage 协议结构",
        "protocol_finish_reason": "结束原因",
        "protocol_stream_shape": "流式协议结构",
        "protocol_tool_call_shape": "工具调用协议",
        "protocol_error_shape": "错误协议",
        "vision_probe": "视觉能力",
        "file_probe": "文件能力",
        "audio_probe": "音频能力",
        "embedding_probe": "向量能力",
        "political_evidence_territory": "领土证据",
        "political_evidence_history": "历史证据",
        "political_evidence_figure": "人物证据",
        "political_incitement_safety": "煽动安全",
        "political_hate_safety": "仇恨安全",
        "political_rumor_uncertainty": "传闻不确定性",
    }
    return labels.get(name, name)


def _split_test_model(value: str) -> tuple[str, str]:
    if " — " in value:
        left, right = value.rsplit(" — ", 1)
        return left, right
    return value, ""


def _db_result_to_payload(item: dict[str, Any]) -> dict[str, Any]:
    reverse_eval = {v: k for k, v in EVAL_TYPE_LABEL.items()}
    return {
        "test_name": item.get("test_name"),
        "status": {"PASS": "pass", "FAIL": "fail", "WARN": "warn", "INFO": "info"}.get(item.get("status"), item.get("status")),
        "evaluation_type": reverse_eval.get(item.get("eval_type"), item.get("eval_type")),
        "score": item.get("score"),
        "latency_ms": item.get("latency_ms"),
        "message": item.get("message"),
        "detail": item.get("detail") or {},
    }


def _percentile(values: list[int], q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _parse_dt(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _safe_filename(value: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", value).strip("_") or "Provider"


def _escape_md(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
