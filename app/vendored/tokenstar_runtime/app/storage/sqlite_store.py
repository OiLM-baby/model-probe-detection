"""v3 SQLite 存储实现，同时提供 v2 → v3 自动迁移。

Schema v3 整合 Part F-K 设计：
  - reports（单表 + suite_type 字段，替代 v2 的 runs + 4 张 report_* 表）
  - provider_snapshot（替代 provider_summaries，加打标字段）
  - test_results（沿用，外键改为 provider_snapshot_id）
  - alert_log（新加，provider 级故障事件）
  - db_meta（schema 版本追踪，避免与 vault.db 的 meta 表混淆）
"""

import json
import logging
import os
import sqlite3
import time
from typing import Any

from app.storage._connect import connect_sqlite
from app.utils.model_family import detect_family

logger = logging.getLogger("tokenstar")

SCHEMA_VERSION = 3
BEIJING_TIME_SQL = "datetime('now','+8 hours')"

_SUITE_TYPE_MAP = {
    "availability": "availability", "daily": "availability", "daily_full": "availability", "connectivity_matrix": "availability",
    "first_token_connectivity": "availability",
    "model_audit": "audit_light", "protocol_audit": "audit_light", "capability_probe": "audit_light",
    "audit": "audit_deep", "cache_audit": "audit_deep", "concurrency_audit": "audit_deep",
    "political_sensitivity": "audit_deep",
    "all": "live_board", "all_no_political": "live_board",
}

_TEMPLATE_MAP = {
    "availability": "availability_template.html",
    "audit_light": "model_audit_light_template.html",
    "audit_deep": "model_audit_deep_template.html",
    "live_board": "live_board_template.html",
}


def suite_type_for(suite: str) -> str:
    return _SUITE_TYPE_MAP.get(suite, "live_board")


def template_name_for(suite_type: str) -> str:
    return _TEMPLATE_MAP.get(suite_type, "live_board_template.html")


# ── DDL ────────────────────────────────────────────────────

_V3_DDL = """
CREATE TABLE IF NOT EXISTS db_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    env TEXT NOT NULL,
    suite TEXT NOT NULL,
    suite_type TEXT NOT NULL,
    template_name TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    crash_reason TEXT,
    overall_status TEXT,
    provider_count INTEGER,
    provider_count_planned INTEGER,
    total_tests INTEGER,
    passed INTEGER, failed INTEGER, warned INTEGER,
    hard_pass_rate REAL,
    soft_health REAL,
    critical_failed INTEGER,
    info_count INTEGER,
    request_count INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
    estimated_cost REAL, cost_currency TEXT,
    duration_seconds REAL,
    triggered_by TEXT DEFAULT 'cron',
    tags_json TEXT DEFAULT '[]',
    payload_json TEXT,
    created_at TEXT DEFAULT (datetime('now','+8 hours'))
);
CREATE INDEX IF NOT EXISTS idx_reports_env_suite ON reports(env, suite, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_suite_type ON reports(suite_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status, created_at DESC);

CREATE TABLE IF NOT EXISTS provider_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    group_name TEXT DEFAULT '',
    model_family TEXT DEFAULT '',
    provider_format TEXT DEFAULT '',
    base_url TEXT DEFAULT '',
    tags_json TEXT DEFAULT '[]',
    status TEXT DEFAULT 'completed',
    started_at TEXT, finished_at TEXT, crash_reason TEXT,
    total INTEGER, passed INTEGER, failed INTEGER, warned INTEGER,
    avg_latency_ms REAL, p95_latency_ms REAL,
    hard_total INTEGER, hard_passed INTEGER,
    soft_avg_score REAL,
    critical_failed INTEGER, info_count INTEGER,
    request_count INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
    estimated_cost REAL, cost_currency TEXT,
    price_matched INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ps_report ON provider_snapshot(report_id);
CREATE INDEX IF NOT EXISTS idx_ps_group ON provider_snapshot(group_name);
CREATE INDEX IF NOT EXISTS idx_ps_family ON provider_snapshot(model_family);
CREATE INDEX IF NOT EXISTS idx_ps_model ON provider_snapshot(model);
CREATE INDEX IF NOT EXISTS idx_ps_status ON provider_snapshot(status);

CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_snapshot_id INTEGER NOT NULL REFERENCES provider_snapshot(id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    latency_ms INTEGER,
    message TEXT,
    evaluation_type TEXT,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tr_snapshot_name ON test_results(provider_snapshot_id, test_name);

CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER REFERENCES reports(id) ON DELETE CASCADE,
    provider_snapshot_id INTEGER REFERENCES provider_snapshot(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    group_name TEXT,
    error_kind TEXT NOT NULL,
    error_message TEXT,
    test_name TEXT,
    occurred_at TEXT DEFAULT (datetime('now','+8 hours')),
    notified_wechat INTEGER DEFAULT 0,
    notified_email INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alert_provider_time ON alert_log(provider, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_kind_time ON alert_log(error_kind, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_report ON alert_log(report_id);

CREATE TABLE IF NOT EXISTS model_probe_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    first_token_ms INTEGER,
    chars_per_second REAL,
    char_count INTEGER,
    response_preview TEXT,
    error TEXT,
    error_category TEXT,
    error_detail TEXT,
    tested_at TEXT DEFAULT (datetime('now','+8 hours'))
);
CREATE INDEX IF NOT EXISTS idx_probe_group_model_time
    ON model_probe_log(group_name, model, tested_at DESC);
CREATE INDEX IF NOT EXISTS idx_probe_time
    ON model_probe_log(tested_at DESC);
CREATE INDEX IF NOT EXISTS idx_probe_ok_time
    ON model_probe_log(ok, tested_at DESC);
"""


# ── SqliteReportStore ──────────────────────────────────────


class SqliteReportStore:
    """v3 SQLite 报告存储。"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path)

    # ── 生命周期 ──────────────────────────────────────────

    def init(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if not parent:
            raise ValueError(f"history.db 路径缺少目录层级: {self.db_path!r}")
        os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            cur_version = self._detect_version(conn)
            if cur_version == 0:
                self._create_v3(conn)
                self._ensure_v3_extensions(conn)
                conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('schema_version','3')")
                conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('timezone','Asia/Shanghai')")
                conn.commit()
                logger.info("v3 schema 已创建: %s", self.db_path)
            elif cur_version < SCHEMA_VERSION:
                self._migrate(conn, cur_version)
            elif cur_version == SCHEMA_VERSION:
                self._ensure_v3_extensions(conn)
        finally:
            conn.close()

    def _detect_version(self, conn) -> int:
        try:
            row = conn.execute(
                "SELECT value FROM db_meta WHERE key='schema_version'"
            ).fetchone()
            if row:
                return int(row[0])
        except sqlite3.OperationalError:
            pass
        # v2 检测：有 runs 表但没有 db_meta
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        ).fetchone()
        if tables:
            return 2
        return 0

    def _create_v3(self, conn) -> None:
        conn.executescript(_V3_DDL)

    def _migrate(self, conn, from_version: int) -> None:
        if from_version == 2:
            self._migrate_v2_to_v3(conn)
        self._ensure_v3_extensions(conn)

    def _ensure_v3_extensions(self, conn) -> None:
        conn.executescript(_V3_DDL)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reports)").fetchall()}
        if "triggered_by" not in columns:
            conn.execute("ALTER TABLE reports ADD COLUMN triggered_by TEXT DEFAULT 'cron'")
        conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('timezone','Asia/Shanghai')")
        conn.commit()

    def _migrate_v2_to_v3(self, conn) -> None:
        """v2 → v3：备份 → 建 v3 表 → 迁移数据 → 升版本。"""
        backup_path = self.db_path + f".bak.v2.{int(time.time())}"
        logger.info("开始 v2 → v3 迁移，备份: %s", backup_path)

        # 备份
        src = self._connect()
        dst = connect_sqlite(backup_path)
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()

        try:
            conn.execute("BEGIN EXCLUSIVE")
            self._create_v3(conn)

            # 迁移 runs → reports
            suite_type_cols = self._migrate_runs_to_reports(conn)

            # 迁移 provider_summaries → provider_snapshot
            snapshot_id_map = self._migrate_provider_summaries(conn, suite_type_cols)

            # 迁移 test_results（外键改名 + 重建）
            self._migrate_test_results(conn, snapshot_id_map)

            # 标记 schema 版本
            conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('schema_version','3')")
            conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('timezone','Asia/Shanghai')")
            conn.commit()
            logger.info("v2 → v3 迁移完成")
        except Exception:
            conn.rollback()
            logger.exception("v2 → v3 迁移失败，已回滚。备份在 %s", backup_path)
            raise

    def _migrate_runs_to_reports(self, conn) -> dict[int, tuple[str, str, str]]:
        """迁移 runs 表到 reports，返回 {run_id: (suite_type, template_name, created_at)}。"""
        suite_info: dict[int, tuple[str, str, str]] = {}
        try:
            rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return suite_info

        col_names = [d[0] for d in rows[0].cursor.description] if rows else []
        for row in rows:
            data = dict(zip(col_names, row, strict=True))
            old_id = data["id"]
            suite = data.get("suite", "daily")
            stype = suite_type_for(suite)
            tpl = template_name_for(stype)
            created = data.get("created_at", data.get("generated_at", ""))
            suite_info[old_id] = (stype, tpl, created)

            conn.execute(
                """INSERT INTO reports
                   (run_id, env, suite, suite_type, template_name, status,
                    started_at, finished_at,
                    overall_status, provider_count, total_tests,
                    passed, failed, warned,
                    hard_pass_rate, soft_health, critical_failed, info_count,
                    request_count, input_tokens, output_tokens, cached_tokens,
                    estimated_cost, cost_currency, duration_seconds, created_at)
                   VALUES (?,?,?,?,?,'completed',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    data.get("run_id", ""),
                    data.get("env", ""),
                    suite, stype, tpl,
                    created, created,
                    data.get("overall_status", ""),
                    data.get("provider_count"),
                    data.get("total_tests"),
                    data.get("passed"),
                    data.get("failed"),
                    data.get("warned"),
                    data.get("hard_pass_rate"),
                    data.get("soft_health"),
                    data.get("critical_failed"),
                    data.get("info_count"),
                    data.get("request_count"),
                    data.get("input_tokens"),
                    data.get("output_tokens"),
                    data.get("cached_tokens"),
                    data.get("estimated_cost"),
                    data.get("cost_currency"),
                    data.get("duration_seconds"),
                    created,
                ),
            )
        return suite_info

    def _migrate_provider_summaries(self, conn, suite_info: dict) -> dict[int, int]:
        """迁移 provider_summaries → provider_snapshot，返回 {old_ps_id: new_snapshot_id}。"""
        id_map: dict[int, int] = {}
        try:
            rows = conn.execute("SELECT * FROM provider_summaries ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return id_map

        col_names = [d[0] for d in rows[0].cursor.description] if rows else []
        for row in rows:
            data = dict(zip(col_names, row, strict=True))
            old_id = data["id"]
            provider = data.get("provider", "")
            model = data.get("model", "")
            group = provider.split("__")[0] if "__" in provider else provider
            vendor, family = detect_family(model)
            tags = []
            if vendor != "unknown":
                tags = [f"vendor:{vendor}", f"family:{family}"]

            # 查找对应的 report_id
            old_run_fk = data.get("run_id_fk")
            new_report_id = None
            if old_run_fk:
                run_row = conn.execute(
                    "SELECT run_id FROM runs WHERE id=?", (old_run_fk,)
                ).fetchone()
                if run_row:
                    rep_row = conn.execute(
                        "SELECT id FROM reports WHERE run_id=?", (run_row[0],)
                    ).fetchone()
                    if rep_row:
                        new_report_id = rep_row[0]

            cursor = conn.execute(
                """INSERT INTO provider_snapshot
                   (report_id, provider, model, group_name, model_family,
                    tags_json, status,
                    total, passed, failed, warned,
                    avg_latency_ms, p95_latency_ms,
                    hard_total, hard_passed, soft_avg_score,
                    critical_failed, info_count,
                    request_count, input_tokens, output_tokens, cached_tokens,
                    estimated_cost, cost_currency, price_matched)
                   VALUES (?,?,?,?,?,?,'completed',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_report_id, provider, model, group, family,
                    json.dumps(tags, ensure_ascii=False),
                    data.get("total"), data.get("passed"), data.get("failed"),
                    data.get("warned"),
                    data.get("avg_latency_ms"), data.get("p95_latency_ms"),
                    data.get("hard_total"), data.get("hard_passed"),
                    data.get("soft_avg_score"),
                    data.get("critical_failed"), data.get("info_count"),
                    data.get("request_count"), data.get("input_tokens"),
                    data.get("output_tokens"), data.get("cached_tokens"),
                    data.get("estimated_cost"), data.get("cost_currency"),
                    data.get("price_matched"),
                ),
            )
            id_map[old_id] = cursor.lastrowid
        return id_map

    def _migrate_test_results(self, conn, id_map: dict[int, int]) -> None:
        """test_results 外键从 provider_summary_id 改为 provider_snapshot_id。"""
        try:
            rows = conn.execute("SELECT * FROM test_results ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return

        if not rows:
            return
        col_names = [d[0] for d in rows[0].cursor.description]
        old_results = [dict(zip(col_names, row, strict=True)) for row in rows]

        conn.execute("DROP TABLE IF EXISTS test_results")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_snapshot_id INTEGER NOT NULL REFERENCES provider_snapshot(id) ON DELETE CASCADE,
                test_name TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL,
                latency_ms INTEGER,
                message TEXT,
                evaluation_type TEXT,
                detail_json TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tr_snapshot_name ON test_results(provider_snapshot_id, test_name)"
        )

        for old in old_results:
            old_ps_id = old.get("provider_summary_id") or old.get("provider_snapshot_id")
            new_ps_id = id_map.get(old_ps_id, old_ps_id) if old_ps_id else None
            conn.execute(
                """INSERT INTO test_results
                   (id, provider_snapshot_id, test_name, status, score,
                    latency_ms, message, evaluation_type, detail_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    old.get("id"), new_ps_id,
                    old.get("test_name", ""), old.get("status", ""),
                    old.get("score"), old.get("latency_ms"),
                    old.get("message", ""), old.get("evaluation_type", ""),
                    old.get("detail_json"),
                ),
            )

    # ── Report 生命周期 ────────────────────────────────────

    def start_report(self, env: str, suite: str, run_id: str,
                     planned: int, suite_type: str = "",
                     template_name: str = "", triggered_by: str = "cron") -> int:
        stype = suite_type or suite_type_for(suite)
        tpl = template_name or template_name_for(stype)
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO reports
                   (run_id, env, suite, suite_type, template_name,
                    status, started_at, provider_count_planned, triggered_by)
                   VALUES (?,?,?,?,?,'running',datetime('now','+8 hours'),?,?)""",
                (run_id, env, suite, stype, tpl, planned, triggered_by or "cron"),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def write_model_probe_log(self, **payload) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO model_probe_log
                   (group_name, provider, model, prompt, ok, latency_ms, first_token_ms,
                    chars_per_second, char_count, response_preview, error,
                    error_category, error_detail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    payload.get("group_name", ""),
                    payload.get("provider", ""),
                    payload.get("model", ""),
                    str(payload.get("prompt", "") or "")[:500],
                    1 if payload.get("ok") else 0,
                    payload.get("latency_ms"),
                    payload.get("first_token_ms"),
                    payload.get("chars_per_second"),
                    payload.get("char_count"),
                    payload.get("response_preview", ""),
                    payload.get("error", ""),
                    payload.get("error_category", ""),
                    payload.get("error_detail", ""),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def list_model_probe_logs(
        self,
        group_name: str,
        model: str,
        start: str,
        end: str,
        *,
        page: int = 1,
        page_size: int = 50,
        ok: bool | None = None,
        error_category: str = "",
        min_first_token_ms: int | None = None,
    ) -> dict[str, Any]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            where = ["group_name=?", "model=?", "tested_at>=?", "tested_at<=?"]
            params: list[Any] = [group_name, model, start, end]
            if ok is not None:
                where.append("ok=?")
                params.append(1 if ok else 0)
            if error_category:
                where.append("error_category=?")
                params.append(error_category)
            if min_first_token_ms is not None:
                where.append("first_token_ms>=?")
                params.append(min_first_token_ms)
            where_clause = " AND ".join(where)
            total_row = conn.execute(f"SELECT COUNT(*) FROM model_probe_log WHERE {where_clause}", params).fetchone()
            total = total_row[0] if total_row else 0
            offset = (page - 1) * page_size
            rows = conn.execute(
                f"""SELECT * FROM model_probe_log
                    WHERE {where_clause}
                    ORDER BY tested_at DESC, id DESC
                    LIMIT ? OFFSET ?""",
                params + [page_size, offset],
            ).fetchall()
            items = [dict(row) for row in rows]
            for item in items:
                item["ok"] = bool(item.get("ok"))
            return {"items": items, "total": total, "page": page, "page_size": page_size}
        finally:
            conn.close()

    def mark_report_completed(self, report_id: int, payload_json: str = "") -> None:
        conn = self._connect()
        try:
            # 从 provider_snapshot 聚合统计
            stats = conn.execute(
                """SELECT COUNT(*) AS cnt,
                          COALESCE(SUM(total),0), COALESCE(SUM(passed),0),
                          COALESCE(SUM(failed),0), COALESCE(SUM(warned),0),
                          COALESCE(SUM(hard_total),0), COALESCE(SUM(hard_passed),0),
                          COALESCE(SUM(critical_failed),0), COALESCE(SUM(info_count),0),
                          COALESCE(SUM(request_count),0), COALESCE(SUM(input_tokens),0),
                          COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0)
                   FROM provider_snapshot WHERE report_id=?""",
                (report_id,),
            ).fetchone()

            soft_scores = conn.execute(
                "SELECT soft_avg_score FROM provider_snapshot WHERE report_id=? AND soft_avg_score IS NOT NULL",
                (report_id,),
            ).fetchall()

            provider_count = stats[0]
            total_tests = stats[1]
            passed = stats[2]
            failed = stats[3]
            warned = stats[4]
            hard_total = stats[5]
            hard_passed = stats[6]
            hard_pass_rate = round(hard_passed / hard_total * 100, 2) if hard_total else None
            soft_health = round(sum(s[0] for s in soft_scores) / len(soft_scores), 1) if soft_scores else None
            critical_failed = stats[7]
            info_count = stats[8]

            # 整体状态
            if critical_failed > 0:
                overall = "失败"
            elif hard_pass_rate is not None and hard_pass_rate < 80:
                overall = "失败"
            elif soft_health is not None and soft_health < 60:
                overall = "警告"
            elif failed > 0:
                overall = "警告"
            else:
                overall = "成功"

            # 聚合 cost 和 duration
            cost_row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost),0), MAX(cost_currency) FROM provider_snapshot WHERE report_id=?",
                (report_id,),
            ).fetchone()
            estimated_cost = cost_row[0] if cost_row else 0
            cost_currency = cost_row[1] or ""
            dur = conn.execute(
                "SELECT (julianday(datetime('now','+8 hours')) - julianday(started_at)) * 86400 FROM reports WHERE id=?",
                (report_id,),
            ).fetchone()
            duration_seconds = round(dur[0], 1) if dur and dur[0] else None

            conn.execute(
                """UPDATE reports SET status='completed', finished_at=datetime('now','+8 hours'),
                   overall_status=?, provider_count=?,
                   total_tests=?, passed=?, failed=?, warned=?,
                   hard_pass_rate=?, soft_health=?, critical_failed=?, info_count=?,
                   request_count=?, input_tokens=?, output_tokens=?, cached_tokens=?,
                   estimated_cost=?, cost_currency=?, duration_seconds=?,
                   payload_json=NULL
                   WHERE id=?""",
                (overall, provider_count,
                 total_tests, passed, failed, warned,
                 hard_pass_rate, soft_health, critical_failed, info_count,
                 stats[9], stats[10], stats[11], stats[12],
                 estimated_cost, cost_currency, duration_seconds,
                 report_id),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_report_crashed(self, report_id: int, reason: str,
                            payload_json: str = "") -> None:
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE reports SET status='crashed', finished_at=datetime('now','+8 hours'),
                   crash_reason=?, payload_json=NULL WHERE id=?""",
                (reason, report_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Provider 生命周期 ──────────────────────────────────

    def mark_provider_started(self, report_id: int, provider: str,
                              model: str, group_name: str = "",
                              model_family: str = "",
                              provider_format: str = "",
                              base_url: str = "",
                              tags_json: str = "[]") -> int:
        vendor, family = detect_family(model)
        mf = model_family or family
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO provider_snapshot
                   (report_id, provider, model, group_name, model_family,
                    provider_format, base_url, tags_json,
                    status, started_at)
                   VALUES (?,?,?,?,?,?,?,?,'running',datetime('now','+8 hours'))""",
                (report_id, provider, model, group_name or "",
                 mf, provider_format, base_url, tags_json),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def mark_provider_done(self, snapshot_id: int,
                           summary, results: list) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE provider_snapshot SET
                   status='completed', finished_at=datetime('now','+8 hours'),
                   total=?, passed=?, failed=?, warned=?,
                   avg_latency_ms=?, p95_latency_ms=?,
                   hard_total=?, hard_passed=?, soft_avg_score=?,
                   critical_failed=?, info_count=?,
                   request_count=?, input_tokens=?, output_tokens=?,
                   cached_tokens=?, estimated_cost=?, cost_currency=?,
                   price_matched=?
                   WHERE id=?""",
                (
                    summary.total, summary.passed, summary.failed, summary.warned,
                    summary.avg_latency_ms, summary.p95_latency_ms,
                    summary.hard_total, summary.hard_passed, summary.soft_avg_score,
                    summary.critical_failed, summary.info_count,
                    summary.request_count, summary.input_tokens,
                    summary.output_tokens, summary.cached_tokens,
                    summary.estimated_cost, summary.cost_currency,
                    1 if summary.price_matched else 0,
                    snapshot_id,
                ),
            )

            for result in results:
                detail = result.detail or {}
                conn.execute(
                    """INSERT INTO test_results
                       (provider_snapshot_id, test_name, status, score,
                        latency_ms, message, evaluation_type, detail_json)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        snapshot_id,
                        result.test_name,
                        str(result.status),
                        result.score,
                        result.latency_ms,
                        result.message or "",
                        result.evaluation_type,
                        json.dumps(detail, ensure_ascii=False) if detail else None,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def mark_provider_crashed(self, snapshot_id: int, error: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE provider_snapshot SET
                   status='crashed', finished_at=datetime('now','+8 hours'),
                   crash_reason=? WHERE id=?""",
                (error, snapshot_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Alert ──────────────────────────────────────────────

    def write_alert(self, snapshot_id: int | None = None, error_kind: str = "unknown",
                    error_msg: str = "", test_name: str = "",
                    report_id: int | None = None, provider: str = "", model: str = "",
                    group_name: str = "") -> int:
        conn = self._connect()
        try:
            ps_id: int | None = None
            rpt_id: int | None = report_id
            p, m, g = provider, model, group_name

            # 优先级 1: snapshot_id 有效
            if snapshot_id and snapshot_id > 0:
                ps = conn.execute(
                    "SELECT report_id, provider, model, group_name FROM provider_snapshot WHERE id=?",
                    (snapshot_id,),
                ).fetchone()
                if ps:
                    ps_id = snapshot_id
                    rpt_id = ps[0]
                    p = provider or ps[1]
                    m = model or ps[2]
                    g = group_name or ps[3] or ""

            # 优先级 2: 按 (report_id + provider + model) 反查
            if ps_id is None and rpt_id and provider:
                ps = conn.execute(
                    "SELECT id, group_name FROM provider_snapshot "
                    "WHERE report_id=? AND provider=? AND model=?",
                    (rpt_id, provider, model),
                ).fetchone()
                if ps:
                    ps_id = ps[0]
                    g = group_name or ps[1] or ""

            # 优先级 3: 都失败，ps_id 留 None

            cursor = conn.execute(
                """INSERT INTO alert_log
                   (report_id, provider_snapshot_id, provider, model, group_name,
                    error_kind, error_message, test_name)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (rpt_id, ps_id, p, m, g, error_kind, error_msg, test_name),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    # ── 查询 ───────────────────────────────────────────────

    def cleanup_zombie_reports(self, older_than_hours: int = 24) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """UPDATE reports SET status='crashed',
                   crash_reason='zombie cleanup', finished_at=datetime('now','+8 hours')
                   WHERE status='running'
                   AND started_at < datetime('now','+8 hours', ? || ' hours')""",
                (f"-{older_than_hours}",),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_completed_providers(self, report_id: int) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT provider FROM provider_snapshot WHERE report_id=? AND status='completed'",
                (report_id,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def list_recent_reports(self, env: str,
                            suite_type: str | None = None,
                            limit: int = 30) -> list[dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            if suite_type:
                rows = conn.execute(
                    "SELECT * FROM reports WHERE env=? AND suite_type=? ORDER BY created_at DESC LIMIT ?",
                    (env, suite_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM reports WHERE env=? ORDER BY created_at DESC LIMIT ?",
                    (env, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── 历史基线 ──────────────────────────────────────────

    def get_historical_baseline(self, env: str, suite: str,
                                 lookback: int = 10) -> dict[str, Any] | None:
        """从最近 N 次报告计算历史基线值，供偏差对比使用。"""
        if not os.path.exists(self.db_path):
            return None
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT hard_pass_rate, soft_health,
                          duration_seconds, estimated_cost, request_count,
                          provider_count
                   FROM reports
                   WHERE env=? AND suite=? AND status='completed'
                     AND hard_pass_rate IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (env, suite, lookback),
            ).fetchall()
            if not rows:
                return None
            rates = [r["hard_pass_rate"] for r in rows if r["hard_pass_rate"] is not None]
            scores = [r["soft_health"] for r in rows if r["soft_health"] is not None]
            durations = [r["duration_seconds"] for r in rows if r["duration_seconds"] is not None]
            costs = []
            for r in rows:
                if r["estimated_cost"] is not None and r["request_count"]:
                    costs.append(r["estimated_cost"] / r["request_count"])
            return {
                "lookback_runs": len(rows),
                "avg_hard_pass_rate": round(sum(rates) / len(rates), 2) if rates else None,
                "avg_soft_health": round(sum(scores) / len(scores), 2) if scores else None,
                "avg_duration_seconds": round(sum(durations) / len(durations), 1) if durations else None,
                "avg_cost_per_request": round(sum(costs) / len(costs), 6) if costs else None,
                "min_hard_pass_rate": round(min(rates), 2) if rates else None,
                "max_hard_pass_rate": round(max(rates), 2) if rates else None,
            }
        finally:
            conn.close()

    # ── 兼容旧 history.py API ─────────────────────────────

    def load_previous_run(self, env: str, suite: str) -> dict[str, Any] | None:
        """兼容 history.load_previous_run 的返回格式。"""
        if not os.path.exists(self.db_path):
            return None
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            run_row = conn.execute(
                """SELECT * FROM reports
                   WHERE env=? AND suite=? AND status='completed'
                   ORDER BY COALESCE(finished_at, started_at) DESC, id DESC
                   LIMIT 1""",
                (env, suite),
            ).fetchone()
            if not run_row:
                return None

            run = dict(run_row)
            ps_rows = conn.execute(
                "SELECT * FROM provider_snapshot WHERE report_id=? ORDER BY id",
                (run["id"],),
            ).fetchall()

            providers = []
            for ps in ps_rows:
                ps_dict = dict(ps)
                ps_dict.pop("id", None)
                ps_dict.pop("report_id", None)
                ps_dict["price_matched"] = bool(ps_dict.get("price_matched"))

                tr_rows = conn.execute(
                    """SELECT test_name, status, score, latency_ms,
                              message, evaluation_type, detail_json
                       FROM test_results WHERE provider_snapshot_id=? ORDER BY id""",
                    (ps["id"],),
                ).fetchall()

                results = []
                for tr in tr_rows:
                    tr_dict = dict(tr)
                    detail_json_str = tr_dict.pop("detail_json", None)
                    tr_dict["detail"] = json.loads(detail_json_str) if detail_json_str else {}
                    results.append(tr_dict)
                ps_dict["results"] = results
                providers.append(ps_dict)

            return {
                "env": run.get("env", ""),
                "suite": run.get("suite", ""),
                "run_id": run.get("run_id", ""),
                "generated_at": run.get("finished_at") or run.get("started_at", ""),
                "duration_seconds": run.get("duration_seconds", 0),
                "summary": {
                    "provider_count": run.get("provider_count"),
                    "total": run.get("total_tests"),
                    "passed": run.get("passed"),
                    "failed": run.get("failed"),
                    "warned": run.get("warned"),
                    "hard_pass_rate": run.get("hard_pass_rate"),
                    "soft_health": run.get("soft_health"),
                    "critical_failed": run.get("critical_failed"),
                    "info_count": run.get("info_count"),
                    "overall": run.get("overall_status"),
                    "request_count": run.get("request_count"),
                    "input_tokens": run.get("input_tokens"),
                    "output_tokens": run.get("output_tokens"),
                    "cached_tokens": run.get("cached_tokens"),
                    "estimated_cost": run.get("estimated_cost"),
                    "cost_currency": run.get("cost_currency"),
                },
                "providers": providers,
            }
        except Exception:
            logger.exception("加载历史记录失败: %s", self.db_path)
            return None
        finally:
            conn.close()

    def build_comparison(self, current: dict[str, Any],
                         previous: dict[str, Any]) -> dict[str, Any]:
        """兼容 history.build_comparison。"""
        cur_summary = current.get("summary") or {}
        prev_summary = previous.get("summary") or {}

        def _delta(key):
            cur = cur_summary.get(key)
            prev = prev_summary.get(key)
            if cur is None or prev is None:
                return None
            if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
                return round(cur - prev, 4)
            return None

        def _friendly(value, suffix=""):
            if value is None:
                return "-"
            sign = "+" if value > 0 else ""
            formatted = f"{value:,.2f}".rstrip("0").rstrip(".")
            text = f"{sign}{formatted}"
            return f"{text}{suffix}" if suffix else text

        def _trend(key):
            d = _delta(key)
            if d is None:
                return "neutral"
            if key in ("hard_pass_rate", "soft_health"):
                if d > 0.5:
                    return "up"
                if d < -0.5:
                    return "down"
                return "neutral"
            if key in ("failed", "critical_failed", "avg_latency_ms", "estimated_cost"):
                if d < -0.01:
                    return "up"
                if d > 0.01:
                    return "down"
                return "neutral"
            return "neutral"

        deltas = {}
        for key in ("hard_pass_rate", "soft_health", "critical_failed", "failed", "warned", "estimated_cost"):
            d = _delta(key)
            if d is not None:
                deltas[key] = {
                    "value": d,
                    "display": _friendly(d, "%" if key in ("hard_pass_rate",) else ""),
                    "trend": _trend(key),
                }

        cur_lats = [p.get("avg_latency_ms") for p in (current.get("providers") or []) if p.get("avg_latency_ms")]
        prev_lats = [p.get("avg_latency_ms") for p in (previous.get("providers") or []) if p.get("avg_latency_ms")]
        if cur_lats and prev_lats:
            cur_avg = sum(cur_lats) / len(cur_lats)
            prev_avg = sum(prev_lats) / len(prev_lats)
            lat_delta = round(cur_avg - prev_avg)
            lat_trend = "neutral"
            if lat_delta < -5:
                lat_trend = "up"
            elif lat_delta > 5:
                lat_trend = "down"
            deltas["avg_latency_ms"] = {
                "value": lat_delta,
                "display": _friendly(lat_delta, "ms"),
                "trend": lat_trend,
            }

        prev_providers = {}
        for p in previous.get("providers") or []:
            key_name = f"{p.get('provider', '')}||{p.get('model', '')}"
            prev_providers[key_name] = p

        provider_changes = []
        for p in current.get("providers") or []:
            key_name = f"{p.get('provider', '')}||{p.get('model', '')}"
            prev = prev_providers.get(key_name)
            if not prev:
                provider_changes.append({
                    "provider": p.get("provider", ""), "model": p.get("model", ""),
                    "change": "new", "label": "新增",
                })
                continue
            cur_failed = p.get("failed", 0) or 0
            prev_failed = prev.get("failed", 0) or 0
            cur_status = "fail" if cur_failed else "pass"
            prev_status = "fail" if prev_failed else "pass"
            if cur_status != prev_status:
                provider_changes.append({
                    "provider": p.get("provider", ""), "model": p.get("model", ""),
                    "change": "status", "from": prev_status, "to": cur_status,
                    "label": "恶化" if cur_status == "fail" else "恢复",
                })

        prev_keys = set(prev_providers.keys())
        cur_keys = {f"{p.get('provider', '')}||{p.get('model', '')}" for p in (current.get("providers") or [])}
        for removed_key in prev_keys - cur_keys:
            prev = prev_providers[removed_key]
            provider_changes.append({
                "provider": prev.get("provider", ""), "model": prev.get("model", ""),
                "change": "removed", "label": "移除",
            })

        return {
            "has_previous": True,
            "previous_run_id": previous.get("run_id", ""),
            "previous_time": previous.get("generated_at", ""),
            "deltas": deltas,
            "provider_changes": provider_changes[:20],
        }

    def prune_history(self, keep: int = 30,
                       older_than_days: int | None = None) -> int:
        """按 keep 数量 + 可选时间窗口双口子清理旧报告。"""
        conn = self._connect()
        try:
            deleted = 0
            groups = conn.execute("SELECT DISTINCT env, suite FROM reports").fetchall()
            for env, suite in groups:
                ids_to_remove: set[int] = set()

                # 1. 按 keep 数量
                rows = conn.execute(
                    "SELECT id FROM reports WHERE env=? AND suite=? ORDER BY created_at DESC",
                    (env, suite),
                ).fetchall()
                if len(rows) > keep:
                    ids_to_remove.update(row[0] for row in rows[keep:])

                # 2. 按时间窗口
                if older_than_days:
                    old_rows = conn.execute(
                        "SELECT id FROM reports WHERE env=? AND suite=?"
                        " AND created_at < datetime('now','+8 hours', ?)",
                        (env, suite, f"-{older_than_days} days"),
                    ).fetchall()
                    ids_to_remove.update(row[0] for row in old_rows)

                if ids_to_remove:
                    placeholders = ",".join("?" for _ in ids_to_remove)
                    conn.execute(
                        f"DELETE FROM reports WHERE id IN ({placeholders})",
                        list(ids_to_remove),
                    )
                    deleted += len(ids_to_remove)
            conn.commit()
            return deleted
        except Exception:
            logger.exception("清理历史记录失败: %s", self.db_path)
            return 0
        finally:
            conn.close()
