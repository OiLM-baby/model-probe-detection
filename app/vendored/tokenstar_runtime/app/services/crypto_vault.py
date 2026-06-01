"""加密存储模块。

使用 Fernet (AES-128-CBC + HMAC-SHA256) 对称加密保护 API key、
webhook URL、邮箱密码等敏感配置字段。
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.storage._connect import SQLITE_BUSY_TIMEOUT_MS, connect_sqlite

logger = logging.getLogger("tokenstar")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VAULT_DB = os.path.join(_PROJECT_ROOT, "data", "vault.db")


def _connect(db_path: str) -> sqlite3.Connection:
    return connect_sqlite(db_path, foreign_keys=False, busy_timeout_ms=SQLITE_BUSY_TIMEOUT_MS)

# ── 数据模型 ──────────────────────────────────────────────


@dataclass
class VaultProvider:
    name: str
    base_url: str
    api_key: str
    format: str = "openai"
    enabled: bool = True
    models: list[str] = field(default_factory=list)
    group: str = ""
    timeout: int = 180
    extra_headers: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class VaultWechat:
    enabled: bool = False
    webhook_url: str = ""


@dataclass
class VaultEmail:
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 465
    username: str = ""
    password: str = ""
    sender: str = ""
    receivers: list[str] = field(default_factory=list)


# ── 密钥管理 ──────────────────────────────────────────────


def generate_key() -> str:
    """生成新的 Fernet 密钥。"""
    return Fernet.generate_key().decode("ascii")


def _find_key_file() -> str:
    """从常见位置查找 .tokenstar_key 文件，返回内容或空串。"""
    candidates = [
        os.path.join(os.path.expanduser("~/Desktop/api"), ".tokenstar_key"),
        os.path.join(os.path.dirname(__file__), "..", "..", ".tokenstar_key"),
        os.path.expanduser("~/.tokenstar_key"),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("TOKENSTAR_SECRET_KEY="):
                        return line.split("=", 1)[1]
        except (OSError, UnicodeDecodeError):
            continue
    return ""


def _get_fernet(key: str | None = None) -> Fernet | MultiFernet:
    env_key = key or os.environ.get("TOKENSTAR_SECRET_KEY", "") or _find_key_file()
    keys = [k for k in (env_key, os.environ.get("TOKENSTAR_SECRET_KEY_OLD", "")) if k]
    if not keys:
        raise RuntimeError("TOKENSTAR_SECRET_KEY 未设置，请设置环境变量或创建 Desktop/api/.tokenstar_key 文件")
    if len(keys) == 1:
        return Fernet(keys[0].encode("ascii"))
    return MultiFernet([Fernet(k.encode("ascii")) for k in keys])


def rotate_keys(db_path: str = VAULT_DB) -> int:
    """用新密钥重加密所有密文，旧密钥标记为 OLD。返回重加密行数。"""
    fernet = _get_fernet()
    if isinstance(fernet, Fernet):
        logger.info("只有一个密钥，无需轮换")
        return 0
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"vault 不存在: {db_path}")
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        # 轮换 verify_token
        new_fernet = Fernet(os.environ.get("TOKENSTAR_SECRET_KEY", "").encode("ascii"))
        verify_row = conn.execute("SELECT value FROM meta WHERE key='verify_token'").fetchone()
        if verify_row:
            plain = fernet.decrypt(verify_row[0].encode("ascii"))
            new_token = new_fernet.encrypt(plain).decode("ascii")
            conn.execute("UPDATE meta SET value=? WHERE key='verify_token'", (new_token,))

        count = 0

        # 轮换通知配置密文
        notif_row = conn.execute(
            "SELECT wechat_webhook_enc, email_password_enc FROM notification_secrets WHERE id=1"
        ).fetchone()
        if notif_row:
            notif_data = dict(notif_row)
            notif_updates = []
            notif_params = []
            for enc_col in ("wechat_webhook_enc", "email_password_enc"):
                old_cipher = notif_data.get(enc_col, "")
                if old_cipher:
                    plain = fernet.decrypt(old_cipher.encode("ascii")).decode("utf-8")
                    new_cipher = new_fernet.encrypt(plain.encode("utf-8")).decode("ascii")
                    notif_updates.append(f"{enc_col}=?")
                    notif_params.append(new_cipher)
            if notif_updates:
                conn.execute(
                    f"UPDATE notification_secrets SET {', '.join(notif_updates)} WHERE id=1",
                    notif_params,
                )
                count += 1

        # 轮换 provider 密文
        rows = conn.execute(
            "SELECT id, api_key_enc FROM provider_secrets"
        ).fetchall()
        for row in rows:
            data = dict(row)
            old_cipher = data.get("api_key_enc", "")
            if old_cipher:
                plain = fernet.decrypt(old_cipher.encode("ascii")).decode("utf-8")
                new_cipher = new_fernet.encrypt(plain.encode("utf-8")).decode("ascii")
                conn.execute(
                    "UPDATE provider_secrets SET api_key_enc=? WHERE id=?",
                    (new_cipher, data["id"]),
                )
                count += 1
        conn.commit()
        logger.info("密钥轮换完成，已重加密 %d 行", count)
        return count
    except Exception:
        conn.rollback()
        logger.exception("密钥轮换失败，已回滚")
        raise
    finally:
        conn.close()


# ── 字段加解密 ────────────────────────────────────────────


def encrypt_str(plaintext: str, key: str | None = None) -> str:
    if not plaintext:
        return ""
    return _get_fernet(key).encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_str(ciphertext: str, key: str | None = None) -> str:
    if not ciphertext:
        return ""
    return _get_fernet(key).decrypt(ciphertext.encode("ascii")).decode("utf-8")


# ── 数据库操作 ────────────────────────────────────────────


def init_vault(db_path: str = VAULT_DB) -> None:
    data_dir = os.path.dirname(db_path)
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS provider_secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                enabled INTEGER DEFAULT 1,
                format TEXT DEFAULT 'openai',
                base_url TEXT NOT NULL,
                api_key_enc TEXT NOT NULL,
                models_json TEXT NOT NULL DEFAULT '[]',
                group_name TEXT DEFAULT '',
                timeout INTEGER DEFAULT 180,
                extra_headers_json TEXT DEFAULT '{}',
                capabilities_json TEXT DEFAULT '{}',
                tags_json TEXT DEFAULT '[]',
                wechat_webhook_enc TEXT DEFAULT '',
                wechat_enabled INTEGER DEFAULT 0,
                email_enabled INTEGER DEFAULT 0,
                email_smtp_server TEXT DEFAULT '',
                email_smtp_port INTEGER DEFAULT 465,
                email_username TEXT DEFAULT '',
                email_password_enc TEXT DEFAULT '',
                email_sender TEXT DEFAULT '',
                email_receivers_json TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_secrets (
                id INTEGER PRIMARY KEY CHECK (id=1),
                wechat_enabled INTEGER DEFAULT 0,
                wechat_webhook_enc TEXT DEFAULT '',
                email_enabled INTEGER DEFAULT 0,
                email_smtp_server TEXT DEFAULT '',
                email_smtp_port INTEGER DEFAULT 465,
                email_username TEXT DEFAULT '',
                email_password_enc TEXT DEFAULT '',
                email_sender TEXT DEFAULT '',
                email_receivers_json TEXT DEFAULT '[]',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()
    os.chmod(db_path, 0o600)


# ── 密钥校验 ──────────────────────────────────────────────


def _verify_key(conn, fernet):
    row = conn.execute("SELECT value FROM meta WHERE key='verify_token'").fetchone()
    if not row:
        raise RuntimeError("vault 缺失校验 token，请重新运行 --init-vault")
    try:
        token = fernet.decrypt(row[0].encode("ascii")).decode("utf-8")
    except InvalidToken:
        raise RuntimeError("TOKENSTAR_SECRET_KEY 不匹配，请检查环境变量") from None
    if token != "tokenstar-vault-v1":
        raise RuntimeError("vault 校验失败")


# ── 导入 / 导出 ───────────────────────────────────────────


def import_providers(
    providers: list[dict[str, Any]],
    wechat_config: dict[str, Any] | None = None,
    email_config: dict[str, Any] | None = None,
    db_path: str = VAULT_DB,
    key: str | None = None,
) -> int:
    """将 provider 列表和通知配置加密后写入 SQLite。返回写入条数。"""
    fernet = _get_fernet(key)
    init_vault(db_path)
    wechat = wechat_config or {}
    email = email_config or {}

    conn = _connect(db_path)
    try:
        # 存入校验 token，用于启动时验证密钥正确性
        # MultiFernet.encrypt() 总是使用第一个密钥（即 TOKENSTAR_SECRET_KEY）
        verify_token = fernet.encrypt(b"tokenstar-vault-v1").decode("ascii")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('verify_token', ?)",
            (verify_token,),
        )

        # 通知配置写入独立的 notification_secrets 表（单行）
        wechat_webhook_enc = ""
        if wechat.get("webhook_url"):
            wechat_webhook_enc = fernet.encrypt(wechat["webhook_url"].encode("utf-8")).decode("ascii")
        email_password_enc = ""
        if email.get("password"):
            email_password_enc = fernet.encrypt(email["password"].encode("utf-8")).decode("ascii")
        conn.execute(
            """INSERT OR REPLACE INTO notification_secrets
               (id, wechat_enabled, wechat_webhook_enc,
                email_enabled, email_smtp_server, email_smtp_port,
                email_username, email_password_enc, email_sender,
                email_receivers_json, updated_at)
               VALUES (1,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (
                1 if wechat.get("enabled") else 0,
                wechat_webhook_enc,
                1 if email.get("enabled") else 0,
                email.get("smtp_server", ""),
                email.get("smtp_port", 465),
                email.get("username", ""),
                email_password_enc,
                email.get("sender", ""),
                json.dumps(email.get("receivers", []), ensure_ascii=False),
            ),
        )

        count = 0
        for p in providers:
            if not p.get("enabled", True):
                continue
            api_key = p.get("api_key", "")
            if not api_key:
                continue

            conn.execute(
                """INSERT OR REPLACE INTO provider_secrets
                   (name, enabled, format, base_url, api_key_enc, models_json,
                    group_name, timeout, extra_headers_json, capabilities_json,
                    tags_json, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                (
                    p.get("name", ""),
                    1 if p.get("enabled", True) else 0,
                    p.get("format", "openai"),
                    p.get("base_url", ""),
                    fernet.encrypt(api_key.encode("utf-8")).decode("ascii"),
                    json.dumps(p.get("models", []) if isinstance(p.get("models"), list) else [], ensure_ascii=False),
                    p.get("group", ""),
                    p.get("timeout", 180),
                    json.dumps(p.get("extra_headers", {}), ensure_ascii=False),
                    json.dumps(p.get("capabilities", {}), ensure_ascii=False),
                    json.dumps(p.get("tags", []) if isinstance(p.get("tags"), list) else [], ensure_ascii=False),
                ),
            )
            count += 1
        conn.commit()
        logger.info("已加密导入 %d 个中转配置到 %s", count, db_path)
        return count
    finally:
        conn.close()


def load_providers(
    db_path: str = VAULT_DB,
    key: str | None = None,
) -> tuple[list[VaultProvider], VaultWechat, VaultEmail]:
    """从加密数据库加载中转和通知配置。"""
    if not os.path.exists(db_path):
        return [], VaultWechat(), VaultEmail()
    fernet = _get_fernet(key)

    providers: list[VaultProvider] = []
    wechat = VaultWechat()
    email = VaultEmail()

    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # 校验密钥正确性（MultiFernet 自动尝试新旧密钥）
        _verify_key(conn, fernet)

        rows = conn.execute("SELECT * FROM provider_secrets WHERE enabled=1 ORDER BY id").fetchall()
        for row in rows:
            data = dict(row)
            providers.append(VaultProvider(
                name=data["name"],
                base_url=data["base_url"],
                api_key=fernet.decrypt(data["api_key_enc"].encode("ascii")).decode("utf-8"),
                format=data.get("format", "openai"),
                enabled=bool(data.get("enabled", 1)),
                models=json.loads(data.get("models_json", "[]")),
                group=data.get("group_name", ""),
                timeout=data.get("timeout", 180),
                extra_headers=json.loads(data.get("extra_headers_json", "{}")),
                capabilities=json.loads(data.get("capabilities_json", "{}")),
                tags=json.loads(data.get("tags_json", "[]")),
            ))

        # 优先从独立通知表加载，回退到 provider_secrets 旧字段（兼容旧 vault）
        notif_row = conn.execute("SELECT * FROM notification_secrets WHERE id=1").fetchone()
        if notif_row:
            data = dict(notif_row)
            wechat.enabled = bool(data.get("wechat_enabled"))
            if data.get("wechat_webhook_enc"):
                wechat.webhook_url = fernet.decrypt(data["wechat_webhook_enc"].encode("ascii")).decode("utf-8")
            email.enabled = bool(data.get("email_enabled"))
            email.smtp_server = data.get("email_smtp_server", "")
            email.smtp_port = int(data.get("email_smtp_port") or 465)
            email.username = data.get("email_username", "")
            if data.get("email_password_enc"):
                email.password = fernet.decrypt(data["email_password_enc"].encode("ascii")).decode("utf-8")
            email.sender = data.get("email_sender", "")
            receivers_raw = data.get("email_receivers_json", "[]")
            email.receivers = json.loads(receivers_raw) if receivers_raw else []
        else:
            # 回退：从 provider_secrets 旧字段读取通知配置
            for row in rows:
                data = dict(row)
                if data.get("wechat_webhook_enc") and not wechat.webhook_url:
                    wechat.enabled = bool(data.get("wechat_enabled"))
                    wechat.webhook_url = fernet.decrypt(data["wechat_webhook_enc"].encode("ascii")).decode("utf-8")
                if data.get("email_password_enc") and not email.password:
                    email.enabled = bool(data.get("email_enabled"))
                    email.smtp_server = data.get("email_smtp_server", "")
                    email.smtp_port = int(data.get("email_smtp_port") or 465)
                    email.username = data.get("email_username", "")
                    email.password = fernet.decrypt(data["email_password_enc"].encode("ascii")).decode("utf-8")
                    email.sender = data.get("email_sender", "")
                    receivers_raw = data.get("email_receivers_json", "[]")
                    email.receivers = json.loads(receivers_raw) if receivers_raw else []
                if wechat.webhook_url and email.password:
                    break
    finally:
        conn.close()

    return providers, wechat, email


def list_providers(db_path: str = VAULT_DB) -> list[dict[str, Any]]:
    """列出已存储的中转（不显示 API key）。"""
    if not os.path.exists(db_path):
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name, base_url, format, group_name, models_json, enabled FROM provider_secrets ORDER BY id"
        ).fetchall()
        return [
            {
                "name": r[0],
                "base_url": r[1],
                "format": r[2],
                "group": r[3] or r[0],
                "models": json.loads(r[4]) if r[4] else [],
                "enabled": bool(r[5]),
            }
            for r in rows
        ]
    finally:
        conn.close()
