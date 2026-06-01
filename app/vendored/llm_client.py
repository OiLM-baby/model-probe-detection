"""大模型接口客户端 — 从 tokenstar 移植。

统一封装 OpenAI Chat Completions 和 Anthropic Messages 两类接口。
"""

import json
import time
from typing import Any

import requests

MAX_MODEL_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 64 * 1024


def _normalize_base_url(raw: str) -> str:
    """规范化 base_url，兼容各种常见写法：
    - https://api.openai.com/v1/          → https://api.openai.com
    - https://api.openai.com/v1           → https://api.openai.com
    - https://api.openai.com/             → https://api.openai.com
    - https://api.openai.com              → https://api.openai.com
    - https://api.openai.com/v1/chat/completions → https://api.openai.com  (截掉多余路径)
    """
    base = (raw or "").strip().rstrip("/")
    # 截掉 /v1 及其后面所有路径（如 /v1/chat/completions）
    if "/v1" in base:
        base = base[:base.index("/v1")]
    return base.rstrip("/")


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, api_format: str = "openai", timeout: int = 60):
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.timeout = min(timeout, MAX_MODEL_TIMEOUT_SECONDS)
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_format == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> tuple[list[str], int, str]:
        """拉取模型列表，返回 (model_ids, latency_ms, error)。"""
        url = f"{self.base_url}/v1/models"
        started = time.time()
        try:
            resp = self.session.get(url, headers=self._headers(), timeout=10)
            latency_ms = int((time.time() - started) * 1000)
            if resp.status_code != 200:
                return [], latency_ms, f"HTTP {resp.status_code}"
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            return models, latency_ms, ""
        except Exception as e:
            latency_ms = int((time.time() - started) * 1000)
            return [], latency_ms, str(e)

    def chat_stream_metrics(self, messages: list[dict], max_tokens: int = 64) -> tuple[str, int, int | None, float, str]:
        """先尝试流式，流式返回空或异常时降级非流式。
        返回 (text, latency_ms, first_token_ms, chars_per_second, error)
        """
        started = time.time()
        stream_error = ""
        text = ""
        latency_ms = 0
        first_token_ms = None

        # ── 1. 尝试流式 ──
        try:
            if self.api_format == "anthropic":
                text, latency_ms, first_token_ms = self._anthropic_stream(messages, max_tokens)
            elif self.api_format == "responses":
                text, latency_ms, first_token_ms = self._responses_stream(messages, max_tokens)
            else:
                text, latency_ms, first_token_ms = self._openai_stream(messages, max_tokens)
        except Exception as exc:
            stream_error = _tag_error(exc)
            latency_ms = int((time.time() - started) * 1000)

        # ── 2. 流式为空或失败 → 降级非流式 ──
        if not text:
            try:
                text, latency_ms, first_token_ms = self._fallback_non_stream(messages, max_tokens, started)
            except Exception as exc:
                # 两种方式都失败，返回最后的错误
                latency_ms = int((time.time() - started) * 1000)
                fallback_error = _tag_error(exc)
                final_error = (
                    f"非流式失败: {fallback_error}; 流式失败: {stream_error}"
                    if stream_error else fallback_error
                )
                return "", latency_ms, None, 0.0, final_error

        if text and first_token_ms is None:
            first_token_ms = latency_ms
        if first_token_ms is None or first_token_ms >= latency_ms:
            generation_ms = max(latency_ms, 1)
        else:
            generation_ms = max(latency_ms - first_token_ms, 1)
        chars_per_second = round(len(text) / (generation_ms / 1000), 2) if text else 0.0
        return text, latency_ms, first_token_ms, chars_per_second, ""

    def _fallback_non_stream(self, messages: list[dict], max_tokens: int, started: float) -> tuple[str, int, int | None]:
        """非流式降级请求。"""
        if self.api_format == "anthropic":
            url = f"{self.base_url}/v1/messages"
            payload = {
                "model": self.model,
                "messages": [m for m in messages if m["role"] != "system"],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            }
            system_msgs = [m["content"] for m in messages if m["role"] == "system"]
            if system_msgs:
                payload["system"] = "\n".join(system_msgs)
        elif self.api_format == "responses":
            url = f"{self.base_url}/v1/responses"
            input_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            payload = {
                "model": self.model,
                "input": input_text,
                "max_output_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            }
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            }

        resp = self.session.post(
            url,
            headers=self._headers(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
        )
        latency_ms = int((time.time() - started) * 1000)
        if not resp.ok:
            try:
                err = resp.json()
                msg = err.get("error", {}).get("message") if isinstance(err.get("error"), dict) else resp.text[:300]
            except Exception:
                msg = resp.text[:300]
            raise RuntimeError(f"HTTP {resp.status_code}: {msg}")

        data = resp.json()
        if self.api_format == "anthropic":
            parts = data.get("content") or []
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        elif self.api_format == "responses":
            text = data.get("output_text") or ""
            if not text:
                for item in data.get("output", []) or []:
                    for content in item.get("content", []) or []:
                        if content.get("type") in {"output_text", "text"}:
                            text += content.get("text", "")
        else:
            choices = data.get("choices") or []
            if choices:
                choice = choices[0]
                message = choice.get("message") or {}
                text, source = _extract_openai_message_text(message)
                if source == "reasoning_content" and choice.get("finish_reason") == "length":
                    raise RuntimeError("响应仅包含 reasoning_content 且 finish_reason=length，未生成最终内容")
            else:
                text = ""

        if not text:
            # 解析出空时把响应体带出来，方便排查
            snippet = json.dumps(data, ensure_ascii=False)[:400]
            raise RuntimeError(f"响应解析为空: {snippet}")

        first_token_ms = latency_ms
        return text, latency_ms, first_token_ms

    def _openai_stream(self, messages, max_tokens):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        return self._post_stream(url, payload, _extract_openai_text)

    def _anthropic_stream(self, messages, max_tokens):
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        if system_msgs:
            payload["system"] = "\n".join(system_msgs)
        return self._post_stream(url, payload, _extract_anthropic_text)

    def _responses_stream(self, messages, max_tokens):
        url = f"{self.base_url}/v1/responses"
        input_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        payload = {
            "model": self.model,
            "input": input_text,
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        return self._post_stream(url, payload, _extract_responses_text)

    def _post_stream(self, url, payload, extractor):
        started = time.time()
        text_parts = []
        first_token_ms = None
        with self.session.post(
            url,
            headers=self._headers(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
            stream=True,
        ) as response:
            if not response.ok:
                try:
                    err = response.json()
                    msg = err.get("error", {}).get("message") if isinstance(err.get("error"), dict) else response.text[:300]
                except Exception:
                    msg = response.text[:300]
                raise RuntimeError(f"HTTP {response.status_code}: {msg}")
            for raw_line in response.iter_lines(decode_unicode=True):
                if time.time() - started > MAX_MODEL_TIMEOUT_SECONDS:
                    raise TimeoutError("模型响应超时")
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                text = extractor(event)
                if text:
                    if first_token_ms is None:
                        first_token_ms = int((time.time() - started) * 1000)
                    text_parts.append(text)
        latency_ms = int((time.time() - started) * 1000)
        return "".join(text_parts), latency_ms, first_token_ms


def _extract_openai_text(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    text, _source = _extract_openai_message_text(delta, include_reasoning=False)
    return text


def _extract_openai_message_text(message: dict, include_reasoning: bool = True) -> tuple[str, str]:
    content = message.get("content")
    if isinstance(content, str) and content:
        return content, "content"
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
        if text:
            return text, "content"

    if not include_reasoning:
        return "", ""

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content, "reasoning_content"
    if isinstance(reasoning_content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in reasoning_content
        ), "reasoning_content"
    return "", ""


def _extract_responses_text(event: dict) -> str:
    """Extract text from OpenAI Responses API stream events."""
    # response.output_text.delta or response.content_part.added
    event_type = event.get("type", "")
    if event_type == "response.output_text.delta":
        return event.get("delta", "")
    if event_type == "response.content_part.added":
        part = event.get("part") or {}
        if part.get("type") == "output_text":
            return part.get("text", "")
    return ""


def _extract_anthropic_text(event: dict) -> str:
    if event.get("type") == "content_block_delta":
        return (event.get("delta") or {}).get("text") or ""
    return ""


def _tag_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"[timeout] 连接超时: {msg}"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"[timeout] 响应超时: {msg}"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"[network] {msg}"
    return msg
