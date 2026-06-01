"""Lightweight image/video-aware probes for OpenAI-compatible relays."""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any
from urllib.parse import urljoin

import requests

from app.core.error_category import classify_error

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at runtime
    Image = None


IMAGE_PROMPT = "simple product photo of a white mug on a clean table"
EDIT_PROMPT = "turn this simple reference image into a clean ecommerce product photo"
BANNNA_PROMPT = "犬夜叉大战奈落，日式热血动漫风格，动态构图，刀光和妖气碰撞"

NON_CHAT_KEYWORDS = (
    "gpt-image",
    "image",
    "video",
    "audio",
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "flux",
    "stable-diffusion",
)


def is_likely_non_chat_model(model: str) -> bool:
    text = (model or "").lower()
    return any(keyword in text for keyword in NON_CHAT_KEYWORDS)


def skipped_chat_result(model: str) -> dict[str, Any]:
    message = "非聊天模型，请切换到图片/视频探针"
    return {
        "model": model,
        "ok": False,
        "skipped": True,
        "probe_type": "chat",
        "modality": "other",
        "endpoint": "",
        "latency_ms": 0,
        "first_token_ms": None,
        "chars_per_second": 0,
        "response_preview": "",
        "error": message,
        "error_category": "参数不兼容",
    }


def probe_multimodal(base_url: str, api_key: str, probe_type: str, model: str, timeout: int = 120) -> dict[str, Any]:
    started = time.time()
    try:
        if probe_type == "image_generation":
            return _probe_image_generation(base_url, api_key, model, started, timeout)
        if probe_type == "image_edit":
            return _probe_image_edit(base_url, api_key, model, started, timeout)
        if probe_type == "responses_image":
            return _probe_responses_image(base_url, api_key, model, started, timeout)
        if probe_type == "banna_image":
            return _probe_banna_image(base_url, api_key, model, started, timeout)
        return _error_row(model, probe_type, started, f"未知探针类型: {probe_type}")
    except Exception as exc:
        return _error_row(model, probe_type, started, str(exc))


def _probe_image_generation(base_url: str, api_key: str, model: str, started: float, timeout: int) -> dict[str, Any]:
    payload = {"model": model, "prompt": IMAGE_PROMPT, "n": 1, "clarity": "auto"}
    response = requests.post(
        _url(base_url, "/v1/images/generations"),
        headers=_bearer_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    return _parse_image_response(model, "image_generation", "images/generations", response, started)


def _probe_image_edit(base_url: str, api_key: str, model: str, started: float, timeout: int) -> dict[str, Any]:
    data = {"model": model, "prompt": EDIT_PROMPT, "n": "1", "clarity": "auto"}
    response = requests.post(
        _url(base_url, "/v1/images/edits"),
        headers=_auth_headers(api_key),
        data=data,
        files={"image": ("probe.png", _probe_png(), "image/png")},
        timeout=timeout,
    )
    return _parse_image_response(model, "image_edit", "images/edits", response, started)


def _probe_responses_image(base_url: str, api_key: str, model: str, started: float, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": IMAGE_PROMPT,
        "tools": [{"type": "image_generation", "model": model, "quality": "auto"}],
        "tool_choice": {"type": "image_generation"},
    }
    response = requests.post(
        _url(base_url, "/v1/responses"),
        headers=_bearer_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    return _parse_image_response(model, "responses_image", "responses:image_generation", response, started)


def _probe_banna_image(base_url: str, api_key: str, model: str, started: float, timeout: int) -> dict[str, Any]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": BANNNA_PROMPT}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": _aspect_ratio_from_model(model)},
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(
        _url(_strip_api_suffix(base_url), f"/v1beta/models/{model}:generateContent"),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return _parse_image_response(model, "banna_image", "v1beta:generateContent", response, started)


def _parse_image_response(model: str, probe_type: str, endpoint: str, response: requests.Response, started: float) -> dict[str, Any]:
    latency_ms = int((time.time() - started) * 1000)
    raw = _json_or_text(response)
    images = _extract_images(raw)
    error = "" if response.ok else _extract_error(raw, response)
    if response.ok and not images:
        error = "响应成功但未返回图片"
    first = images[0] if images else {}
    ok = response.ok and bool(images)
    preview = _image_preview(images, raw)
    return {
        "model": model,
        "ok": ok,
        "skipped": False,
        "probe_type": probe_type,
        "modality": "image",
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "first_token_ms": None,
        "chars_per_second": 0,
        "response_preview": preview,
        "error": "" if ok else error,
        "error_category": classify_error(error),
        "http_status": response.status_code,
        "image_width": first.get("width"),
        "image_height": first.get("height"),
        "image_source": first.get("source", ""),
    }


def _extract_images(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    images: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("data", []) or []):
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            images.append(_decode_info(item["b64_json"], f"data[{index}].b64_json"))
        elif item.get("url"):
            images.append({"source": f"data[{index}].url", "url": item.get("url")})
    for index, item in enumerate(raw.get("output", []) or []):
        if isinstance(item, dict) and item.get("type") == "image_generation_call" and item.get("result"):
            images.append(_decode_info(item["result"], f"output[{index}].result"))
    for c_index, candidate in enumerate(raw.get("candidates", []) or []):
        content = candidate.get("content") or {}
        for p_index, part in enumerate(content.get("parts", []) or []):
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                images.append(_decode_info(inline["data"], f"candidates[{c_index}].parts[{p_index}].inlineData"))
            elif isinstance(inline, dict) and (inline.get("oss_url") or inline.get("url")):
                images.append({"source": f"candidates[{c_index}].parts[{p_index}].inlineData.url", "url": inline.get("oss_url") or inline.get("url")})
    return [item for item in images if item]


def _decode_info(value: str, source: str) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(value)
    except Exception:
        return {"source": source}
    result: dict[str, Any] = {"source": source, "bytes": len(image_bytes)}
    if Image is not None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                result.update({"width": image.width, "height": image.height, "format": image.format or ""})
        except Exception:
            pass
    return result


def _image_preview(images: list[dict[str, Any]], raw: Any) -> str:
    if images:
        first = images[0]
        size = f"{first.get('width')}x{first.get('height')}" if first.get("width") and first.get("height") else "未知尺寸"
        if first.get("url"):
            return f"返回图片 URL，{size}"
        return f"返回图片，{size}"
    if isinstance(raw, dict):
        texts = []
        for candidate in raw.get("candidates", []) or []:
            for part in (candidate.get("content") or {}).get("parts", []) or []:
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
        if texts:
            return "\n".join(texts)[:200]
    return ""


def _error_row(model: str, probe_type: str, started: float, error: str) -> dict[str, Any]:
    return {
        "model": model,
        "ok": False,
        "skipped": False,
        "probe_type": probe_type,
        "modality": "image" if "image" in probe_type else "other",
        "endpoint": "",
        "latency_ms": int((time.time() - started) * 1000),
        "first_token_ms": None,
        "chars_per_second": 0,
        "response_preview": "",
        "error": error,
        "error_category": classify_error(error),
    }


def _extract_error(raw: Any, response: requests.Response) -> str:
    if isinstance(raw, dict):
        error = raw.get("error")
        if isinstance(error, dict):
            return error.get("message") or json.dumps(error, ensure_ascii=False)[:500]
        if error:
            return str(error)
        if raw.get("text"):
            return str(raw["text"])[:500]
    return response.text[:500]


def _json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text[:1000]}


def _probe_png() -> bytes:
    if Image is None:
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGN49+4dAAUtAq77L+1bAAAAAElFTkSuQmCC")
    image = Image.new("RGB", (256, 256), (236, 238, 242))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _bearer_headers(api_key: str) -> dict[str, str]:
    headers = _auth_headers(api_key)
    headers["Content-Type"] = "application/json"
    return headers


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _url(base_url: str, path: str) -> str:
    return urljoin(_strip_api_suffix(base_url).rstrip("/") + "/", path.lstrip("/"))


def _strip_api_suffix(raw: str) -> str:
    base = (raw or "").strip().rstrip("/")
    for marker in ("/v1beta", "/v1"):
        if marker in base:
            base = base[: base.index(marker)]
    return base.rstrip("/")


def _aspect_ratio_from_model(model: str) -> str:
    for ratio in ("16-9", "9-16", "1-1", "4-3", "3-4"):
        if f"_{ratio}_" in model:
            return ratio.replace("-", ":")
    return "1:1"
