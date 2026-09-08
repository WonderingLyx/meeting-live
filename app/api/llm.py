"""LLM API 端点 — LLM 关闭/失败时静默降级到 extractive,响应带 source 字段"""
import importlib
import logging
import os
from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.config import LLMConfig
from app.services.llm_gateway import LLMGateway, EndpointSecurityError
from app.services.llm_prompts import DEFAULT_PROMPTS
from app.services.model_config import (
    apply_model_settings_to_runtime,
    public_model_settings,
    read_model_settings,
    section_has_file_config,
    update_model_section,
)

logger = logging.getLogger("Matrix_LLM_API")

router = APIRouter()

# settings_repo 存 prompt 的 key 前缀:llm.prompt.<op>
LLM_PROMPT_KEY_PREFIX = "llm.prompt."


def _clean_secret(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.startswith("#") or text.startswith("＃"):
        return None
    return text


def _load_prompts(settings_repo) -> dict:
    """从 settings_repo 加载用户自定义 prompt,缺失的用 DEFAULT_PROMPTS 补。

    每次 _get_gateway 调用(每请求)读一次,保证 PUT 后立即可见,无内存副本陈旧。
    """
    prompts = dict(DEFAULT_PROMPTS)
    if settings_repo is None:
        return prompts
    for op in DEFAULT_PROMPTS:
        v = settings_repo.get(f"{LLM_PROMPT_KEY_PREFIX}{op}")
        if v:
            prompts[op] = v
    return prompts


def _save_prompts(settings_repo, payload: dict) -> dict:
    """写 settings_repo 持久化 prompt(重启不丢)。校验字段后逐 key 写入。"""
    for k, v in payload.items():
        if not isinstance(v, str):
            raise HTTPException(
                status_code=422,
                detail=f"prompt 字段 {k} 必须是字符串,实际 {type(v).__name__}",
            )
        settings_repo.set(f"{LLM_PROMPT_KEY_PREFIX}{k}", v)
    return _load_prompts(settings_repo)

LLM_SETTING_PREFIX = "llm."


def _llm_config_fingerprint(cfg: LLMConfig) -> tuple:
    """Identify the non-secret settings a connection test applies to."""
    return (
        cfg.enabled,
        cfg.endpoint.rstrip("/"),
        cfg.model,
        cfg.mock,
        cfg.allow_public,
    )


def _recent_probe(request: Request, cfg: LLMConfig) -> dict | None:
    result = getattr(request.app.state, "llm_probe_result", None)
    if not result or result.get("fingerprint") != _llm_config_fingerprint(cfg):
        return None
    return result


def _store_probe(
    request: Request,
    cfg: LLMConfig,
    *,
    available: bool,
    error: str | None = None,
) -> dict:
    result = {
        "fingerprint": _llm_config_fingerprint(cfg),
        "available": available,
        "last_tested_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    request.app.state.llm_probe_result = result
    return result


def _env_llm_cfg():
    """动态获取当前 config.llm(每次访问,便于测试 reload/monkeypatch)"""
    return importlib.import_module("app.config").config.llm


def _settings_repo(request: Request):
    return getattr(request.app.state, "settings_repo", None)


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "on")


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _effective_llm_cfg(repo=None) -> tuple[LLMConfig, str, Optional[str]]:
    """返回页面覆盖后的 LLMConfig.

    .env 提供默认值;旧 settings 表兼容读取;config/model-settings.json
    存在 llm 段时作为新的主配置源。API key 优先用环境变量,否则用
    model-settings.json。
    """
    base = _env_llm_cfg()
    cfg = base
    provider: Optional[str] = None
    source = "env"

    has_file_config = section_has_file_config("llm")
    if repo is not None and not has_file_config:
        provider = repo.get(f"{LLM_SETTING_PREFIX}provider")
        has_override = any(k.startswith(LLM_SETTING_PREFIX) for k in repo.all_keys())
        if has_override:
            cfg = replace(
                base,
                enabled=_to_bool(repo.get(f"{LLM_SETTING_PREFIX}enabled"), base.enabled),
                endpoint=repo.get(f"{LLM_SETTING_PREFIX}endpoint") or base.endpoint,
                model=repo.get(f"{LLM_SETTING_PREFIX}model") or base.model,
                # Secrets are environment/model-config-owned and are never loaded from SQLite.
                api_key=base.api_key,
                timeout_sec=_to_int(repo.get(f"{LLM_SETTING_PREFIX}timeout_sec"), base.timeout_sec),
                max_input_tokens=_to_int(repo.get(f"{LLM_SETTING_PREFIX}max_input_tokens"), base.max_input_tokens),
                mock=_to_bool(repo.get(f"{LLM_SETTING_PREFIX}mock"), base.mock),
                allow_public=_to_bool(repo.get(f"{LLM_SETTING_PREFIX}allow_public"), base.allow_public),
            )
            source = "settings"

    if has_file_config:
        try:
            llm_settings = read_model_settings(include_env_secrets=False).get("llm", {})
        except ValueError:
            llm_settings = {}
        provider = llm_settings.get("provider") or provider
        file_api_key = _clean_secret(llm_settings.get("api_key"))
        cfg = replace(
            cfg,
            enabled=_to_bool(llm_settings.get("enabled"), cfg.enabled),
            endpoint=llm_settings.get("endpoint") or cfg.endpoint,
            model=llm_settings.get("model") or cfg.model,
            api_key=base.api_key or file_api_key,
            timeout_sec=_to_int(str(llm_settings.get("timeout_sec")) if llm_settings.get("timeout_sec") is not None else None, cfg.timeout_sec),
            max_input_tokens=_to_int(str(llm_settings.get("max_input_tokens")) if llm_settings.get("max_input_tokens") is not None else None, cfg.max_input_tokens),
            mock=_to_bool(llm_settings.get("mock"), cfg.mock),
            allow_public=_to_bool(llm_settings.get("allow_public"), cfg.allow_public),
        )
        source = "model-settings"

    return cfg, source, provider


def _get_gateway(request: Request) -> LLMGateway:
    cfg, _source, _provider = _effective_llm_cfg(_settings_repo(request))
    prompts = _load_prompts(_settings_repo(request))
    try:
        return LLMGateway(cfg, prompts)
    except EndpointSecurityError:
        logger.warning("[LLM] endpoint 安全校验失败，已降级为本地摘要")
        return LLMGateway(replace(cfg, enabled=False), prompts)


def _is_authenticated_status_request(request: Request) -> bool:
    """LLM status 是白名单端点;这里手动判断是否可返回完整配置."""
    if os.environ.get("TEST_AUTH_BYPASS") == "1":
        return True
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return False
    auth_service = getattr(request.app.state, "auth_service", None)
    if auth_service is None:
        return False
    decoded = auth_service.decode_token(token)
    if not decoded:
        return False
    try:
        user = auth_service.get_user(int(decoded["sub"]))
    except (KeyError, TypeError, ValueError):
        return False
    if not user or not user.get("is_active") or user.get("must_change_password"):
        return False
    try:
        token_pwd_iat = float(decoded.get("pwd_iat", 0))
        current_pwd_iat = float(user.get("password_changed_at") or 0)
    except (TypeError, ValueError):
        return False
    return current_pwd_iat <= token_pwd_iat


def _normalize_llm_endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint 必须是 http:// 或 https:// 开头的完整地址")
    lowered = value.lower()
    for suffix in ("/chat/completions", "/models"):
        if lowered.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break
    return value


class LLMSettingsRequest(BaseModel):
    provider: str = Field("ollama", min_length=1, max_length=40)
    enabled: bool = False
    endpoint: str = Field(..., min_length=1, max_length=500)
    model: str = Field(..., min_length=1, max_length=200)
    api_key: Optional[str] = Field(None, max_length=500)
    allow_public: bool = False
    timeout_sec: int = Field(200, ge=1, le=600)
    max_input_tokens: int = Field(8000, ge=500, le=200000)
    mock: bool = False

    @field_validator("endpoint")
    @classmethod
    def _endpoint_must_be_openai_compatible_base(cls, value: str) -> str:
        return _normalize_llm_endpoint(value)

    @field_validator("provider", "model")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


def _ollama_tags_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/api/tags"


def _extract_openai_model_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else item
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return sorted(dict.fromkeys(ids), key=str.lower)


def _extract_ollama_model_ids(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("models")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("name") or item.get("model")
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return sorted(dict.fromkeys(ids), key=str.lower)


async def _get_json_with_llm_guard(gateway: LLMGateway, url: str, cfg: LLMConfig) -> object:
    headers = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    gateway._assert_no_dns_rebind()
    pinned_ip = gateway._resolve_pinned_ip(url)
    async with gateway._dns_pin_guard(url, pinned_ip):
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
            resp = await client.get(url, headers=headers)
    if 300 <= resp.status_code < 400:
        raise EndpointSecurityError("LLM endpoint returned redirect")
    resp.raise_for_status()
    return resp.json()


@router.get("/v1/llm/models")
async def llm_models(
    request: Request,
    provider_override: str | None = Query(None, alias="provider", max_length=40),
    endpoint: str | None = Query(None, max_length=500),
    allow_public: bool | None = Query(None),
):
    """List model ids from the configured LLM endpoint on explicit request."""
    cfg, _source, saved_provider = _effective_llm_cfg(_settings_repo(request))
    if endpoint is not None:
        try:
            endpoint = _normalize_llm_endpoint(endpoint)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        cfg = replace(cfg, endpoint=endpoint)
    if allow_public is not None:
        cfg = replace(cfg, allow_public=allow_public)
    provider = provider_override or saved_provider
    try:
        gateway = LLMGateway(replace(cfg, enabled=True, mock=False))
    except EndpointSecurityError:
        return {
            "items": [],
            "source": "security",
            "error": "LLM endpoint 配置不安全，默认只允许本机或内网地址",
        }

    attempts: list[tuple[str, str]] = [
        ("openai-compatible", f"{cfg.endpoint.rstrip('/')}/models")
    ]
    if (provider or "").lower() == "ollama" or urlparse(cfg.endpoint).port == 11434:
        ollama_url = _ollama_tags_url(cfg.endpoint)
        if ollama_url:
            attempts.append(("ollama-api", ollama_url))

    last_error = None
    for source, url in attempts:
        try:
            payload = await _get_json_with_llm_guard(gateway, url, cfg)
            ids = (
                _extract_ollama_model_ids(payload)
                if source == "ollama-api"
                else _extract_openai_model_ids(payload)
            )
            if ids:
                return {
                    "items": [{"id": item, "label": item} for item in ids],
                    "source": source,
                    "error": None,
                }
            last_error = f"{source} 未返回模型列表"
        except EndpointSecurityError:
            return {
                "items": [],
                "source": source,
                "error": "LLM endpoint 返回重定向或安全校验失败",
            }
        except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
            last_error = f"{source} 模型列表请求失败: {type(exc).__name__}"

    return {"items": [], "source": "none", "error": last_error or "未发现可用模型"}


@router.get("/v1/llm/status")
async def llm_status(request: Request):
    cfg, source, provider = _effective_llm_cfg(_settings_repo(request))
    if not _is_authenticated_status_request(request):
        return {
            "enabled": cfg.enabled,
            "available": False,
            "fallback": "extractive-textrank",
            "auth_required": True,
        }

    # Status is intentionally passive. It must never trigger DNS resolution or
    # an external LLM request just because a page was opened.
    probe = _recent_probe(request, cfg)
    return {
        "enabled": cfg.enabled,
        "available": probe["available"] if probe else None,
        "last_tested_at": probe["last_tested_at"] if probe else None,
        "endpoint": cfg.endpoint if cfg.enabled else None,
        "model": cfg.model if cfg.enabled else None,
        "mock": cfg.mock,
        "provider": provider or "custom",
        "allow_public": cfg.allow_public,
        "timeout_sec": cfg.timeout_sec,
        "max_input_tokens": cfg.max_input_tokens,
        "has_api_key": bool(cfg.api_key),
        "config_source": source,
        "config_path": public_model_settings()["config_path"],
        "error": probe["error"] if probe else None,
        "fallback": "extractive-textrank",
    }


@router.post("/v1/llm/test")
async def test_llm_connection(request: Request):
    """Explicitly test the saved LLM configuration once.

    This is the only settings/status endpoint allowed to call the configured
    ``/chat/completions`` service.
    """
    cfg, source, provider = _effective_llm_cfg(_settings_repo(request))
    error = None
    if not cfg.enabled:
        available = False
        error = "LLM 未启用"
    elif cfg.mock:
        available = True
    else:
        try:
            available = await LLMGateway(cfg).is_available()
            if not available:
                error = "LLM 连接测试失败"
        except EndpointSecurityError:
            available = False
            error = "LLM endpoint 配置不安全"

    probe = _store_probe(
        request,
        cfg,
        available=available,
        error=error,
    )
    return {
        "enabled": cfg.enabled,
        "available": probe["available"],
        "last_tested_at": probe["last_tested_at"],
        "endpoint": cfg.endpoint if cfg.enabled else None,
        "model": cfg.model if cfg.enabled else None,
        "mock": cfg.mock,
        "provider": provider or "custom",
        "allow_public": cfg.allow_public,
        "timeout_sec": cfg.timeout_sec,
        "max_input_tokens": cfg.max_input_tokens,
        "has_api_key": bool(cfg.api_key),
        "config_source": source,
        "config_path": public_model_settings()["config_path"],
        "error": probe["error"],
        "fallback": "extractive-textrank",
    }


@router.get("/v1/llm/settings")
def get_llm_settings(request: Request):
    cfg, source, provider = _effective_llm_cfg(_settings_repo(request))
    return {
        "provider": provider or "custom",
        "enabled": cfg.enabled,
        "endpoint": cfg.endpoint,
        "model": cfg.model,
        "allow_public": cfg.allow_public,
        "timeout_sec": cfg.timeout_sec,
        "max_input_tokens": cfg.max_input_tokens,
        "mock": cfg.mock,
        "has_api_key": bool(cfg.api_key),
        "api_key_preview": public_model_settings().get("llm", {}).get("api_key_preview"),
        "config_source": source,
        "config_path": public_model_settings()["config_path"],
    }


@router.put("/v1/llm/settings")
def update_llm_settings(body: LLMSettingsRequest, request: Request):
    repo = _settings_repo(request)
    if repo is None:
        raise HTTPException(status_code=500, detail="settings repository unavailable")

    try:
        update_model_section(
            "llm",
            {
                "provider": body.provider,
                "enabled": body.enabled,
                "endpoint": body.endpoint,
                "model": body.model,
                "api_key": body.api_key,
                "allow_public": body.allow_public,
                "timeout_sec": body.timeout_sec,
                "max_input_tokens": body.max_input_tokens,
                "mock": body.mock,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    repo.set(f"{LLM_SETTING_PREFIX}provider", body.provider)
    repo.set(f"{LLM_SETTING_PREFIX}enabled", str(body.enabled).lower())
    repo.set(f"{LLM_SETTING_PREFIX}endpoint", body.endpoint)
    repo.set(f"{LLM_SETTING_PREFIX}model", body.model)
    repo.set(f"{LLM_SETTING_PREFIX}allow_public", str(body.allow_public).lower())
    repo.set(f"{LLM_SETTING_PREFIX}timeout_sec", str(body.timeout_sec))
    repo.set(f"{LLM_SETTING_PREFIX}max_input_tokens", str(body.max_input_tokens))
    repo.set(f"{LLM_SETTING_PREFIX}mock", str(body.mock).lower())
    # Remove plaintext keys left by pre-release builds.
    repo.delete(f"{LLM_SETTING_PREFIX}api_key")
    apply_model_settings_to_runtime()
    # A result for the previous endpoint/model must not survive a save. Saving
    # is passive; a new result is created only by POST /v1/llm/test.
    request.app.state.llm_probe_result = None

    return get_llm_settings(request)


@router.get("/v1/llm/prompts")
def get_prompts(request: Request):
    # 从 settings_repo 读(含用户改的),缺失用 DEFAULT_PROMPTS。不返模块常量。
    return _load_prompts(_settings_repo(request))

@router.put("/v1/llm/prompts")
def update_prompts(payload: dict, request: Request):
    # 与 PUT /v1/llm/settings 一致:只走 JWT 鉴权中间件,不额外限制本机 host
    # (支持 LAN 部署从别的机器登录后修改 prompt)。
    # 未知字段直接 422,不静默丢弃
    unknown_keys = set(payload.keys()) - set(DEFAULT_PROMPTS.keys())
    if unknown_keys:
        raise HTTPException(
            status_code=422,
            detail=f"未知 prompt 字段: {sorted(unknown_keys)};合法字段: {sorted(DEFAULT_PROMPTS.keys())}",
        )
    # 长度/类型校验:防恶意写入超大 prompt 撑爆 DB 与 LLM 调用成本。
    for key, value in payload.items():
        if not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"prompt {key} 必须为字符串")
        if not value:
            # 拒空串:_load_prompts 读回时空串被当 falsy 跳过用默认,用户以为清空
            # 实则默认值仍在用,行为不一致。明确要求非空。
            raise HTTPException(status_code=422, detail=f"prompt {key} 不能为空")
        if len(value) > 20_000:
            raise HTTPException(
                status_code=422,
                detail=f"prompt {key} 超过 20000 字符上限",
            )
    repo = _settings_repo(request)
    if repo is None:
        raise HTTPException(status_code=500, detail="settings 仓库不可用")
    # 持久化到 settings_repo(重启不丢);_get_gateway 下次请求自动读最新。
    return _save_prompts(repo, payload)
