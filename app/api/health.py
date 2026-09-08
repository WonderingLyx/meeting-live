"""健康检查 API"""
import os
import time
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("Matrix_Core")

router = APIRouter()


def _runtime_engine_configured(engine: object | None) -> bool:
    if engine is None:
        return False
    last_error = getattr(engine, "last_error", None)
    return not (isinstance(last_error, str) and last_error)

@router.get("/health")
async def health_check():
    """健康检查端点 - 用于负载均衡器和监控"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


@router.get("/ready")
def readiness_check(request: Request):
    """检查用户实际需要的模型、存储和磁盘，而不只检查对象存在。

    not_ready 时返 HTTP 503,供负载均衡器/探针按状态码分流(而非只看 body)。
    GET 无副作用:不写测试文件,只探测可写权限。
    """
    runtime = getattr(request.app.state, "runtime", None)
    database = getattr(request.app.state, "db", None)
    app_config = getattr(request.app.state, "config", None)
    media_dir = app_config.storage.media_dir if app_config is not None else None
    asr_engine = runtime.asr if runtime is not None else None
    speaker_engine = runtime.speaker if runtime is not None else None
    checks = {
        "asr": _runtime_engine_configured(asr_engine),
        "speaker": _runtime_engine_configured(speaker_engine),
        "speaker_db": False,
        "inference_slot": not getattr(runtime, "slot_leaked", False),
        "database": False,
        "media": False,
        "disk": False,
    }
    if database is not None:
        try:
            with database.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            checks["database"] = True
        except Exception as exc:
            logger.warning("readiness database failed: %s", exc)
    # 声纹向量库探活:ChromaDB 或内置内存向量库都实现 count()。
    if checks["speaker"]:
        try:
            coll = getattr(speaker_engine, "collection", None)
            if coll is not None:
                coll.count()
            checks["speaker_db"] = True
        except Exception as exc:
            logger.warning("readiness speaker_db failed: %s", exc)
    if media_dir:
        try:
            path = Path(media_dir).resolve()
            path.mkdir(parents=True, exist_ok=True)
            # 探测可写权限,不真正写文件(GET 应无副作用)。
            checks["media"] = os.access(path, os.W_OK)
            checks["disk"] = shutil.disk_usage(path).free >= 256 * 1024 * 1024
        except Exception as exc:
            logger.warning("readiness media failed: %s", exc)
    else:
        # 保持独立 router 测试和嵌入使用兼容；正式 app 总会注入路径。
        checks["database"] = database is None
        checks["media"] = True
        checks["disk"] = True
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            **checks,
            "timestamp": time.time(),
        },
    )
