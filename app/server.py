"""
FastAPI server with OpenAI Whisper-compatible API for Omnilingual-ASR.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.config import OMNILINGUAL_API_KEY, OMNILINGUAL_ROOT_PATH
from app.exceptions import APIError
from app.handlers import api_error_handler, validation_error_handler
from app.middleware import BearerAuthMiddleware, RequestIdMiddleware
from app.routes import router
from app.service import lifespan

_RAW_ROOT_PATH = OMNILINGUAL_ROOT_PATH.strip()
NORMALIZED_ROOT_PATH = (
    ""
    if not _RAW_ROOT_PATH or _RAW_ROOT_PATH == "/"
    else "/" + _RAW_ROOT_PATH.strip("/")
)


class RootPathPrefixMiddleware:
    """Allow direct access through a configured URL prefix such as `/omni-asr`."""

    def __init__(self, app, *, root_path: str):
        self.app = app
        self.root_path = root_path
        self.root_path_bytes = root_path.encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"} or not self.root_path:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == self.root_path or path.startswith(f"{self.root_path}/"):
            stripped_path = path[len(self.root_path) :] or "/"
            updated_scope = dict(scope)
            updated_scope["root_path"] = self.root_path
            updated_scope["path"] = stripped_path

            raw_path = scope.get("raw_path")
            if isinstance(raw_path, (bytes, bytearray)) and raw_path.startswith(
                self.root_path_bytes
            ):
                updated_scope["raw_path"] = raw_path[len(self.root_path_bytes) :] or b"/"

            await self.app(updated_scope, receive, send)
            return

        await self.app(scope, receive, send)


app = FastAPI(
    title="Omnilingual-ASR Server",
    description="OpenAI Whisper-compatible API for Omnilingual-ASR",
    version=__version__,
    lifespan=lifespan,
)
# Middleware ordering note: ASGI middleware execute in LIFO order around each
# request, so the *last* add_middleware call is the outermost wrapper. We
# want the request id to be set BEFORE auth runs (so unauthorized attempts
# are still logged with an id), and the path-prefix stripping to happen
# BEFORE any of that.
app.add_middleware(RootPathPrefixMiddleware, root_path=NORMALIZED_ROOT_PATH)
app.add_middleware(BearerAuthMiddleware, api_key=OMNILINGUAL_API_KEY)
app.add_middleware(RequestIdMiddleware)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.include_router(router)
