"""ASGI middleware: bearer auth, request id, structured access logs."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("app.access")

# Public paths that never require auth. Probes and OpenAPI tooling must work
# without credentials so load balancers and Swagger UI keep working.
_PUBLIC_PATH_PREFIXES = (
    "/healthz",
    "/health-check",
    "/readyz",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# Context variable holding the current request's ID. Log records read it via
# `RequestIdFilter` so every line is correlated even from background tasks.
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class BearerAuthMiddleware:
    """Reject requests without a valid bearer token, except on public paths.

    When `api_key` is empty the middleware is a no-op — useful for local dev.
    """

    def __init__(self, app: ASGIApp, *, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.api_key:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES):
            await self.app(scope, receive, send)
            return

        if not _has_valid_bearer(scope, self.api_key):
            await _send_unauthorized(send)
            return

        await self.app(scope, receive, send)


class RequestIdMiddleware:
    """Generate an X-Request-ID per request and bind it to logs.

    Honors an inbound `X-Request-ID` so external trace IDs survive end-to-end.
    """

    HEADER_NAME = b"x-request-id"

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _extract_header(scope, self.HEADER_NAME) or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((self.HEADER_NAME, request_id.encode("utf-8")))
                message["headers"] = headers
                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.info(
                    "request",
                    extra={
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": message.get("status"),
                        "latency_ms": round(latency_ms, 2),
                    },
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            request_id_ctx.reset(token)


class RequestIdFilter(logging.Filter):
    """Attach the current request_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _extract_header(scope: Scope, name: bytes) -> str | None:
    for header_name, header_value in scope.get("headers", []):
        if header_name == name:
            try:
                return header_value.decode("latin-1")
            except UnicodeDecodeError:
                return None
    return None


def _has_valid_bearer(scope: Scope, expected: str) -> bool:
    auth = _extract_header(scope, b"authorization")
    if not auth:
        return False
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return token.strip() == expected


async def _send_unauthorized(send: Send) -> None:
    body = (
        b'{"error":{"message":"Missing or invalid Authorization bearer token.",'
        b'"type":"invalid_request_error","code":"unauthorized"}}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="omniasr"'),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
