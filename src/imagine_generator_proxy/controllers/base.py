"""Shared controller plumbing: the Controller contract and request helpers.

Every controller is a class with injected dependencies and an `add_routes`
method — the aiohttp router is the only framework object it touches.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from aiohttp import web


@runtime_checkable
class Controller(Protocol):
	"""A group of related HTTP endpoints, self-registering on a router."""

	def add_routes(self, router: web.UrlDispatcher) -> None:
		...


async def json_body(request: web.Request) -> Optional[dict]:
	"""Parse a JSON request body, or None when the payload is not JSON."""
	try:
		body = await request.json()
		return body if isinstance(body, dict) else None
	except Exception:
		return None


def bad_request(message: str, **extra: Any) -> web.Response:
	"""400 with a consistent error envelope."""
	return web.json_response({"error": message, **extra}, status=400)


def not_found(message: str) -> web.Response:
	"""404 with a consistent error envelope."""
	return web.json_response({"error": message}, status=404)
