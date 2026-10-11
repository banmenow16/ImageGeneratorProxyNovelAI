"""MCP adapter over the image proxy's HTTP API (stdio transport).

Run by Claude Code as a project MCP server (see .mcp.json): every tool is
a typed, schema-generated wrapper around one proxy endpoint, so prompt
payloads (braces, quotes, danbooru tags) never touch shell quoting. The
proxy server stays the single source of truth — this process only
forwards requests and never imports server internals, so any server
restart or version change needs nothing here.

Requires the proxy running at IMAGE_PROXY_URL (default
http://localhost:8080); tools return the proxy's error JSON when it is
down instead of crashing the session. Handlers are async end-to-end —
sync handlers run on worker threads that can wedge the stdio transport
on Windows."""

from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

BASE_URL = os.environ.get("IMAGE_PROXY_URL", "http://localhost:8080").rstrip("/")
TIMEOUT_S = float(os.environ.get("IMAGE_PROXY_TIMEOUT", "300"))


class ImageProxyClient:
	"""Minimal async HTTP client for the proxy's JSON API."""

	def __init__(self, base_url: str, timeout_s: float) -> None:
		self._base = base_url
		self._timeout = aiohttp.ClientTimeout(total=timeout_s)

	async def get(self, path: str) -> Any:
		return await self._request("GET", path, None)

	async def post(self, path: str, payload: dict) -> Any:
		return await self._request("POST", path, payload)

	async def _request(self, method: str, path: str,
			payload: Optional[dict]) -> Any:
		url = self._base + path
		try:
			async with aiohttp.ClientSession(timeout=self._timeout) as session:
				async with session.request(method, url, json=payload) as resp:
					data: Any = await resp.json(content_type=None)
					if resp.status >= 400:
						return {"error": f"proxy returned {resp.status}: "
							+ str(data)[:500]}
					return data
		except aiohttp.ClientError as ex:
			return {"error": f"proxy unreachable at {self._base}: {ex}"}


client = ImageProxyClient(BASE_URL, TIMEOUT_S)

try:
	from fastmcp import FastMCP
except ImportError as ex:  # pragma: no cover — surfaced to the operator
	raise SystemExit(
		"fastmcp is required: pip install fastmcp (or use the project venv)") from ex

mcp = FastMCP("image-proxy")


@mcp.tool
async def generate_image(prompt: str, negative: str = "",
		engine: str = "novelai", save: bool = True) -> dict:
	"""Generate one image from a tag-list prompt via the proxy (NovelAI
	v4.5 backends). Returns the saved image URL/path, the massaged prompt
	the artist actually received, and the negative prompt used. Prompts
	are comma-separated danbooru tags; the proxy adds quality boosters
	and generic negatives automatically."""
	return await client.post("/api/generate", {"prompt": prompt,
		"negative": negative, "engine": engine, "save": save})


@mcp.tool
async def vision_analyze(images: list[str], prompt: str, model: str = "",
		temperature: float = 0.0, max_tokens: int = 300) -> dict:
	"""Ask a vision model (default qwen3-vl) about one to four stored
	images. Images are data-dir-relative paths (e.g.
	"advice/runs/19/nude_r1.png" or "debug/20261009_020508.png") or
	base64 payloads. For reliable answers: neutral factual questions with
	an explicit attention cue ("look closely at X"), temperature 0."""
	payload: dict = {"images": images, "prompt": prompt,
		"temperature": temperature, "max_tokens": max_tokens}
	if model:
		payload["model"] = model
	return await client.post("/api/vision/analyze", payload)


@mcp.tool
async def list_characters() -> dict:
	"""List all characters with their Mirror-verified canonical identity
	tags and self-written advice blocks."""
	return await client.get("/api/characters")


@mcp.tool
async def get_character(name: str) -> dict:
	"""One character's canonical tags, advice block, and full
	description."""
	return await client.get(f"/api/characters/{name}")


@mcp.tool
async def list_advice_runs() -> dict:
	"""List Mirror advice runs (run id, character, identity tags) —
	newest first. Cross-reference with vision_analyze using
	"advice/runs/<id>/<stage>_r<retry>.png" image paths."""
	return await client.get("/api/advice/runs")


if __name__ == "__main__":
	mcp.run()
