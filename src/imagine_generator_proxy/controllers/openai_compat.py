"""OpenAI Images API compatibility: /v1/images/generations.

Lets OpenAI-protocol clients (the camgirl sim, SillyTavern, custom apps)
use this proxy as a swappable backend — point them at
<base>/v1/images/generations and they can be re-pointed at any other
OpenAI-compatible image server without code changes.

Accepted body (OpenAI shape plus harmless extensions):
- model: any string; matched against backend names, else the default
  engine serves the request (OpenAI clients send model names freely)
- prompt (required), negative_prompt (extension — OpenAI has no negative
  field, but every image client that supports one sends this name)
- n: number of images, capped at 4
- size: "WxH" (e.g. "832x1216") or portrait|landscape|square
- response_format: "url" (default) or "b64_json"
- character (extension): massager context, as on /api/generate

Response is the standard {created, data:[{url|b64_json}]} envelope.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import Backend
from imagine_generator_proxy.controllers.base import bad_request, json_body

LOGGER = logging.getLogger(__name__)

NAMED_SIZES = {"portrait": (832, 1216), "landscape": (1216, 832),
	"square": (1024, 1024)}
MAX_IMAGES = 4
_SIZE_RE = re.compile(r"^(\d{3,4})x(\d{3,4})$")


class OpenAICompatController:
	"""OpenAI Images-protocol adapter over the generation backends."""

	def __init__(self, backends: dict[str, Backend], default_engine: str,
			data_dir: Path) -> None:
		self._backends = backends
		self._default_engine = default_engine
		self._data_dir = data_dir
		(self._data_dir / "compat").mkdir(parents=True, exist_ok=True)

	def add_routes(self, router: web.UrlDispatcher) -> None:
		router.add_post("/v1/images/generations", self.generations)
		router.add_get("/v1/models", self.models)
		router.add_static("/compat-files", str(self._data_dir / "compat"))

	# --- endpoints ---

	async def models(self, request: web.Request) -> web.Response:
		"""Minimal /v1/models so clients can discovery-check the endpoint."""
		return web.json_response({"object": "list", "data": [
			{"id": name, "object": "model"}
			for name in self._backends
			if not getattr(self._backends[name].generator, "nsfw_blocked", False)
		] + [{"id": "default", "object": "model"}]})

	async def generations(self, request: web.Request) -> web.Response:
		body = await json_body(request)
		if body is None:
			return bad_request("request body must be a JSON object")
		prompt = str(body.get("prompt") or "").strip()
		if not prompt:
			return bad_request("prompt is required")

		engine = self._engine_for(str(body.get("model") or ""))
		backend: Optional[Backend] = self._backends.get(engine)
		if backend is None or getattr(backend.generator, "nsfw_blocked", False):
			return bad_request(f"engine {engine!r} unavailable",
				available=[k for k, v in self._backends.items()
					if not getattr(v.generator, "nsfw_blocked", False)])

		try:
			n = max(1, min(int(body.get("n", 1)), MAX_IMAGES))
		except (TypeError, ValueError):
			return bad_request("n must be a number")
		size = self._size_for(body.get("size"))
		if size == "invalid":
			return bad_request(f"unknown size {body.get('size')!r}",
				hint='"WxH" (e.g. 832x1216) or portrait|landscape|square')
		response_format = str(body.get("response_format") or "url")
		if response_format not in ("url", "b64_json"):
			return bad_request("response_format must be url or b64_json")

		character = str(body.get("character") or "").strip()
		negative_requested = str(body.get("negative_prompt")
			or body.get("negative") or "").strip()
		massaged = backend.massager.massage(prompt, character)
		if massaged.blocked_reason is not None:
			return web.json_response({"error": {"message":
				f"blocked by safety filter: {massaged.blocked_reason}",
				"type": "invalid_request_error"}}, status=400)
		negative = ", ".join(filter(None, [negative_requested,
			", ".join(massaged.negative_additions)
			if massaged.negative_additions else ""]))

		data: list[dict] = []
		for _ in range(n):
			image = await backend.generator.generate(massaged.effective,
				negative, prompt_is_complete=True, size=size)
			if image is None:
				return web.json_response({"error": {"message":
					f"generation failed on engine {engine!r}",
					"type": "server_error"}}, status=502)
			if response_format == "b64_json":
				data.append({"b64_json": base64.b64encode(
					bytes(image.data)).decode("ascii")})
			else:
				filename = self._save(bytes(image.data))
				url = (f"{request.scheme}://{request.host}"
					f"/compat-files/{filename}")
				data.append({"url": url})
		return web.json_response({"created": int(time.time()), "data": data})

	# --- helpers ---

	def _engine_for(self, model: str) -> str:
		"""OpenAI clients send free-form model names — match a backend if
		possible, otherwise serve on the default engine."""
		if model in self._backends:
			return model
		return self._default_engine

	def _size_for(self, size: object) -> Optional[tuple]:
		"""'WxH' or named size -> (w, h); None for the backend default;
		"invalid" sentinel for an unparseable request."""
		if size is None or str(size).strip() == "":
			return None
		text = str(size).strip().lower()
		if text in NAMED_SIZES:
			return NAMED_SIZES[text]
		match = _SIZE_RE.match(text)
		if match:
			width, height = int(match.group(1)), int(match.group(2))
			if width % 64 or height % 64:
				return "invalid"  # type: ignore[return-value]
			return (width, height)
		return "invalid"  # type: ignore[return-value]

	def _save(self, image_bytes: bytes) -> str:
		compat_dir = self._data_dir / "compat"
		compat_dir.mkdir(parents=True, exist_ok=True)
		filename = time.strftime("%Y%m%d_%H%M%S") + f"_{len(image_bytes)}.png"
		(compat_dir / filename).write_bytes(image_bytes)
		return filename
