"""Direct generation: one image from a JSON request, no review chain.

POST /api/generate  {prompt, negative?, character?, engine?, save?,
size?: portrait|landscape|square}
→ {url, path?, massaged_prompt, negative_used, bytes}

This is the iteration endpoint for applications debugging prompts — the
full production path (GET /generate) keeps its vision-review chain and
cache; this one always runs fresh, applies the massager, and returns the
massaged prompt alongside the image so callers see exactly what the artist
received. Images are saved under data/debug/ and served at /debug-files/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import Backend
from imagine_generator_proxy.controllers.base import bad_request, json_body

LOGGER = logging.getLogger(__name__)

#: Named canvas presets for the size request option — NovelAI's standard
#: "Normal" resolutions. Landscape is required for lying poses; portrait
#: is the historical default.
SIZES = {
	"portrait": (832, 1216),
	"landscape": (1216, 832),
	"square": (1024, 1024),
}


class GenerationController:
	"""Endpoint for single-shot JSON generation without the review chain."""

	def __init__(self, backends: dict[str, Backend], default_engine: str,
			data_dir: Path) -> None:
		self._backends = backends
		self._default_engine = default_engine
		self._data_dir = data_dir
		(self._data_dir / "debug").mkdir(parents=True, exist_ok=True)

	def add_routes(self, router: web.UrlDispatcher) -> None:
		router.add_post("/api/generate", self.generate)
		router.add_static("/debug-files", str(self._data_dir / "debug"))

	# --- endpoint ---

	async def generate(self, request: web.Request) -> web.Response:
		body = await json_body(request)
		if body is None:
			return bad_request("request body must be a JSON object")
		prompt = str(body.get("prompt") or "").strip()
		if not prompt:
			return bad_request("prompt is required")

		engine = str(body.get("engine") or self._default_engine)
		backend: Optional[Backend] = self._backends.get(engine)
		if backend is None or getattr(backend.generator, "nsfw_blocked", False):
			return bad_request(f"engine {engine!r} unavailable",
				available=[k for k, v in self._backends.items()
					if not getattr(v.generator, "nsfw_blocked", False)])

		character = str(body.get("character") or "").strip()
		massaged = backend.massager.massage(prompt, character)
		if massaged.blocked_reason is not None:
			return web.json_response({"error": "blocked by safety filter",
				"reason": massaged.blocked_reason}, status=400)

		requested_negative = str(body.get("negative") or "").strip()
		negative = ", ".join(filter(None, [requested_negative,
			", ".join(massaged.negative_additions)
			if massaged.negative_additions else ""]))

		size = SIZES.get(str(body.get("size") or "portrait").strip().lower())
		if str(body.get("size") or "").strip() and size is None:
			return bad_request(f"unknown size {body.get('size')!r}",
				hint=f"one of: {', '.join(SIZES)}")
		image = await backend.generator.generate(massaged.effective, negative,
			prompt_is_complete=True, size=size)
		if image is None:
			return web.json_response(
				{"error": "generation failed", "engine": engine}, status=502)

		result = {
			"massaged_prompt": massaged.effective,
			"negative_used": negative,
			"engine": engine,
			"bytes": len(image.data),
		}
		if body.get("save", True):
			filename = self._save(bytes(image.data))
			result["path"] = f"debug/{filename}"
			result["url"] = f"/debug-files/{filename}"
		return web.json_response(result)

	# --- helpers ---

	def _save(self, image_bytes: bytes) -> str:
		debug_dir = self._data_dir / "debug"
		debug_dir.mkdir(parents=True, exist_ok=True)
		filename = time.strftime("%Y%m%d_%H%M%S") + ".png"
		(debug_dir / filename).write_bytes(image_bytes)
		return filename
