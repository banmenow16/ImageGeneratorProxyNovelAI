"""Vision analysis: ask a vision model about one or more stored images.

POST /api/vision/analyze  {images: [...], prompt, model?, temperature?,
max_tokens?} → {text, model, images_used, images_unreadable}

Images are referenced by path relative to the data directory (e.g.
"advice/runs/16/nude_r1.png") or as base64 payloads. Up to four per call.
Designed for external applications debugging their own generation output
against any vision question.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

from aiohttp import web
from openai import AsyncOpenAI

from imagine_generator_proxy.controllers.base import bad_request, json_body

LOGGER = logging.getLogger(__name__)

VISION_MODEL_DEFAULT = "qwen/qwen3-vl-32b-instruct"
MAX_IMAGES_PER_CALL = 4
MIN_BASE64_LENGTH = 256  # below this a "payload" is probably a path


class VisionController:
	"""Endpoints for arbitrary vision-model questions about stored images."""

	def __init__(self, config: dict, data_dir: Path) -> None:
		self._config = config
		self._data_dir = data_dir

	def add_routes(self, router: web.UrlDispatcher) -> None:
		router.add_post("/api/vision/analyze", self.analyze)

	# --- endpoint ---

	async def analyze(self, request: web.Request) -> web.Response:
		body = await json_body(request)
		if body is None:
			return bad_request("request body must be a JSON object")
		prompt = str(body.get("prompt") or "").strip()
		references = body.get("images") or []
		if not prompt:
			return bad_request("prompt is required")
		if not isinstance(references, list) or not references:
			return bad_request("images is required (list of paths or base64)")
		if len(references) > MAX_IMAGES_PER_CALL:
			return bad_request(f"at most {MAX_IMAGES_PER_CALL} images per call")

		api_key = self._config.get("openRouterToken")
		if not api_key:
			return bad_request("server has no openRouterToken configured")

		payloads = []
		for reference in references:
			payload = self._resolve_image(str(reference or ""))
			if payload:
				payloads.append(payload)
		if not payloads:
			return bad_request("no readable images",
				hint="paths are relative to the data directory,"
					" e.g. advice/runs/16/nude_r1.png")

		model = str(body.get("model") or self._config.get("compliance_model")
			or VISION_MODEL_DEFAULT)
		try:
			temperature = float(body.get("temperature", 0.0))
			max_tokens = int(body.get("max_tokens", 300))
		except (TypeError, ValueError):
			return bad_request("temperature and max_tokens must be numbers")

		client = AsyncOpenAI(api_key=api_key,
			base_url="https://openrouter.ai/api/v1", timeout=90.0, max_retries=1)
		content: list = [{"type": "text", "text": prompt}]
		for payload in payloads:
			content.append({"type": "image_url",
				"image_url": {"url": f"data:image/png;base64,{payload}"}})
		try:
			response = await client.chat.completions.create(model=model,
				max_tokens=max_tokens, temperature=temperature,
				messages=[{"role": "user", "content": content}])
		except Exception as ex:  # noqa: BLE001 — surfaced to the caller
			LOGGER.exception("vision analyze failed")
			return web.json_response({"error": f"vision call failed: {ex}",
				"model": model}, status=502)
		return web.json_response({
			"text": response.choices[0].message.content or "",
			"model": model,
			"images_used": len(payloads),
			"images_unreadable": len(references) - len(payloads),
		})

	# --- helpers ---

	def _resolve_image(self, reference: str) -> Optional[str]:
		"""An image reference → base64 payload, or None when unreadable.

		Accepts data: URLs, bare base64 payloads, and paths relative to the
		data directory. Paths are confined to the data directory — this
		endpoint analyzes generated images, not arbitrary local files.
		"""
		reference = reference.strip()
		if not reference:
			return None
		if reference.startswith("data:"):
			return reference.split(",", 1)[1] if "," in reference else None
		if (len(reference) >= MIN_BASE64_LENGTH
				and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", reference)):
			return reference
		candidate = (self._data_dir / reference).resolve()
		if not candidate.is_relative_to(self._data_dir.resolve()):
			return None
		if not candidate.is_file():
			return None
		return base64.b64encode(candidate.read_bytes()).decode("ascii")
