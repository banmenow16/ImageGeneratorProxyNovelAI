"""OpenRouter video backends (image-to-video via /api/v1/videos).

Async jobs: POST /api/v1/videos with frame_images[0] as a data URL
(first_frame), poll GET /api/v1/videos/{id} until completed, download
GET /api/v1/videos/{id}/content?index=0 with the Authorization header.
Pricing is per video-second (e.g. wan-2.7 $0.10/s); NSFW tolerance is
provider-dependent and marked per model until verified empirically.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

from aiohttp import ClientSession

from imagine_generator_proxy.generation.video_base import (
	GeneratedVideo,
	VideoGenerator,
	VideoRequest,
)

LOGGER = logging.getLogger(__name__)

API = "https://openrouter.ai/api/v1/videos"


class OpenRouterVideoGenerator(VideoGenerator):
	"""Template-method parent; subclasses set model id and availability."""

	backend_name = "or-video"
	display_name = "OpenRouter video"
	# gates listing on the video page: NSFW tolerance unknown until tested
	nsfw_verified = False

	def __init__(self, api_key: str, session: ClientSession, timeout: float = 900.0):
		self.last_error = ""
		self._api_key = api_key
		self._session = session
		self._timeout = timeout

	def model_id(self) -> str:
		raise NotImplementedError

	def extra_payload(self) -> dict:
		return {}

	def _headers(self) -> dict:
		return {"Authorization": f"Bearer {self._api_key}"}

	async def generate_video(self, request: VideoRequest) -> Optional[GeneratedVideo]:
		try:
			data_url = "data:image/png;base64," + base64.b64encode(
				request.source_image.read_bytes()).decode("ascii")
			body = {
				"model": self.model_id(),
				"prompt": request.prompt or "gentle natural motion, subtle breathing",
				"frame_images": [{
					"type": "image_url",
					"image_url": {"url": data_url},
					"frame_type": "first_frame",
				}],
				**self.extra_payload(),
			}
			job = await self._submit(body)
			job = await self._await_completed(job)
			content_url = f"{API}/{job['id']}/content?index=0"
			async with self._session.get(content_url, headers=self._headers(),
				timeout=self._timeout) as dl:
				if dl.status != 200:
					raise RuntimeError(f"video download HTTP {dl.status}: {(await dl.text())[:200]}")
				return GeneratedVideo(name="video.mp4", data=await dl.read(),
					content_type="video/mp4")
		except Exception as e:
			LOGGER.exception("OpenRouter video generation failed for %s", self.display_name)
			self.last_error = str(e)[:200]
			return None

	async def _submit(self, body: dict) -> dict:
		import json
		async with self._session.post(API, json=body, headers={
			**self._headers(), "Content-Type": "application/json"},
			timeout=self._timeout) as resp:
			text = await resp.text()
			if resp.status not in (200, 201, 202):
				raise RuntimeError(f"video submit HTTP {resp.status}: {text[:300]}")
			return json.loads(text)

	async def _await_completed(self, job: dict) -> dict:
		import json
		job_id = job.get("id")
		if not job_id:
			raise RuntimeError(f"submit response without id: {str(job)[:200]}")
		poll_url = job.get("polling_url") or f"{API}/{job_id}"
		deadline = asyncio.get_event_loop().time() + self._timeout
		while job.get("status") not in ("completed", "failed", "cancelled", "expired"):
			if asyncio.get_event_loop().time() > deadline:
				raise RuntimeError("video job timed out")
			await asyncio.sleep(5.0)
			async with self._session.get(poll_url, headers=self._headers(),
				timeout=self._timeout) as resp:
				if resp.status != 200:
					raise RuntimeError(f"poll HTTP {resp.status}: {(await resp.text())[:200]}")
				job = json.loads(await resp.text())
		if job.get("status") != "completed":
			raise RuntimeError(f"video job {job.get('status')}: {str(job.get('error', ''))[:200]}")
		return job


class Wan27Video(OpenRouterVideoGenerator):
	"""Wan 2.7 — $0.10/s, i2v in the official examples."""

	backend_name = "or-wan27"
	display_name = "Wan 2.7 (OpenRouter, $0.10/s)"

	def model_id(self) -> str:
		return "alibaba/wan-2.7"


class Wan26VideoOR(OpenRouterVideoGenerator):
	"""Wan 2.6 — explicit image_to_video SKUs ($0.10/s @720p)."""

	backend_name = "or-wan26"
	display_name = "Wan 2.6 (OpenRouter, i2v)"

	def model_id(self) -> str:
		return "alibaba/wan-2.6"

	def extra_payload(self) -> dict:
		return {"resolution": "720p"}


class Wan30Video(OpenRouterVideoGenerator):
	"""Wan 3.0 — cheapest ($0.05/s @480p)."""

	backend_name = "or-wan30"
	display_name = "Wan 3.0 (OpenRouter, $0.05/s)"

	def model_id(self) -> str:
		return "alibaba/wan-3.0"

	def extra_payload(self) -> dict:
		return {"resolution": "480p"}
