"""Civitai Orchestration video backends (image-to-video).

Transport: upload the source image via POST /v2/consumer/blobs (multipart),
then POST /v2/consumer/recipes/videoGen with allowMatureContent=true,
long-poll the workflow, download the signed video URL. Output contract
(whatif-verified): steps[0].output.video = {id, available, url}.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aiohttp import ClientSession

from imagine_generator_proxy.generation.video_base import (
	GeneratedVideo,
	VideoGenerator,
	VideoRequest,
)

LOGGER = logging.getLogger(__name__)

WF_BASE = "https://orchestration.civitai.com/v2/consumer/workflows"
RECIPE = "https://orchestration.civitai.com/v2/consumer/recipes/videoGen"


class CivitaiVideoGenerator(VideoGenerator):
	"""Template-method parent; subclasses override engine_payload()."""

	backend_name = "civitai-video"
	display_name = "Civitai video"

	def __init__(self, api_key: str, session: ClientSession, timeout: float = 600.0):
		self.last_error = ""
		self._api_key = api_key
		self._session = session
		self._timeout = timeout

	def engine_payload(self) -> dict:
		"""Engine/version discriminator fields (e.g. engine=wan, version=v2.2)."""
		raise NotImplementedError

	def _headers(self) -> dict:
		return {"Authorization": f"Bearer {self._api_key}"}

	async def generate_video(self, request: VideoRequest) -> Optional[GeneratedVideo]:
		try:
			image_url = await self._upload_image(request.source_image)
			if not image_url:
				return None
			body = {
				**self.engine_payload(),
				"prompt": request.prompt or "gentle natural motion, subtle breathing",
				"negativePrompt": request.negative_prompt or None,
				"duration": int(request.duration_seconds),
				"width": int(request.width), "height": int(request.height),
				"startImage": image_url,
			}
			body = {k: v for k, v in body.items() if v is not None}
			result = await self._submit(body)
			# the recipe responds synchronously when the video completes within
			# the wait window: {"progress": 1, "video": {..., "available": true, "url"}}
			video_ref = result.get("video") or {}
			if not video_ref.get("url"):
				workflow = await self._await_terminal(result)
				video_ref = ((workflow.get("steps") or [{}])[0].get("output") or {}).get("video") or {}
			url = video_ref.get("url")
			if not url:
				raise RuntimeError(f"no video in output: {str(video_ref)[:200]}")
			async with self._session.get(url, timeout=self._timeout) as dl:
				if dl.status != 200:
					raise RuntimeError(f"video download HTTP {dl.status}")
				return GeneratedVideo(name="video.mp4", data=await dl.read(),
					content_type="video/mp4")
		except Exception as e:
			LOGGER.exception("Civitai video generation failed for %s", self.display_name)
			self.last_error = str(e)[:200]
			return None

	async def _upload_image(self, path) -> Optional[str]:
		"""startImage accepts 'A URL, A DataURL or a Base64 string' — pass the
		local image as raw base64 (no upload round-trip, no public URL needed)."""
		import base64
		return base64.b64encode(path.read_bytes()).decode("ascii")

	async def _submit(self, body: dict) -> dict:
		async with self._session.post(
			f"{RECIPE}?wait=60&allowMatureContent=true", json=body,
			headers=self._headers(), timeout=self._timeout,
		) as resp:
			text = await resp.text()
			if resp.status != 200:
				raise RuntimeError(f"videoGen HTTP {resp.status}: {text[:300]}")
			import json
			LOGGER.info("videoGen submit response: %s", text[:400])
			return json.loads(text)

	async def _await_terminal(self, workflow: dict) -> dict:
		import json
		deadline = asyncio.get_event_loop().time() + self._timeout
		while workflow.get("status") not in ("succeeded", "failed", "cancelled"):
			if asyncio.get_event_loop().time() > deadline:
				raise RuntimeError("video workflow timed out")
			wf_id = workflow.get("id")
			await asyncio.sleep(5.0)
			async with self._session.get(f"{WF_BASE}/{wf_id}?wait=60",
				headers=self._headers(), timeout=self._timeout) as resp:
				if resp.status != 200:
					raise RuntimeError(f"poll HTTP {resp.status}")
				workflow = json.loads(await resp.text())
		if workflow.get("status") != "succeeded":
			raise RuntimeError(f"video workflow {workflow.get('status')}: {str(workflow)[:300]}")
		return workflow


class Wan22Video(CivitaiVideoGenerator):
	"""Wan 2.2 via Civitai infra — bang for buck default, NSFW-capable."""

	backend_name = "wan22"
	display_name = "Wan 2.2 (Civitai)"
	# NSFW Buzz on Civitai requires crypto top-up (user won't); disabled from
	# the page — flip to re-enable. Code stays fully intact.
	nsfw_blocked = True

	def engine_payload(self) -> dict:
		return {"engine": "wan", "version": "v2.2"}


class Wan26Video(CivitaiVideoGenerator):
	"""Wan 2.6 — higher quality tier."""

	backend_name = "wan26"
	display_name = "Wan 2.6 (Civitai)"
	nsfw_blocked = True  # see Wan22Video note

	def engine_payload(self) -> dict:
		return {"engine": "wan", "version": "v2.6"}


class Ltx2Video(CivitaiVideoGenerator):
	"""LTX-2 via ComfyUI on Civitai infra — fast/cheap alternative."""

	backend_name = "ltx2"
	display_name = "LTX-2 (Civitai)"
	nsfw_blocked = True  # see Wan22Video note

	def engine_payload(self) -> dict:
		return {"engine": "ltx2", "operation": "createVideo"}
