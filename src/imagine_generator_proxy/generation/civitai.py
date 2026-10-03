"""Civitai Orchestration API generation backends.

Transport: POST https://orchestration.civitai.com/v2/consumer/workflows
submitting an imageGen step (SDXL ecosystem, model addressed by AIR URN),
long-polling until succeeded, then downloading the signed image URL.
Cost at 832x1216/28 steps is ~3 Buzz per image (whatif-verified).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from aiohttp import ClientSession

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator

LOGGER = logging.getLogger(__name__)

API_BASE = "https://orchestration.civitai.com/v2/consumer/workflows"


class CivitaiGenerator(ImageGenerator):
	"""Template-method parent for Civitai Orchestration generation."""

	backend_name = "civitai"
	display_name = "Civitai model"

	def __init__(self, api_key: str, session: ClientSession,
			urn_override: Optional[str] = None, timeout: float = 240.0,
			lora_overrides: Optional[dict] = None):
		self._api_key = api_key
		self._session = session
		self._urn_override = urn_override
		self._timeout = timeout
		self._lora_overrides = lora_overrides or {}

	# --- hooks for subclasses ---

	def air_urn(self) -> str:
		"""AIR URN addressing the checkpoint: urn:air:sdxl:checkpoint:civitai:<modelId>@<versionId>."""
		raise NotImplementedError

	def generation_params(self) -> dict:
		return {"steps": 28, "cfgScale": 6.0, "width": 832, "height": 1216,
			"sampleMethod": "euler", "schedule": "karras"}

	def rules_overrides(self) -> dict:
		return {}

	def loras(self) -> dict:
		"""AIR URN -> strength map applied to every generation; config
		lora_overrides replace these wholesale when set."""
		return {}

	# --- template ---

	def _urn(self) -> str:
		return str(self._urn_override or self.air_urn())

	def _input(self, prompt: str, negative_prompt: str) -> dict:
		params = self.generation_params()
		data = {
			"engine": "sdcpp", "ecosystem": "sdxl", "operation": "createImage",
			"model": self._urn(),
			"prompt": prompt, "negativePrompt": negative_prompt,
			"sampleMethod": params.get("sampleMethod", "euler"),
			"schedule": params.get("schedule", "karras"),
			"steps": int(params.get("steps", 28)),
			"cfgScale": float(params.get("cfgScale", 6.0)),
			"width": int(params.get("width", 832)),
			"height": int(params.get("height", 1216)),
			"quantity": 1,
		}
		loras = self._lora_overrides or self.loras()
		if loras:
			data["loras"] = dict(loras)
		return data

	async def generate(self, prompt: str, negative_prompt: str,
			prompt_is_complete: bool) -> Optional[GeneratedImage]:
		body = {"steps": [{"$type": "imageGen", "input": self._input(prompt, negative_prompt)}]}
		try:
			workflow = await self._post_workflow(body)
			workflow = await self._await_terminal(workflow)
			placeholder = self._image_from(workflow)
			return await self._download(placeholder)
		except _BusyError as e:
			LOGGER.warning("Civitai busy for %s: %s", self.display_name, e)
			return None
		except Exception:
			LOGGER.exception("Civitai generation failed for %s", self.display_name)
			return None

	async def _post_workflow(self, body: dict) -> dict:
		async with self._session.post(
			f"{API_BASE}?wait=60", json=body,
			headers={"Authorization": f"Bearer {self._api_key}"},
			timeout=self._timeout,
		) as resp:
			text = await resp.text()
			if resp.status in (429, 500, 502, 503):
				raise _BusyError(f"HTTP {resp.status}: {text[:200]}")
			if resp.status != 200:
				raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
			return _json(text)

	async def _await_terminal(self, workflow: dict) -> dict:
		deadline = asyncio.get_event_loop().time() + self._timeout
		while workflow.get("status") not in ("succeeded", "failed", "cancelled"):
			if asyncio.get_event_loop().time() > deadline:
				raise RuntimeError("orchestration workflow timed out")
			wf_id = workflow.get("id")
			if not wf_id:
				raise RuntimeError(f"workflow without id: {str(workflow)[:200]}")
			await asyncio.sleep(3.0)
			async with self._session.get(
				f"{API_BASE}/{wf_id}?wait=60",
				headers={"Authorization": f"Bearer {self._api_key}"},
				timeout=self._timeout,
			) as resp:
				if resp.status != 200:
					raise RuntimeError(f"poll HTTP {resp.status}: {(await resp.text())[:200]}")
				workflow = _json(await resp.text())
		if workflow.get("status") != "succeeded":
			raise RuntimeError(f"workflow {workflow.get('status')}: {str(workflow)[:300]}")
		return workflow

	def _image_from(self, workflow: dict) -> GeneratedImage:
		steps = workflow.get("steps") or []
		if not steps:
			raise RuntimeError("workflow has no steps")
		output = steps[0].get("output") or {}
		images = output.get("images") or []
		if not images or not images[0].get("url"):
			raise RuntimeError(f"no image in output: {str(output)[:200]}")
		return GeneratedImage(name=images[0]["url"], data=b"", seed=images[0].get("seed"))

	async def _download(self, image: GeneratedImage) -> GeneratedImage:
		async with self._session.get(image.name, timeout=self._timeout) as dl:
			if dl.status != 200:
				raise RuntimeError(f"image download HTTP {dl.status}")
			return GeneratedImage(name="civitai.png", data=await dl.read(), seed=image.seed)


def _json(text: str) -> dict:
	import json
	return json.loads(text)


class _BusyError(Exception):
	"""Transient upstream condition worth surfacing as a retry-able failure."""


class IllustriousGenerator(CivitaiGenerator):
	"""Illustrious-XL v0.1 — best bang for buck."""

	backend_name = "illustrious"
	display_name = "Illustrious-XL"

	def air_urn(self) -> str:
		return "urn:air:sdxl:checkpoint:civitai:795765@889818"

	def loras(self) -> dict:
		# Breast Size Slider (Pony/Illustrious) — 100k+ downloads, the
		# ecosystem favorite for exactly this tag class
		return {"urn:air:illustrious:lora:civitai:481016@1835318": 0.7}

	def rules_overrides(self) -> dict:
		return {
			"remove": ["ensure-v45-suffix"],
			"replace": {"ensure-quality-prefix": {
				"id": "ensure-quality-prefix", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "ensure_prepend",
					"value": "masterpiece, best quality, very aesthetic, absurdres"},
				"notes": "Illustrious native quality tags"}},
		}


class PonyV6Generator(CivitaiGenerator):
	"""Pony Diffusion V6 XL — easiest prompting, huge NSFW LoRA ecosystem."""

	backend_name = "pony"
	display_name = "Pony Diffusion V6 XL"

	def air_urn(self) -> str:
		return "urn:air:sdxl:checkpoint:civitai:257749@290640"

	def loras(self) -> dict:
		# Pony-native Breast Size Slider
		return {"urn:air:pony:lora:civitai:1121948@1260971": 0.7}

	def rules_overrides(self) -> dict:
		return {
			"remove": ["ensure-v45-suffix"],
			"replace": {"ensure-quality-prefix": {
				"id": "ensure-quality-prefix", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "ensure_prepend",
					"value": "score_9, score_8_up, score_7_up, source_anime"},
				"notes": "Pony native quality syntax"}},
			"add": [{
				"id": "pony-negate-low-scores", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "negate", "value": "score_6, score_5, score_4"},
				"notes": "Pony reacts strongly to negated low quality scores"}],
		}


class NoobAIGenerator(CivitaiGenerator):
	"""NoobAI-XL epsilon-pred 1.1 — top quality."""

	backend_name = "noobai"
	display_name = "NoobAI-XL"

	def air_urn(self) -> str:
		return "urn:air:sdxl:checkpoint:civitai:833294@1116447"

	def loras(self) -> dict:
		# NoobAI is Illustrious-trained — the Illustrious slider applies
		return {"urn:air:illustrious:lora:civitai:481016@1835318": 0.7}

	def generation_params(self) -> dict:
		params = super().generation_params()
		params.update({"steps": 30, "cfgScale": 5.0})
		return params

	def rules_overrides(self) -> dict:
		return {
			"remove": ["ensure-v45-suffix"],
			"replace": {"ensure-quality-prefix": {
				"id": "ensure-quality-prefix", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "ensure_prepend",
					"value": "masterpiece, best quality, amazing quality, very aesthetic, absurdres"},
				"notes": "NoobAI native quality tags"}},
		}
