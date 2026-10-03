"""Civitai on-site generation backends.

CivitaiGenerator is a template-method parent (itself an ImageGenerator):
subclasses override the hook methods — model_version_id, generation_params,
rules_overrides — to describe their model. Rules provided here are merged
into the shared core ruleset by PromptMassager (safety/leak-cut/taste rules
stay universal; only quality-tag vocabulary and model quirks vary).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aiohttp import ClientSession

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator

LOGGER = logging.getLogger(__name__)

API_BASE = "https://civitai.com/api/v1"


class CivitaiGenerator(ImageGenerator):
	"""Parent for Civitai on-site generation; subclasses override hooks."""

	backend_name = "civitai"
	display_name = "Civitai model"

	def __init__(self, api_key: str, session: ClientSession,
			version_override: Optional[int] = None, timeout: float = 180.0):
		self._api_key = api_key
		self._session = session
		self._version_override = version_override
		self._timeout = timeout

	# --- hooks for subclasses ---

	def model_version_id(self) -> int:
		"""Civitai modelVersionId to generate with."""
		raise NotImplementedError

	def generation_params(self) -> dict:
		"""Sampler/step/cfg/resolution params for the Civitai API."""
		return {
			"scheduler": "EulerAncestralDiscrete",
			"steps": 28,
			"cfgScale": 6.0,
			"width": 832,
			"height": 1216,
			"nsfw": True,
		}

	def rules_overrides(self) -> dict:
		"""Per-model rule overrides merged into the shared core ruleset:
		{"remove": [rule_ids], "replace": {id: rule}, "add": [rules]}.
		The universal rules (safety, leak-cut, dedupe, taste calibration)
		are never overridden here."""
		return {}

	# --- template ---

	def _resolved_version_id(self) -> int:
		return int(self._version_override or self.model_version_id())

	async def generate(self, prompt: str, negative_prompt: str,
			prompt_is_complete: bool) -> Optional[GeneratedImage]:
		# Civitai never auto-decorates prompts, so prompt_is_complete is
		# respected by construction.
		body = {
			"modelVersionId": self._resolved_version_id(),
			"prompt": prompt,
			"negativePrompt": negative_prompt,
			"params": self.generation_params(),
		}
		for attempt in range(3):
			try:
				return await self._generate_once(body)
			except _BusyError as e:
				LOGGER.warning("Civitai busy for %s (attempt %d): %s",
					self.display_name, attempt + 1, e)
				await asyncio.sleep(2.0 * (attempt + 1))
			except Exception:
				LOGGER.exception("Civitai generation failed for %s", self.display_name)
				return None
		return None

	async def _generate_once(self, body: dict) -> GeneratedImage:
		async with self._session.post(
			f"{API_BASE}/images/generate",
			json=body,
			headers={"Authorization": f"Bearer {self._api_key}"},
			timeout=self._timeout,
		) as resp:
			text = await resp.text()
			if resp.status in (429, 500, 502, 503):
				raise _BusyError(f"HTTP {resp.status}: {text[:200]}")
			if resp.status != 200:
				raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
			import json
			data = json.loads(text)

		images = data.get("images") or []
		if not images:
			raise RuntimeError(f"no images in response: {text[:300]}")
		first = images[0]
		url = first.get("url")
		if not url:
			raise RuntimeError(f"image entry without url: {str(first)[:200]}")

		async with self._session.get(url, timeout=self._timeout) as dl:
			if dl.status != 200:
				raise RuntimeError(f"image download HTTP {dl.status}")
			image_bytes = await dl.read()
		return GeneratedImage(name="civitai.png", data=image_bytes,
			seed=first.get("seed"))


class _BusyError(Exception):
	"""Transient upstream condition worth retrying."""


class IllustriousGenerator(CivitaiGenerator):
	"""Illustrious-XL v0.1 — best bang for buck: strong defaults, forgiving."""

	backend_name = "illustrious"
	display_name = "Illustrious-XL"

	def model_version_id(self) -> int:
		return 889818  # Illustrious-XL v0.1

	def rules_overrides(self) -> dict:
		return {
			"remove": ["ensure-v45-suffix"],  # NAI-specific
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

	def model_version_id(self) -> int:
		return 290640  # V6 (start with this one)

	def rules_overrides(self) -> dict:
		return {
			"remove": ["ensure-v45-suffix"],
			"replace": {"ensure-quality-prefix": {
				"id": "ensure-quality-prefix", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "ensure_prepend",
					"value": "score_9, score_8_up, score_7_up, source_anime"},
				"notes": "Pony native quality syntax"}},
			"add": [{  # Pony wants its LOW quality scores negated
				"id": "pony-negate-low-scores", "stage": "enhance", "severity": "safe",
				"match": {"type": "always"},
				"action": {"type": "negate", "value": "score_6, score_5, score_4"},
				"notes": "Pony reacts strongly to negated low quality scores"}],
		}


class NoobAIGenerator(CivitaiGenerator):
	"""NoobAI-XL epsilon-pred 1.1 — top quality; v-pred 1.0 via config."""

	backend_name = "noobai"
	display_name = "NoobAI-XL"

	def model_version_id(self) -> int:
		return 1116447  # Epsilon-pred 1.1 (forgiving); v-pred 1.0 is 1190596

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
