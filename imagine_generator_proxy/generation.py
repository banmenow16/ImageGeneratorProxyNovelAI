from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.ImagePreset import ImageModel, ImagePreset

LOGGER = logging.getLogger(__name__)


@dataclass
class GeneratedImage:
	"""A single image returned by NovelAI."""

	name: str
	data: bytes
	seed: Optional[int] = None


def build_preset(model: ImageModel, library_quality_toggle: bool) -> ImagePreset:
	"""Build the default preset for `model`.

	`library_quality_toggle` controls the library's automatic quality-tag
	suffix (", location, very aesthetic, masterpiece, no text" on v4.5). It
	must be False when the prompt massager owns the quality tags, True to
	keep the historical (legacy) behavior.
	"""
	preset = ImagePreset.from_default_config(model)
	preset["quality_toggle"] = library_quality_toggle
	return preset


class NovelAIGenerator:
	"""NovelAI image generation behind a global lock with retry/backoff."""

	def __init__(self, api: NovelAIAPI, config: dict, lock: Optional[asyncio.Lock] = None):
		self._api = api
		self._config = config
		self._lock = lock or asyncio.Lock()
		self._model = ImageModel.Anime_v45_Full

	async def generate(
		self, prompt: str, negative_prompt: str, library_quality_toggle: bool
	) -> Optional[GeneratedImage]:
		"""Generate an image, retrying while NovelAI reports being busy.

		Returns a GeneratedImage on success, or None when NovelAI remains
		busy after `generation_retries` attempts.
		"""
		max_retries = int(self._config.get("generation_retries", 5))
		backoff = float(self._config.get("generation_backoff", 1.0))
		attempt = 0
		generated = False
		name = None
		data = None

		preset = build_preset(self._model, library_quality_toggle)

		async with self._lock:
			while attempt < max_retries:
				try:
					async for pair in self._api.high_level.generate_image(
						prompt, self._model, preset, negative_prompt=negative_prompt
					):
						try:
							name, data = pair
						except Exception:
							name = "image.png"
							data = pair

						generated = True
						break

					if generated:
						break

				except Exception as e:
					s = str(e)
					if "429" in s or "Concurrent generation" in s or "concurrent generation" in s.lower():
						attempt += 1
						LOGGER.warning("NovelAI busy (attempt %s/%s), retrying in %ss: %s", attempt, max_retries, backoff, s)
						await asyncio.sleep(backoff)
						backoff *= 2
						continue
					raise

		if not generated:
			return None

		return GeneratedImage(name=name, data=data, seed=getattr(preset, "last_seed", None))
