from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.ImagePreset import ImageGenerationType, ImageModel, ImagePreset

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator  # noqa: F401 (re-exported)

__all__ = ["GeneratedImage", "ImageGenerator", "NovelAIGenerator", "build_preset", "ImageGenerationType"]

LOGGER = logging.getLogger(__name__)


def build_preset(model: ImageModel, add_quality_tags: bool,
		size: Optional[tuple] = None, steps: Optional[int] = None) -> ImagePreset:
	"""Build the default preset for `model`.

	`add_quality_tags` controls the library's automatic quality-tag
	suffix (", location, very aesthetic, masterpiece, no text" on v4.5). It
	must be False when the prompt massager owns the quality tags, True to
	keep the historical (legacy) behavior. `size` is an optional
	(width, height) tuple — NovelAI's standard pairs are 832x1216
	(portrait, the default) and 1216x832 (landscape); lying poses need
	landscape.
	"""
	preset = ImagePreset.from_default_config(model)
	preset["quality_toggle"] = add_quality_tags
	if size is not None:
		preset["resolution"] = size  # library key: "resolution", not "size"
	if steps is not None:
		preset["steps"] = max(8, min(int(steps), 50))  # clamp: sane sampler range
	return preset


class NovelAIGenerator(ImageGenerator):
	"""NovelAI backend; explicitly implements the ImageGenerator interface
	(declaration-site conformance checking by mypy — the Python equivalent
	of Java's `implements`)."""
	"""NovelAI image generation behind a global lock with retry/backoff."""

	def __init__(self, api: NovelAIAPI, config: dict, lock: Optional[asyncio.Lock] = None):
		self._api = api
		self._config = config
		self._lock = lock or asyncio.Lock()
		self._model = ImageModel.Anime_v45_Full
		# Proactive pacing: generations are serialized by the lock, but this
		# additionally spaces their starts at least `generation_min_interval`
		# seconds apart so queued bursts don't fire back-to-back at NovelAI.
		self._min_interval = float(config.get("generation_min_interval", 1.0))
		self._last_started = 0.0

	def rules_overrides(self) -> dict:
		"""The core ruleset is already NovelAI-shaped; nothing to override."""
		return {}

	async def generate(
		self, prompt: str, negative_prompt: str, prompt_is_complete: bool,
		size: Optional[tuple] = None, steps: Optional[int] = None
	) -> Optional[GeneratedImage]:
		"""Generate an image, retrying while NovelAI reports being busy.

		Returns a GeneratedImage on success, or None when NovelAI remains
		busy after `generation_retries` attempts. See ImageGenerator for the
		`prompt_is_complete` contract (inverted into the library's
		quality_toggle here). `size` overrides the canvas (width, height); `steps` overrides
		the sampler step count (default 23; lower = faster, rougher).
		"""
		max_retries = int(self._config.get("generation_retries", 5))
		backoff = float(self._config.get("generation_backoff", 1.0))
		attempt = 0
		generated = False
		name = None
		data = None

		preset = build_preset(self._model, add_quality_tags=not prompt_is_complete,
			size=size, steps=steps)

		async with self._lock:
			pause = self._last_started + self._min_interval - time.monotonic()
			if pause > 0:
				await asyncio.sleep(pause)
			self._last_started = time.monotonic()
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

		if not generated or name is None or data is None:
			return None

		return GeneratedImage(name=name, data=data, seed=getattr(preset, "last_seed", None))
