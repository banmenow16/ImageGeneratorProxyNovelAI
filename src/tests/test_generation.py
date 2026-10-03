"""Unit tests for the NovelAI generation wrapper (stubbed API, no network)."""

from __future__ import annotations

import asyncio

from novelai_api.ImagePreset import ImageModel

from imagine_generator_proxy.generation import NovelAIGenerator, build_preset


class _StubHighLevel:
	"""Each generate_image() call consumes exactly one queued outcome."""

	def __init__(self, outcomes):
		self._outcomes = list(outcomes)

	async def generate_image(self, prompt, model, preset, negative_prompt=None):
		if not self._outcomes:
			raise AssertionError("stub exhausted")
		outcome = self._outcomes.pop(0)
		if isinstance(outcome, Exception):
			raise outcome
		yield outcome


class _StubAPI:
	def __init__(self, outcomes):
		self.high_level = _StubHighLevel(outcomes)


BUSY = Exception("HTTP 429 Concurrent generation limit reached")


def _run(coro):
	return asyncio.run(coro)


def test_retry_then_success():
	api = _StubAPI([BUSY, BUSY, ("image_0.png", b"fake-png-bytes")])
	gen = NovelAIGenerator(api, {"generation_retries": 5, "generation_backoff": 0.0})
	image = _run(gen.generate("prompt", "negative", prompt_is_complete=True))
	assert image is not None
	assert image.name == "image_0.png"
	assert image.data == b"fake-png-bytes"


def test_busy_exhaustion_returns_none():
	api = _StubAPI([BUSY, BUSY])
	gen = NovelAIGenerator(api, {"generation_retries": 2, "generation_backoff": 0.0})
	assert _run(gen.generate("prompt", "negative", prompt_is_complete=True)) is None


def test_non_busy_exception_propagates():
	api = _StubAPI([Exception("connection reset")])
	gen = NovelAIGenerator(api, {"generation_retries": 3, "generation_backoff": 0.0})
	try:
		_run(gen.generate("prompt", "negative", prompt_is_complete=True))
		raise AssertionError("expected exception")
	except Exception as e:
		assert "connection reset" in str(e)


def test_build_preset_honors_quality_toggle_flag():
	for flag in (True, False):
		preset = build_preset(ImageModel.Anime_v45_Full, flag)
		assert preset["quality_toggle"] is flag


def test_generate_uses_global_lock_serially():
	api = _StubAPI([("image_0.png", b"a"), ("image_0.png", b"b")])
	lock = asyncio.Lock()
	gen = NovelAIGenerator(api, {"generation_retries": 1, "generation_backoff": 0.0}, lock=lock)
	assert gen._lock is lock


def test_min_interval_paces_sequential_generations():
	import time

	api = _StubAPI([("image_0.png", b"a"), ("image_0.png", b"b")])
	gen = NovelAIGenerator(api, {"generation_retries": 1, "generation_backoff": 0.0,
		"generation_min_interval": 0.3})
	start = time.monotonic()
	first = _run(gen.generate("p", "n", prompt_is_complete=True))
	mid = time.monotonic()
	second = _run(gen.generate("p", "n", prompt_is_complete=True))
	end = time.monotonic()
	assert first is not None and second is not None
	# first call starts immediately (no prior generation); second waits for the interval
	assert mid - start < 0.3
	assert end - mid >= 0.15  # allowed small scheduling slack below the nominal 0.15
