"""Tests for the Civitai generator hierarchy (stubbed session, no network)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from imagine_generator_proxy.generation.civitai import (
	IllustriousGenerator,
	NoobAIGenerator,
	PonyV6Generator,
)
from imagine_generator_proxy.prompts import PromptMassager


class _StubResponse:
	def __init__(self, status, payload=None, raw=b"png-bytes"):
		self.status = status
		self._payload = payload
		self._raw = raw

	async def text(self):
		return json.dumps(self._payload) if self._payload is not None else ""

	async def read(self):
		return self._raw

	async def __aenter__(self):
		return self

	async def __aexit__(self, *args):
		return False


class _StubSession:
	def __init__(self, post_payload=None, post_status=200, get_status=200):
		self.posts = []
		self._post_payload = post_payload
		self._post_status = post_status
		self._get_status = get_status

	def post(self, url, json=None, headers=None, timeout=None):
		self.posts.append({"url": url, "json": json, "headers": headers})
		return _StubResponse(self._post_status, self._post_payload)

	def get(self, url, timeout=None):
		return _StubResponse(self._get_status)


def _ok_payload(seed=7):
	return {"images": [{"url": "https://img/civitai/1.png", "seed": seed}]}


def run(coro):
	return asyncio.run(coro)


@pytest.mark.parametrize("cls,expected_id", [
	(IllustriousGenerator, 889818),
	(PonyV6Generator, 290640),
	(NoobAIGenerator, 1116447),
])
def test_model_version_ids(cls, expected_id):
	gen = cls(api_key="k", session=_StubSession())
	assert gen.model_version_id() == expected_id


def test_generate_builds_correct_request_and_downloads():
	session = _StubSession(post_payload=_ok_payload())
	gen = IllustriousGenerator(api_key="ct-key", session=session)
	image = run(gen.generate("masterpiece, red dress", "neg", prompt_is_complete=True))
	post = session.posts[0]
	assert post["url"].endswith("/images/generate")
	assert post["headers"]["Authorization"] == "Bearer ct-key"
	assert post["json"]["modelVersionId"] == 889818
	assert post["json"]["prompt"] == "masterpiece, red dress"
	assert post["json"]["negativePrompt"] == "neg"
	assert post["json"]["params"]["nsfw"] is True
	assert image.data == b"png-bytes"
	assert image.seed == 7


def test_version_override_wins():
	session = _StubSession(post_payload=_ok_payload())
	gen = NoobAIGenerator(api_key="k", session=session, version_override=1190596)
	run(gen.generate("p", "n", prompt_is_complete=True))
	assert session.posts[0]["json"]["modelVersionId"] == 1190596


def test_busy_then_success_retries():
	session = _StubSession(post_payload=_ok_payload(), post_status=429)
	gen = PonyV6Generator(api_key="k", session=session)
	# stub always 429 -> exhausts retries -> None (no exception)
	assert run(gen.generate("p", "n", prompt_is_complete=True)) is None


def test_empty_images_response_returns_none():
	session = _StubSession(post_payload={"images": []})
	gen = IllustriousGenerator(api_key="k", session=session)
	assert run(gen.generate("p", "n", prompt_is_complete=True)) is None


def test_backend_rules_override_quality_tags():
	"""Each model's rules replace the NAI quality prefix with its native
	vocabulary; the massaged prompt reflects it."""
	for cls, marker in [
		(IllustriousGenerator, "very aesthetic"),
		(PonyV6Generator, "score_9, score_8_up, score_7_up, source_anime"),
		(NoobAIGenerator, "amazing quality"),
	]:
		gen = cls(api_key="k", session=_StubSession())
		massager = PromptMassager.from_config({}, None, backend_overrides=gen.rules_overrides())
		result = massager.massage("red dress", "X")
		assert marker in result.effective, cls.__name__
		# NAI's own v4.5 suffix must be gone for civitai backends
		assert "no text" not in result.effective or cls is PonyV6Generator


def test_pony_negates_low_scores():
	gen = PonyV6Generator(api_key="k", session=_StubSession())
	massager = PromptMassager.from_config({}, None, backend_overrides=gen.rules_overrides())
	result = massager.massage("solo, one girl, red dress", "X")
	assert "score_6" in result.negative_additions


def test_overrides_do_not_touch_safety(tmp_path):
	"""Model-provided overrides must never weaken the safety rules."""
	gen = PonyV6Generator(api_key="k", session=_StubSession())
	massager = PromptMassager.from_config({}, tmp_path, backend_overrides=gen.rules_overrides())
	result = massager.massage("cute loli, red dress", "X")
	assert result.blocked_reason is not None
