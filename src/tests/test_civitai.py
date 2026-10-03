"""Tests for the Civitai Orchestration generator hierarchy (stubbed, no network)."""

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


def _ok_workflow(status="succeeded", url="https://signed/img.jpeg"):
	return {"id": "wf_1", "status": status,
		"steps": [{"$type": "imageGen", "status": status,
			"output": {"images": [{"url": url, "seed": 9}]}}]}


class _StubResponse:
	def __init__(self, status, payload=None, raw=b""):
		self.status = status
		self._payload = payload
		self._raw = raw

	async def text(self):
		return json.dumps(self._payload) if self._payload is not None else ""

	async def read(self):
		return self._raw or b"png-bytes"

	async def __aenter__(self):
		return self

	async def __aexit__(self, *args):
		return False


class _StubSession:
	def __init__(self, submit_payload=None, submit_status=200):
		self.posts = []
		self._submit_payload = submit_payload or _ok_workflow()
		self._submit_status = submit_status

	def post(self, url, json=None, headers=None, timeout=None):
		self.posts.append({"url": url, "json": json, "headers": headers})
		return _StubResponse(self._submit_status, self._submit_payload)

	def get(self, url, timeout=None):
		if "/workflows/" in str(url):  # poll -> already terminal in stub
			return _StubResponse(200, self._submit_payload)
		return _StubResponse(200)  # signed-url download


def run(coro):
	return asyncio.run(coro)


@pytest.mark.parametrize("cls,urn", [
	(IllustriousGenerator, "urn:air:sdxl:checkpoint:civitai:795765@889818"),
	(PonyV6Generator, "urn:air:sdxl:checkpoint:civitai:257749@290640"),
	(NoobAIGenerator, "urn:air:sdxl:checkpoint:civitai:833294@1116447"),
])
def test_air_urns(cls, urn):
	gen = cls(api_key="k", session=_StubSession())
	assert gen.air_urn() == urn


def test_generate_builds_orchestration_workflow_and_downloads():
	session = _StubSession()
	gen = IllustriousGenerator(api_key="ct-key", session=session)
	image = run(gen.generate("masterpiece, red dress", "neg", prompt_is_complete=True))
	post = session.posts[0]
	assert "orchestration.civitai.com/v2/consumer/workflows" in post["url"]
	assert post["headers"]["Authorization"] == "Bearer ct-key"
	step = post["json"]["steps"][0]
	assert step["$type"] == "imageGen"
	assert step["input"]["model"] == "urn:air:sdxl:checkpoint:civitai:795765@889818"
	assert step["input"]["ecosystem"] == "sdxl"
	assert step["input"]["prompt"] == "masterpiece, red dress"
	assert image.data == b"png-bytes" and image.seed == 9


def test_urn_override_wins():
	session = _StubSession()
	gen = NoobAIGenerator(api_key="k", session=session,
		urn_override="urn:air:sdxl:checkpoint:civitai:833294@1190596")
	run(gen.generate("p", "n", prompt_is_complete=True))
	assert session.posts[0]["json"]["steps"][0]["input"]["model"].endswith("@1190596")


def test_upstream_error_returns_none():
	session = _StubSession(submit_status=500)
	gen = PonyV6Generator(api_key="k", session=session)
	assert run(gen.generate("p", "n", prompt_is_complete=True)) is None


def test_backend_rules_override_quality_tags():
	for cls, marker in [
		(IllustriousGenerator, "very aesthetic"),
		(PonyV6Generator, "score_9, score_8_up, score_7_up, source_anime"),
		(NoobAIGenerator, "amazing quality"),
	]:
		gen = cls(api_key="k", session=_StubSession())
		massager = PromptMassager.from_config({}, None, backend_overrides=gen.rules_overrides())
		result = massager.massage("red dress", "X")
		assert marker in result.effective, cls.__name__
		assert "no text" not in result.effective  # NAI v4.5 suffix removed


def test_pony_negates_low_scores():
	gen = PonyV6Generator(api_key="k", session=_StubSession())
	massager = PromptMassager.from_config({}, None, backend_overrides=gen.rules_overrides())
	result = massager.massage("solo, one girl, red dress", "X")
	assert "score_6" in result.negative_additions


def test_overrides_do_not_touch_safety(tmp_path):
	gen = PonyV6Generator(api_key="k", session=_StubSession())
	massager = PromptMassager.from_config({}, tmp_path, backend_overrides=gen.rules_overrides())
	result = massager.massage("cute loli, red dress", "X")
	assert result.blocked_reason is not None
