"""Tests for the OpenAI Images compatibility controller."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from imagine_generator_proxy.controllers.openai_compat import OpenAICompatController
from imagine_generator_proxy.generation.base import Backend, GeneratedImage


PNG = b"\x89PNG-fake-bytes"


class _FakeGenerator:
	"""Records calls; returns a canned image."""

	def __init__(self) -> None:
		self.calls: list[tuple[str, str, bool, object]] = []

	async def generate(self, prompt: str, negative: str,
			prompt_is_complete: bool, size=None, steps=None):
		self.calls.append((prompt, negative, prompt_is_complete, size, steps))
		return GeneratedImage(name="image.png", data=PNG)


@dataclass
class _FakeMassager:
	"""Zero-transform massager: prompt passes through, nothing added."""

	result: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(
		effective="", negative_additions=[], blocked_reason=None))

	def massage(self, prompt: str, character: str) -> SimpleNamespace:
		self.result.effective = prompt
		return self.result

	def check_safety(self, tags: str):
		return None


class _FakeRequest:
	"""Minimal aiohttp request double: app state + JSON body + host."""

	def __init__(self, payload: dict) -> None:
		self._payload = payload
		self.app: dict = {}
		self.scheme = "http"
		self.host = "localhost:8080"

	async def json(self) -> dict:
		return self._payload


def _controller(tmp_path: Path) -> tuple[OpenAICompatController, _FakeGenerator]:
	generator = _FakeGenerator()
	backend = Backend(generator=generator, massager=_FakeMassager(),
		name="novelai", display="NovelAI")
	controller = OpenAICompatController({"novelai": backend}, "novelai",
		tmp_path)
	return controller, generator


def run(coro):
	return asyncio.run(coro)


def test_b64_json_round_trip(tmp_path: Path):
	controller, generator = _controller(tmp_path)
	response = run(controller.generations(_FakeRequest(
		{"prompt": "1girl, solo", "model": "gpt-image-1", "n": 1,
			"size": "landscape", "response_format": "b64_json"})))
	body = json.loads(response.text)
	assert body["data"][0]["b64_json"] == base64.b64encode(PNG).decode()
	# unknown model fell back to the default engine; size became the tuple
	assert generator.calls[0][3] == (1216, 832)


def test_url_format_saves_file_and_returns_absolute_url(tmp_path: Path):
	controller, _ = _controller(tmp_path)
	response = run(controller.generations(_FakeRequest(
		{"prompt": "1girl, solo", "size": "832x1216"})))
	body = json.loads(response.text)
	url = body["data"][0]["url"]
	assert url.startswith("http://localhost:8080/compat-files/")
	filename = url.rsplit("/", 1)[1]
	assert (tmp_path / "compat" / filename).read_bytes() == PNG


def test_named_and_wxh_sizes_and_default(tmp_path: Path):
	controller, generator = _controller(tmp_path)
	for size, expected in (("portrait", (832, 1216)), ("square", (1024, 1024)),
			("1024x1024", (1024, 1024)), (None, None)):
		run(controller.generations(_FakeRequest(
			{"prompt": "1girl", "size": size, "response_format": "b64_json"})))
	assert generator.calls[-1][3] == expected


def test_invalid_size_and_bad_format_rejected(tmp_path: Path):
	controller, _ = _controller(tmp_path)
	assert run(controller.generations(_FakeRequest(
		{"prompt": "1girl", "size": "ginormous"}))).status == 400
	assert run(controller.generations(_FakeRequest(
		{"prompt": "1girl", "response_format": "yaml"}))).status == 400


def test_n_capped_at_four(tmp_path: Path):
	controller, generator = _controller(tmp_path)
	run(controller.generations(_FakeRequest(
		{"prompt": "1girl", "n": 9, "response_format": "b64_json"})))
	assert len(generator.calls) == 4


def test_negative_prompt_extension_flows_through(tmp_path: Path):
	controller, generator = _controller(tmp_path)
	run(controller.generations(_FakeRequest(
		{"prompt": "1girl", "negative_prompt": "blue hair",
			"response_format": "b64_json"})))
	assert "blue hair" in generator.calls[0][1]
