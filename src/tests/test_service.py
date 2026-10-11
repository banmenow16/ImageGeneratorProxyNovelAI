"""Unit tests for GenerationService (stubbed generator/reviewer, real
massager + Database; no network)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.review.base import Verdict
from imagine_generator_proxy.server.service import (
	GenerationRequest,
	GenerationService,
	PromptBlockedError,
	UpstreamBusyError,
	ImageRejectedError,
)
from imagine_generator_proxy.server.storage import Database


class _StubGenerator:
	def __init__(self, fail=False):
		self.fail = fail
		self.calls = 0
		self.prompts = []

	async def generate(self, prompt, negative_prompt, prompt_is_complete):
		self.calls += 1
		self.prompts.append((prompt, negative_prompt, prompt_is_complete))
		if self.fail:
			return None
		return SimpleNamespace(name="image_0.png", data=b"\x89PNG-fake-service",
			seed=self.calls)


class _ScriptedReviewer:
	def __init__(self, verdicts):
		self._verdicts = list(verdicts)

	async def review(self, image, prompt, attempt=0, previous_notes="", character=""):
		return self._verdicts.pop(0)


def _make_service(tmp_path: Path, generator, reviewer, config=None):
	fd, db = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	database = Database(Path(db))
	database.initialize()
	return GenerationService(
		massager=PromptMassager.from_config({}, tmp_path),
		generator=generator,
		reviewer=reviewer,
		config=config or {},
		cache_dir=tmp_path / "cache",
		database=database,
	)


def _request(prompt="woman, red dress", **kwargs):
	defaults = dict(raw_prompt=prompt, character="Naimi", q_negative="",
		want_raw=False, force=True, use_massage=True)
	defaults.update(kwargs)
	return GenerationRequest(**defaults)


def run(coro):
	return asyncio.run(coro)


def test_blocked_prompt_on_massaged_path_raises(tmp_path: Path):
	"""Regression guard: massage() returns blocked results internally; the
	service must translate them into PromptBlockedError (a missed check here
	once let blocked prompts flow to generation)."""
	gen = _StubGenerator()
	service = _make_service(tmp_path, gen, _ScriptedReviewer([]))

	with pytest.raises(PromptBlockedError) as exc:
		run(service.generate(_request("cute loli, red dress")))
	assert exc.value.rule
	assert gen.calls == 0


def test_blocked_prompt_on_bypass_path_raises(tmp_path: Path):
	gen = _StubGenerator()
	service = _make_service(tmp_path, gen, _ScriptedReviewer([]))

	with pytest.raises(PromptBlockedError):
		run(service.generate(_request("a child playing", use_massage=False)))
	assert gen.calls == 0


def test_successful_generation_writes_cache_and_db(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([Verdict(approved=True)])
	service = _make_service(tmp_path, gen, reviewer)

	response = run(service.generate(_request()))
	assert response.status == 200
	body = run_response_json(response)
	assert body["cached"] is False
	assert body["massaged"] is True

	# cache file exists under the character dir
	cached = list((tmp_path / "cache" / "Naimi").glob("*.png"))
	assert len(cached) == 1
	# images table row recorded
	conn = service._database.connect()
	rows = conn.execute("SELECT character, raw_prompt, massager_version FROM images").fetchall()
	conn.close()
	assert rows == [("Naimi", "woman, red dress", service._massager.version)]


def run_response_json(response):
	import json
	async def _read():
		return json.loads((response.body or b"{}").decode("utf-8"))
	return asyncio.run(_read())


def test_cache_hit_skips_generation(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([Verdict(approved=True)])
	service = _make_service(tmp_path, gen, reviewer)

	first = run(service.generate(_request(force=True)))
	assert gen.calls == 1
	second = run(service.generate(_request(force=False, want_raw=False)))
	assert gen.calls == 1  # served from cache, no new generation
	body = run_response_json(second)
	assert body["cached"] is True


def test_busy_upstream_raises_and_releases_waiters(tmp_path: Path):
	gen = _StubGenerator(fail=True)
	service = _make_service(tmp_path, gen, _ScriptedReviewer([]))

	with pytest.raises(UpstreamBusyError):
		run(service.generate(_request(force=True)))
	assert service._generations == {}


def test_minor_rejection_remediates_then_refuses_at_exhaustion(tmp_path: Path):
	"""A minor-flag regenerates with adult-reinforcement fixes; if it persists
	to the last attempt, the image is refused (fail-closed at exhaustion)."""
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="minor", notes="childlike",
			fix_add=["adult woman"], fix_negate=["chibi"]),
		Verdict(approved=False, category="minor", notes="still childlike",
			fix_add=["adult woman"], fix_negate=["chibi"]),
	])
	service = _make_service(tmp_path, gen, reviewer, config={"vision_review": True, "vision_review_retries": 1})

	with pytest.raises(ImageRejectedError):
		run(service.generate(_request(force=True)))
	# it retried with the age fixes before refusing
	assert gen.calls == 2
	assert "adult woman" in gen.prompts[1][0]
	assert "chibi" in gen.prompts[1][1]
	# both rejections were logged for the hint history
	conn = service._database.connect()
	rows = conn.execute("SELECT approved, category FROM reviews ORDER BY id").fetchall()
	conn.close()
	assert rows == [(0, "minor"), (0, "minor")]


def test_minor_then_quality_rejection_still_fail_open(tmp_path: Path):
	"""Quality exhaustion after an earlier age-flag serves the image — the
	fail-closed rule applies to the FINAL verdict only."""
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="minor", notes="childlike",
			fix_add=["adult woman"], fix_negate=[]),
		Verdict(approved=False, category="quality", notes="hands"),
	])
	service = _make_service(tmp_path, gen, reviewer, config={"vision_review": True, "vision_review_retries": 1})

	response = run(service.generate(_request(force=True)))
	assert response.status == 200
	assert gen.calls == 2


def test_quality_rejection_applies_fixes_and_retries(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="melted torso",
			fix_add=["arms at sides"], fix_negate=["fused fingers"]),
		Verdict(approved=True),
	])
	service = _make_service(tmp_path, gen, reviewer, config={"vision_review": True, "vision_review_retries": 1})

	response = run(service.generate(_request(force=True)))
	assert response.status == 200
	assert gen.calls == 2
	# fix tags reached the regenerated prompt, safety-checked
	assert "arms at sides" in gen.prompts[1][0]
	assert "fused fingers" in gen.prompts[1][1]


def test_quality_exhaustion_still_serves(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="still bad"),
		Verdict(approved=False, category="quality", notes="still bad 2"),
	])
	service = _make_service(tmp_path, gen, reviewer, config={"vision_review": True, "vision_review_retries": 1})

	response = run(service.generate(_request(force=True)))
	assert response.status == 200  # fail-open on quality after retries
	assert gen.calls == 2


def test_bypass_mode_uses_legacy_prompt_and_toggle(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([])  # vision_review off -> reviewer not consulted? No: noop? service consults reviewer regardless
	service = _make_service(tmp_path, gen, reviewer, config={})

	# reviewer with empty script would fail if consulted; vision_review off
	# means outcomes aren't logged but review() still runs. Use an approving one.
	class _Approve:
		async def review(self, *a, **k):
			return Verdict(approved=True)
	service._reviewer = _Approve()

	run(service.generate(_request("woman, red dress", use_massage=False)))
	# legacy path: library adds its own quality tags => prompt_is_complete False
	assert gen.prompts[0][2] is False
	assert gen.prompts[0][0].startswith("score_9, score_8_up, score_7_up, score_6_up, source_anime,woman")
