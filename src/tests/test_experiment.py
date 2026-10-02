"""Unit tests for the experiment workbench pipeline (stubs, no network)."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.review.base import Verdict
from imagine_generator_proxy.server.experiment import ExperimentRunner, _run_payload
from imagine_generator_proxy.server.storage import Database, ensure_db_schema


def _png_bytes(seed: bytes = b"png"):
	return b"\x89PNG-fake-" + seed


class _StubGenerator:
	"""Yields one fake image per call; can be told to fail (return None)."""

	def __init__(self, fail_from: int = 999):
		self.calls = 0
		self._fail_from = fail_from
		self.prompts = []

	async def generate(self, prompt, negative_prompt, prompt_is_complete):
		self.prompts.append((prompt, negative_prompt, prompt_is_complete))
		self.calls += 1
		if self.calls > self._fail_from:
			return None
		return SimpleNamespace(name=f"image_{self.calls}.png", data=_png_bytes(str(self.calls).encode()), seed=self.calls)


class _ScriptedReviewer:
	def __init__(self, verdicts):
		self._verdicts = list(verdicts)
		self.calls = 0
		self.contexts = []

	async def review(self, image, prompt, attempt=0, previous_notes="", character=""):
		self.calls += 1
		self.contexts.append((attempt, previous_notes[:40], character))
		return self._verdicts.pop(0)


def _make_runner(tmp_path: Path, generator, reviewer) -> ExperimentRunner:
	fd, db = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	database = Database(Path(db))
	database.initialize()
	return ExperimentRunner(
		massager=PromptMassager.from_config({}, tmp_path),
		generator=generator,
		reviewer=reviewer,
		config={},
		cache_dir=tmp_path / "cache",
		database=database,
	)


def run(coro):
	return asyncio.run(coro)


def test_happy_path_records_all_variants(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([Verdict(approved=True)])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "huge breasts, red dress, solo",
		do_massage=True, do_review=True, max_attempts=5))

	assert "blocked" not in out
	# raw baseline + massaged attempt 0 (approved) = 2 generations, 2 variants
	assert gen.calls == 2
	assert len(out["variants"]) == 2
	assert out["variants"][0]["variant"] == "raw"
	assert out["variants"][0]["approved"] is None  # baseline unreviewed
	assert out["variants"][1]["variant"] == "attempt-0"
	assert out["variants"][1]["approved"] is True
	assert out["variants"][1]["is_final"] is True
	# massager info present with version
	assert out["massager"]["version"] is not None
	# images written inside cache dir and served via /cache URL
	for v in out["variants"]:
		assert v["image_url"].startswith("/cache/Naimi/experiments/r")
		assert (tmp_path / "cache" / v["image_url"][len("/cache/"):]).exists()


def test_quality_rejection_chain_applies_fixes(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="melted torso",
			fix_add=["arms at sides"], fix_negate=["fused fingers"]),
		Verdict(approved=True),
	])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "massive breasts, sitting",
		do_massage=True, do_review=True, max_attempts=5))

	assert len(out["variants"]) == 3  # raw + attempt-0 + attempt-1
	a0, a1 = out["variants"][1], out["variants"][2]
	assert a0["approved"] is False and a0["is_final"] is False
	assert a1["approved"] is True and a1["is_final"] is True
	assert "arms at sides" in a1["fix_applied"] and "neg:fused fingers" in a1["fix_applied"]
	# the fix reached the second generation's prompt
	assert "arms at sides" in gen.prompts[2][0]
	assert "fused fingers" in gen.prompts[2][1]
	# session context flowed into the second review
	assert reviewer.contexts[1][0] == 1
	assert "melted torso" in reviewer.contexts[1][1]


def test_minor_flag_stops_chain_fail_closed(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="minor", notes="age assessment failed"),
	])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "woman, red dress", do_massage=True, do_review=True, max_attempts=5))

	assert len(out["variants"]) == 2  # raw + attempt-0 only — chain stopped
	assert out["variants"][1]["category"] == "minor"
	assert out["variants"][1]["is_final"] is True  # terminal, but rejected
	assert gen.calls == 2


def test_safety_block_refuses_whole_experiment(tmp_path: Path):
	gen = _StubGenerator()
	runner = _make_runner(tmp_path, gen, _ScriptedReviewer([]))

	out = run(runner.run("Naimi", "cute loli, red dress", do_massage=True))
	assert "blocked" in out
	assert out["blocked"]["rule"]
	assert gen.calls == 0  # nothing generated


def test_unsafe_fix_proposals_are_dropped(tmp_path: Path):
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="bad",
			fix_add=["child", "nice pose"], fix_negate=[]),
		Verdict(approved=True),
	])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "woman, bedroom", do_massage=True, do_review=True, max_attempts=5))

	# the unsafe tag never reaches the regenerated prompt
	assert "child" not in gen.prompts[2][0]
	assert "nice pose" in gen.prompts[2][0]


def test_reviewer_unavailable_still_records_final(tmp_path: Path):
	gen = _StubGenerator()
	runner = _make_runner(tmp_path, gen, None)  # no reviewer

	out = run(runner.run("Naimi", "woman, red dress", do_massage=True, do_review=True, max_attempts=5))

	assert out["reviewer_used"] is False
	assert len(out["variants"]) == 2  # raw + single unreviewed attempt
	assert out["variants"][1]["is_final"] is True
	assert out["variants"][1]["approved"] is None


def test_chain_iterates_until_approved(tmp_path: Path):
	"""Three rejections then approval: the chain keeps going (applying fixes)
	until the reviewer approves — not a fixed attempt count."""
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="r1", fix_add=["fix1"]),
		Verdict(approved=False, category="quality", notes="r2", fix_add=["fix2"]),
		Verdict(approved=False, category="quality", notes="r3", fix_add=["fix3"]),
		Verdict(approved=True),
	])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "massive breasts, two girls", do_massage=True,
		do_review=True, max_attempts=6))

	assert len(out["variants"]) == 5  # raw + 4 attempts
	final = out["variants"][-1]
	assert final["approved"] is True and final["is_final"] is True
	assert "fix3" in final["fix_applied"]
	assert gen.calls == 5


def test_attempt_cap_stops_chain_with_clear_label(tmp_path: Path):
	"""A never-converging prompt stops at the cap, marked still-rejected
	(cost safety: experiments must not loop forever)."""
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="still bad", fix_add=["f"])
		for _ in range(5)
	])
	runner = _make_runner(tmp_path, gen, reviewer)

	out = run(runner.run("Naimi", "impossible prompt", do_massage=True,
		do_review=True, max_attempts=3))

	assert len(out["variants"]) == 4  # raw + 3 capped attempts
	last = out["variants"][-1]
	assert last["is_final"] is True and last["approved"] is False
	assert "attempt cap reached" in last["label"]
	assert gen.calls == 4


def test_db_serializer_roundtrip(tmp_path: Path):
	"""The DB-backed serializer must handle 'raw' and 'attempt-N' variants
	(catches the eager-default .get() bug that crashed the endpoints)."""
	fd, db = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	conn = sqlite3.connect(db)
	ensure_db_schema(conn)
	cur = conn.execute("INSERT INTO experiment_runs (character, raw_prompt, raw_negative)"
		" VALUES ('Naimi', 'raw prompt here', '')")
	conn.commit()
	run_id = cur.lastrowid
	conn.execute("INSERT INTO experiment_images (run_id, variant, prompt_used, negative_used, image_path, approved, category, notes, fix_applied, is_final)"
		" VALUES (?, 'raw', 'p0', 'n0', 'Naimi/experiments/r1/raw.png', NULL, NULL, '', '', 0)", (run_id,))
	conn.execute("INSERT INTO experiment_images (run_id, variant, prompt_used, negative_used, image_path, approved, category, notes, fix_applied, is_final)"
		" VALUES (?, 'attempt-0', 'p1', 'n1', 'Naimi/experiments/r1/attempt-0.png', 0, 'quality', 'melted', '', 0)", (run_id,))
	conn.execute("INSERT INTO experiment_images (run_id, variant, prompt_used, negative_used, image_path, approved, category, notes, fix_applied, is_final)"
		" VALUES (?, 'attempt-1', 'p2', 'n2', 'Naimi/experiments/r1/attempt-1.png', 1, NULL, '', 'arms at sides', 1)", (run_id,))
	conn.commit()
	payload = _run_payload(conn, run_id)
	conn.close()
	os.unlink(db)

	assert payload is not None
	labels = [v["label"] for v in payload["variants"]]
	assert labels[0].startswith("Raw prompt")
	assert "Attempt 1" in labels[1] and "vision-reviewed" in labels[1]
	assert "Attempt 2" in labels[2] and "review fixes: arms at sides" in labels[2]
	finals = [v["variant"] for v in payload["variants"] if v["is_final"]]
	assert finals == ["attempt-1"]


def test_db_serializer_marks_cap_reached(tmp_path: Path):
	"""History reloads label cap-stopped chains as still-rejected."""
	fd, db = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	conn = sqlite3.connect(db)
	ensure_db_schema(conn)
	cur = conn.execute("INSERT INTO experiment_runs (character, raw_prompt, raw_negative)"
		" VALUES ('Naimi', 'p', '')")
	conn.commit()
	run_id = cur.lastrowid
	conn.execute("INSERT INTO experiment_images (run_id, variant, prompt_used, negative_used,"
		" image_path, approved, category, notes, fix_applied, is_final)"
		" VALUES (?, 'attempt-2', 'p', 'n', 'x.png', 0, 'quality', 'bad', '', 1)", (run_id,))
	conn.commit()
	payload = _run_payload(conn, run_id)
	conn.close()
	os.unlink(db)
	assert "attempt cap reached" in payload["variants"][0]["label"]


def test_live_event_stream_sequence(tmp_path: Path):
	"""The on_event callback reports every pipeline step in order."""
	gen = _StubGenerator()
	reviewer = _ScriptedReviewer([
		Verdict(approved=False, category="quality", notes="melted torso", fix_add=["arms at sides"]),
		Verdict(approved=True),
	])
	runner = _make_runner(tmp_path, gen, reviewer)
	events: list = []

	async def collect(event):
		events.append(event)

	out = run(runner.run("Naimi", "massive breasts, sitting", do_massage=True,
		do_review=True, max_attempts=5, on_event=collect))

	assert "blocked" not in out
	phases = [(e["event"], e.get("phase") or e.get("variant"), e.get("variant")) for e in events]
	# ordered narrative: raw generate -> raw image -> attempt 1 generate ->
	# image -> reviewing -> verdict(reject) -> attempt 2 generate -> image ->
	# reviewing -> verdict(approve)
	assert phases == [
		("step", "generating", "raw"),
		("image", "raw", "raw"),
		("step", "generating", "attempt-0"),
		("image", "attempt-0", "attempt-0"),
		("step", "reviewing", "attempt-0"),
		("verdict", "attempt-0", "attempt-0"),
		("step", "generating", "attempt-1"),
		("image", "attempt-1", "attempt-1"),
		("step", "reviewing", "attempt-1"),
		("verdict", "attempt-1", "attempt-1"),
	]
	# the rejection verdict carries the fix so the page can narrate it
	reject = next(e for e in events if e["event"] == "verdict" and e["variant"] == "attempt-0")
	assert reject["approved"] is False and reject["category"] == "quality"
	# images arrive with a servable URL
	first_image = next(e for e in events if e["event"] == "image")
	assert first_image["image_url"].startswith("/cache/")


def test_blocked_prompt_emits_blocked_event(tmp_path: Path):
	gen = _StubGenerator()
	runner = _make_runner(tmp_path, gen, _ScriptedReviewer([]))
	events: list = []

	async def collect(event):
		events.append(event)

	run(runner.run("Naimi", "cute loli", on_event=collect))
	assert events and events[0]["event"] == "blocked"
