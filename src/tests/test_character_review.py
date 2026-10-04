"""Tests for persona-injected character review (Phase 1)."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import os
from pathlib import Path

from imagine_generator_proxy.review.base import Verdict
from imagine_generator_proxy.review.character import CharacterReviewer
from imagine_generator_proxy.server.storage import Database, ensure_db_schema


def run(coro):
	return asyncio.run(coro)


def _db(tmp_path: Path) -> Database:
	fd, path = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	database = Database(Path(path))
	database.initialize()
	return database


def test_character_description_roundtrip(tmp_path: Path):
	database = _db(tmp_path)
	assert database.get_character("Naimi") is None
	database.set_character_description("Naimi", "auburn hair, glasses, massive breasts")
	char = database.get_character("Naimi")
	assert char is not None and "auburn" in char["description"]
	database.set_character_description("Naimi", "updated")
	assert "updated" in database.get_character("Naimi")["description"]


class _RecordingReviewer:
	"""Stub capturing context_note and delegating a fixed verdict."""

	def __init__(self, verdict):
		self._verdict = verdict
		self.notes = []

	async def review(self, image, prompt, attempt=0, previous_notes="",
			character="", context_note=""):
		self.notes.append(context_note)
		return self._verdict


def test_character_reviewer_injects_persona():
	inner = _RecordingReviewer(Verdict(approved=True))
	wrapped = CharacterReviewer(inner, "Shiori", "short black hair, red eyes, gothic")
	verdict = run(wrapped.review(b"img", "prompt", character="Shiori"))
	assert verdict.approved is True
	note = inner.notes[0]
	assert "Shiori" in note and "short black hair" in note and "LIKENESS" in note


def test_character_reviewer_passes_through_existing_note():
	inner = _RecordingReviewer(Verdict(approved=True))
	wrapped = CharacterReviewer(inner, "X", "desc")
	run(wrapped.review(b"img", "p", context_note="STUDIO OVERRIDE"))
	assert inner.notes[0] == "STUDIO OVERRIDE"  # explicit note wins


def test_openrouter_reviewer_accepts_context_note():
	from imagine_generator_proxy.review.openrouter import OpenRouterReviewer
	import inspect
	sig = inspect.signature(OpenRouterReviewer.review)
	assert "context_note" in sig.parameters
