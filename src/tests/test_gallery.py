"""Tests for the cache gallery pages (stubbed request, temp data)."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path

from imagine_generator_proxy.server.gallery import folder_handler, gallery_handler


class _FakeRequest:
	def __init__(self, app: dict, match_info: dict | None = None):
		self.app = app
		self.match_info = match_info or {}


def _make_app(tmp_path: Path) -> dict:
	fd, db = tempfile.mkstemp(suffix=".db", dir=str(tmp_path))
	os.close(fd)
	conn = sqlite3.connect(db)
	conn.execute("""CREATE TABLE images (
		id INTEGER PRIMARY KEY, character TEXT, digest TEXT UNIQUE, prompt TEXT,
		negative_prompt TEXT, model TEXT,
		first_appearance TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
	conn.execute("INSERT INTO images (character, digest, prompt, negative_prompt) VALUES (?, ?, ?, ?)",
		("Naimi", "abc123", "woman, <angle> brackets & quotes",
		 "text, watermark, deformed, ugly, "))
	conn.commit()
	conn.close()

	cache = tmp_path / "cache"
	char_dir = cache / "Naimi"
	char_dir.mkdir(parents=True)
	(char_dir / "abc123.png").write_bytes(b"\x89PNG-fake")
	(char_dir / "fff999.png").write_bytes(b"\x89PNG-fake")  # no DB row
	return {"cache_dir": cache, "db_path": Path(db)}


def run(coro):
	return asyncio.run(coro)


def test_folder_page_shows_positive_and_negative(tmp_path: Path):
	response = run(folder_handler(_FakeRequest(_make_app(tmp_path), {"character": "Naimi"})))
	assert response.status == 200
	body = response.text
	# positive prompt present and HTML-escaped
	assert "woman" in body
	assert "&lt;angle&gt;" in body and "<angle>" not in body
	assert "&amp; quotes" in body
	# negative prompt present, with the legacy trailing comma trimmed for display
	assert "negative" in body
	assert "text, watermark, deformed, ugly" in body
	assert "deformed, ugly, " not in body  # trailing ", " trimmed
	# image without a DB row renders a not-recorded negative instead of crashing
	assert "(not recorded)" in body


def test_folder_index_lists_character_folders(tmp_path: Path):
	response = run(gallery_handler(_FakeRequest(_make_app(tmp_path))))
	assert response.status == 200
	assert 'href="/folder/Naimi"' in response.text
	assert "1" in response.text  # image count


def test_folder_handler_404_for_unknown_character(tmp_path: Path):
	app = _make_app(tmp_path)
	response = run(folder_handler(_FakeRequest(app, {"character": "Nobody"})))
	assert response.status == 404
