"""Tests for the cache gallery pages (stubbed request, temp data)."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from imagine_generator_proxy.prompts.massager import LEGACY_NEGATIVE_BLOCKS
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
		negative_prompt TEXT, model TEXT, raw_prompt TEXT,
		first_appearance TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
	conn.execute("INSERT INTO images (character, digest, prompt, negative_prompt, raw_prompt) VALUES (?, ?, ?, ?, ?)",
		("Naimi", "abc123", "score_9, source_anime, woman, <angle> brackets & quotes",
		 ", ".join(LEGACY_NEGATIVE_BLOCKS + [""]),  # realistic legacy shape (blocks + empty query)
		 "woman, <angle> brackets & quotes"))
	conn.commit()
	conn.close()

	cache = tmp_path / "cache"
	char_dir = cache / "Naimi"
	char_dir.mkdir(parents=True)
	(char_dir / "abc123.png").write_bytes(b"\x89PNG-fake")
	(char_dir / "fff999.png").write_bytes(b"\x89PNG-fake")  # no DB row
	# cards sort by mtime, newest first — pin the order or the two writes can
	# straddle a timestamp tick and flip which card the link assertions see
	now = time.time()
	os.utime(char_dir / "abc123.png", (now, now))
	os.utime(char_dir / "fff999.png", (now - 86400, now - 86400))
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
	# negative prompt present in full, with the legacy trailing comma trimmed
	assert "negative" in body
	assert "text, watermark, signature" in body  # legacy block 1
	assert "deformed, ugly, blurry, lowres" in body  # legacy block 2
	assert "unnatural proportions</div>" in body  # trailing ", " trimmed for display
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


def test_card_links_to_prefilled_workbench(tmp_path: Path):
	"""Each image card links to /experiment with the RAW prompt (not the
	stored prefixed one) and the full displayed negative — matching the
	manual copy-paste workflow the link replaces."""
	import re
	from html import unescape
	from urllib.parse import parse_qs, urlparse
	response = run(folder_handler(_FakeRequest(_make_app(tmp_path), {"character": "Naimi"})))
	body = response.text
	hrefs = re.findall(r'href="(/experiment\?[^"]+)"', body)
	assert hrefs, "no experiment link on the card"
	# hrefs are HTML-escaped in the attribute (& -> &amp;); browsers decode
	# that when parsing, so decode here too before checking the params
	params = parse_qs(urlparse(unescape(hrefs[0])).query)
	assert params["character"] == ["Naimi"]
	assert params["prompt"] == ["woman, <angle> brackets & quotes"]
	# the FULL negative as displayed (trailing comma trimmed), not just the
	# query negative — the user copies what the card shows
	assert params["negative"] == [", ".join(LEGACY_NEGATIVE_BLOCKS)]


def test_card_prefill_strips_negative_leak(tmp_path: Path):
	"""Raw prompts often carry the SillyTavern 'negative:' leak inside the
	positive; the prefill must not reproduce it in the workbench field."""
	import re
	from html import unescape
	from urllib.parse import parse_qs, urlparse
	app = _make_app(tmp_path)
	import sqlite3
	conn = sqlite3.connect(app["db_path"])
	conn.execute("UPDATE images SET raw_prompt=? WHERE digest='abc123'",
		("woman, red dress, negative: deformed, ugly, blurry, lowres",))
	conn.commit()
	conn.close()
	response = run(folder_handler(_FakeRequest(app, {"character": "Naimi"})))
	hrefs = re.findall(r'href="(/experiment\?[^"]+)"', response.text)
	# the no-DB-row card carries an empty prompt param; use the populated one
	parsed = [parse_qs(urlparse(unescape(h)).query) for h in hrefs]
	params = next(p for p in parsed if p.get("prompt"))
	assert params["prompt"] == ["woman, red dress"]
	assert "deformed" not in params["prompt"][0]


def test_strip_negative_leak_helper():
	from imagine_generator_proxy.prompts import strip_negative_leak
	assert strip_negative_leak("girl, red dress, negative: deformed, ugly") == "girl, red dress"
	assert strip_negative_leak("girl, masterpiece, best quality, deformed, ugly, blurry, lowres, extra limbs, outdoor") \
		== "girl, masterpiece, best quality"
	assert strip_negative_leak("girl, red dress") == "girl, red dress"