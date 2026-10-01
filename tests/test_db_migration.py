"""Tests for the additive images-table migration."""

from __future__ import annotations

import sqlite3

from imagine_generator_proxy.main import ensure_db_schema

OLD_SCHEMA = """
CREATE TABLE images (
	id INTEGER PRIMARY KEY,
	character TEXT,
	digest TEXT UNIQUE,
	prompt TEXT,
	negative_prompt TEXT,
	model TEXT,
	first_appearance TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _columns(conn):
	return {row[1] for row in conn.execute("PRAGMA table_info(images)")}


def test_migration_adds_columns_to_old_schema():
	conn = sqlite3.connect(":memory:")
	conn.execute(OLD_SCHEMA)
	assert not {"raw_prompt", "massager_version"} & _columns(conn)
	ensure_db_schema(conn)
	assert {"raw_prompt", "massager_version"} <= _columns(conn)
	# review-memory table is created alongside
	tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
	assert "reviews" in tables


def test_review_logging_roundtrip():
	from imagine_generator_proxy.main import log_review_outcome
	from imagine_generator_proxy.reviewer import Verdict
	from types import SimpleNamespace
	import tempfile, os

	fd, path = tempfile.mkstemp(suffix=".db")
	os.close(fd)
	conn = sqlite3.connect(path)
	conn.execute(OLD_SCHEMA)
	ensure_db_schema(conn)
	conn.close()

	log_review_outcome(path, "abc123", "Naimi", Verdict(approved=False, category="quality", notes="fused fingers"))
	log_review_outcome(path, "def456", "Naimi", Verdict(approved=True))
	conn = sqlite3.connect(path)
	rows = conn.execute("SELECT digest, approved, category FROM reviews ORDER BY id").fetchall()
	conn.close()
	os.unlink(path)
	assert rows == [("abc123", 0, "quality"), ("def456", 1, None)]


def test_migration_is_idempotent():
	conn = sqlite3.connect(":memory:")
	conn.execute(OLD_SCHEMA)
	ensure_db_schema(conn)
	ensure_db_schema(conn)
	ensure_db_schema(conn)
	assert {"raw_prompt", "massager_version"} <= _columns(conn)


def test_migration_on_fresh_schema():
	# on_startup path: CREATE TABLE IF NOT EXISTS then ensure_db_schema
	conn = sqlite3.connect(":memory:")
	conn.execute(OLD_SCHEMA.replace("CREATE TABLE images", "CREATE TABLE IF NOT EXISTS images"))
	ensure_db_schema(conn)
	assert {"raw_prompt", "massager_version"} <= _columns(conn)
