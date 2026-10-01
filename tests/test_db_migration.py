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
