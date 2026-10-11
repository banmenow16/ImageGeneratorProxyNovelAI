"""DAO for the images table (cached generation metadata)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


class ImageDAO:
	"""Data access for the images table — cached generation records."""

	def __init__(self, db_path: Path):
		self._path = db_path

	def connect(self) -> sqlite3.Connection:
		return sqlite3.connect(str(self._path))

	def create_table(self) -> None:
		conn = self.connect()
		try:
			conn.execute('''CREATE TABLE IF NOT EXISTS images (
				id INTEGER PRIMARY KEY,
				character TEXT,
				digest TEXT UNIQUE,
				prompt TEXT,
				negative_prompt TEXT,
				model TEXT,
				first_appearance TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)''')
			conn.commit()
		finally:
			conn.close()

	def migrate(self) -> None:
		"""Additive column migrations."""
		conn = self.connect()
		try:
			cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
			if "raw_prompt" not in cols:
				conn.execute("ALTER TABLE images ADD COLUMN raw_prompt TEXT")
			if "massager_version" not in cols:
				conn.execute("ALTER TABLE images ADD COLUMN massager_version INTEGER")
			conn.commit()
		finally:
			conn.close()

	def insert(self, character: str, digest: str, prompt: str,
			negative_prompt: str, model: str, raw_prompt: str,
			massager_version: Optional[int]) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"INSERT OR IGNORE INTO images (character, digest, prompt, negative_prompt,"
				" model, raw_prompt, massager_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
				(character, digest, prompt, negative_prompt, model, raw_prompt, massager_version))
			conn.commit()
		except Exception:
			LOGGER.exception("Failed to store generated image in database")
		finally:
			conn.close()

	def get_by_character(self, character: str) -> list:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			return conn.execute(
				"SELECT * FROM images WHERE character = ? ORDER BY id DESC",
				(character,)).fetchall()
		finally:
			conn.close()

	def count(self) -> int:
		conn = self.connect()
		try:
			return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
		finally:
			conn.close()
