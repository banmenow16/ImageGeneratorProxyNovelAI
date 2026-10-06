"""DAO for the characters table (stored descriptions, canonical tags, advice)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


class CharacterDAO:
	"""Data access for characters — descriptions, tag blocks, advice."""

	def __init__(self, db_path: Path):
		self._path = db_path

	def connect(self) -> sqlite3.Connection:
		return sqlite3.connect(str(self._path))

	def migrate(self) -> None:
		conn = self.connect()
		try:
			conn.execute('''CREATE TABLE IF NOT EXISTS characters (
				name TEXT PRIMARY KEY,
				description TEXT,
				canonical_tags TEXT,
				advice TEXT,
				updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)''')
			conn.commit()
		finally:
			conn.close()

	def get(self, name: str) -> Optional[dict]:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			row = conn.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
			return dict(row) if row else None
		finally:
			conn.close()

	def set_description(self, name: str, description: str) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"INSERT INTO characters (name, description, updated_at)"
				" VALUES (?, ?, CURRENT_TIMESTAMP)"
				" ON CONFLICT(name) DO UPDATE SET description = excluded.description,"
				" updated_at = CURRENT_TIMESTAMP", (name, description))
			conn.commit()
		finally:
			conn.close()

	def set_tags_and_advice(self, name: str, canonical_tags: str,
			advice: str) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"UPDATE characters SET canonical_tags = ?, advice = ?,"
				" updated_at = CURRENT_TIMESTAMP WHERE name = ?",
				(canonical_tags, advice, name))
			conn.commit()
		finally:
			conn.close()
