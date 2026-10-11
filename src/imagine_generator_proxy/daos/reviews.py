"""DAO for the reviews table (production attempt ledger + hint history)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from imagine_generator_proxy.review.base import Verdict

LOGGER = logging.getLogger(__name__)


class ReviewDAO:
	"""Data access for reviews — per-attempt review outcomes with provenance."""

	def __init__(self, db_path: Path):
		self._path = db_path

	def connect(self) -> sqlite3.Connection:
		return sqlite3.connect(str(self._path))

	def migrate(self) -> None:
		conn = self.connect()
		try:
			conn.execute('''CREATE TABLE IF NOT EXISTS reviews (
				id INTEGER PRIMARY KEY,
				digest TEXT,
				character TEXT,
				approved INTEGER,
				category TEXT,
				notes TEXT,
				attempt INTEGER DEFAULT 0,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)''')
			cols = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
			for col, decl in (("attempt", "INTEGER DEFAULT 0"), ("tags", "TEXT"),
					("prompt_used", "TEXT"), ("negative_used", "TEXT"),
					("fix_applied", "TEXT"), ("is_final", "INTEGER DEFAULT 0")):
				if col not in cols:
					conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {decl}")
			conn.commit()
		finally:
			conn.close()

	def log(self, digest: str, character: str, verdict: Verdict,
			attempt: int = 0, tags: str = "", prompt_used: str = "",
			negative_used: str = "", fix_applied: str = "",
			is_final: bool = False) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"INSERT INTO reviews (digest, character, approved, category, notes,"
				" attempt, tags, prompt_used, negative_used, fix_applied, is_final)"
				" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(digest, character, 1 if verdict.approved else 0, verdict.category,
					(verdict.notes or "")[-1000:], attempt, tags[-2000:],
					prompt_used[-2000:], negative_used[-2000:], fix_applied[:500],
					1 if is_final else 0))
			conn.commit()
		except Exception:
			LOGGER.exception("Failed to log review outcome")
		finally:
			conn.close()

	def recent_for_character(self, character: str, limit: int = 8) -> list:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			return conn.execute(
				"SELECT substr(digest,1,8) as d8, approved, category, attempt,"
				" substr(notes,1,140) as notes"
				" FROM reviews WHERE character = ? ORDER BY id DESC LIMIT ?",
				(character, limit)).fetchall()
		finally:
			conn.close()
