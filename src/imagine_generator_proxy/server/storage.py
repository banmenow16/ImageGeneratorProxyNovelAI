"""Persistence layer: cache digests, database schema/migrations, and
recording of generations and review outcomes."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from imagine_generator_proxy.review.base import Verdict

LOGGER = logging.getLogger(__name__)


def compute_digest(character: str, prompt: str, negative_prompt: str, model: str,
		massager_version: Optional[int]) -> str:
	"""Cache digest for a generation request.

	massager_version None keeps the historical formula byte-identical so
	bypass requests keep hitting the pre-massager cache. Massaged prompts
	are namespaced with a "pmv<N>" prefix: bumping the rule version
	regenerates, an unchanged version keeps everything cached.
	"""
	parts = [character, prompt, negative_prompt, str(model)]
	if massager_version is not None:
		parts.insert(0, "pmv%d" % massager_version)
	return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def ensure_db_schema(conn: sqlite3.Connection) -> None:
	"""Create/extend all tables (idempotent; safe on a fresh database)."""
	cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
	if cols:  # images table exists (created by on_startup just before this)
		if "raw_prompt" not in cols:
			conn.execute("ALTER TABLE images ADD COLUMN raw_prompt TEXT")
		if "massager_version" not in cols:
			conn.execute("ALTER TABLE images ADD COLUMN massager_version INTEGER")
	# Review memory: every vision-review outcome (approved and rejected) is
	# recorded so future hint retrieval has history to learn from.
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
	review_cols = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
	if "attempt" not in review_cols:
		conn.execute("ALTER TABLE reviews ADD COLUMN attempt INTEGER DEFAULT 0")
	if "tags" not in review_cols:
		conn.execute("ALTER TABLE reviews ADD COLUMN tags TEXT")
	# attempt ledger (production twin of experiment_images): full per-attempt
	# provenance for post-hoc failure analysis and rules-engine tuning
	for col, decl in (("prompt_used", "TEXT"), ("negative_used", "TEXT"),
			("fix_applied", "TEXT"), ("is_final", "INTEGER DEFAULT 0")):
		if col not in review_cols:
			conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {decl}")
	# Character studio: per-character descriptions (persona-injected review),
	# canonical tag blocks and advice for the RP bot (Phase 2)
	conn.execute('''CREATE TABLE IF NOT EXISTS characters (
		name TEXT PRIMARY KEY,
		description TEXT,
		canonical_tags TEXT,
		advice TEXT,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)''')
	# Experiment workbench: provenance for /experiment runsExperiment workbench: provenance for /experiment runs
	conn.execute('''CREATE TABLE IF NOT EXISTS experiment_runs (
		id INTEGER PRIMARY KEY,
		character TEXT,
		raw_prompt TEXT,
		raw_negative TEXT,
		effective_prompt TEXT,
		effective_negative TEXT,
		massager_version INTEGER,
		applied_rules TEXT,
		removed_tags TEXT,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)''')
	conn.execute('''CREATE TABLE IF NOT EXISTS experiment_images (
		id INTEGER PRIMARY KEY,
		run_id INTEGER,
		variant TEXT,
		prompt_used TEXT,
		negative_used TEXT,
		image_path TEXT,
		approved INTEGER,
		category TEXT,
		notes TEXT,
		fix_applied TEXT,
		is_final INTEGER DEFAULT 0,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)''')
	conn.commit()


class Database:
	"""Owns the SQLite database path; all writes go through here.

	Every method is best-effort: persistence problems are logged and never
	propagate into request handling.
	"""

	def __init__(self, db_path: Path):
		self._path = db_path

	@property
	def path(self) -> Path:
		return self._path

	def connect(self) -> sqlite3.Connection:
		return sqlite3.connect(str(self._path))

	def initialize(self) -> None:
		"""Create the images table and apply migrations (startup path)."""
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
			ensure_db_schema(conn)
		finally:
			conn.close()

	def record_generated_image(self, character: str, digest: str, prompt: str,
			negative_prompt: str, model: str, raw_prompt: str,
			massager_version) -> None:
		"""Insert a cache row for a freshly generated image."""
		try:
			conn = self.connect()
			try:
				conn.execute(
					"INSERT OR IGNORE INTO images (character, digest, prompt, negative_prompt, model, raw_prompt, massager_version)"
					" VALUES (?, ?, ?, ?, ?, ?, ?)",
					(character, digest, prompt, negative_prompt, model, raw_prompt, massager_version),
				)
				conn.commit()
			finally:
				conn.close()
		except Exception:
			LOGGER.exception("Failed to store generated image in database")


	def get_character(self, name: str) -> Optional[dict]:
		conn = self.connect()
		try:
			conn.row_factory = sqlite3.Row
			row = conn.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
			return dict(row) if row else None
		finally:
			conn.close()

	def set_character_description(self, name: str, description: str) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"INSERT INTO characters (name, description, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)"
				" ON CONFLICT(name) DO UPDATE SET description = excluded.description,"
				" updated_at = CURRENT_TIMESTAMP", (name, description))
			conn.commit()
		finally:
			conn.close()
	def log_review_outcome(self, digest: str, character: str, verdict: Verdict,
			attempt: int = 0, tags: str = "", prompt_used: str = "",
			negative_used: str = "", fix_applied: str = "",
			is_final: bool = False) -> None:
		"""Record one generation attempt's review outcome with full
		provenance (the production attempt ledger)."""
		try:
			conn = self.connect()
			try:
				conn.execute(
					"INSERT INTO reviews (digest, character, approved, category, notes, attempt, tags,"
					" prompt_used, negative_used, fix_applied, is_final)"
					" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
					(digest, character, 1 if verdict.approved else 0, verdict.category,
						(verdict.notes or "")[-1000:], attempt, tags[-2000:],
						prompt_used[-2000:], negative_used[-2000:], fix_applied[:500],
						1 if is_final else 0),
				)
				conn.commit()
			finally:
				conn.close()
		except Exception:
			LOGGER.exception("Failed to log review outcome")
