"""DAO for advice_runs, advice_stages, advice_rounds — Mirror session data."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


class AdviceDAO:
	"""Data access for Mirror sessions: runs, stages, and every round."""

	def __init__(self, db_path: Path):
		self._path = db_path

	def connect(self) -> sqlite3.Connection:
		return sqlite3.connect(str(self._path))

	def migrate(self) -> None:
		conn = self.connect()
		try:
			conn.execute('''CREATE TABLE IF NOT EXISTS advice_runs (
				id INTEGER PRIMARY KEY,
				character TEXT NOT NULL,
				engine TEXT,
				description_hash TEXT,
				identity_tags TEXT,
				negative_tags TEXT,
				advice TEXT,
				converged_count INTEGER DEFAULT 0,
				total_stages INTEGER DEFAULT 5,
				started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				finished_at TIMESTAMP
			)''')
			conn.execute('''CREATE TABLE IF NOT EXISTS advice_stages (
				id INTEGER PRIMARY KEY,
				run_id INTEGER NOT NULL REFERENCES advice_runs(id),
				stage_id TEXT NOT NULL,
				stage_label TEXT,
				status TEXT DEFAULT "pending",
				attempts INTEGER DEFAULT 0,
				notes TEXT,
				final_verdict TEXT
			)''')
			conn.execute('''CREATE TABLE IF NOT EXISTS advice_rounds (
				id INTEGER PRIMARY KEY,
				stage_db_id INTEGER NOT NULL REFERENCES advice_stages(id),
				run_id INTEGER NOT NULL REFERENCES advice_runs(id),
				retry_number INTEGER NOT NULL,
				full_prompt TEXT,
				identity_tags_used TEXT,
				critique_verdict TEXT,
				critique_right TEXT,
				critique_wrong TEXT,
				critique_rewritten TEXT,
				critique_feel TEXT,
				image_path TEXT
			)''')
			conn.commit()
		finally:
			conn.close()

	# --- run lifecycle ---

	def start_run(self, character: str, engine: str, description_hash: str) -> int:
		conn = self.connect()
		try:
			cur = conn.execute(
				"INSERT INTO advice_runs (character, engine, description_hash)"
				" VALUES (?, ?, ?)", (character, engine, description_hash))
			conn.commit()
			return int(cur.lastrowid or 0)
		finally:
			conn.close()

	def finish_run(self, run_id: int, identity_tags: str, negative_tags: str,
			advice: str, converged: int, total: int) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"UPDATE advice_runs SET identity_tags = ?, negative_tags = ?,"
				" advice = ?, converged_count = ?, total_stages = ?,"
				" finished_at = CURRENT_TIMESTAMP WHERE id = ?",
				(identity_tags, negative_tags, advice, converged, total, run_id))
			conn.commit()
		finally:
			conn.close()

	# --- stages ---

	def start_stage(self, run_id: int, stage_id: str, stage_label: str) -> int:
		conn = self.connect()
		try:
			cur = conn.execute(
				"INSERT INTO advice_stages (run_id, stage_id, stage_label)"
				" VALUES (?, ?, ?)", (run_id, stage_id, stage_label))
			conn.commit()
			return int(cur.lastrowid or 0)
		finally:
			conn.close()

	def finish_stage(self, stage_db_id: int, status: str, attempts: int,
			notes: str, final_verdict: str) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"UPDATE advice_stages SET status = ?, attempts = ?,"
				" notes = ?, final_verdict = ? WHERE id = ?",
				(status, attempts, notes[:1000], final_verdict, stage_db_id))
			conn.commit()
		finally:
			conn.close()

	# --- rounds ---

	def record_round(self, stage_db_id: int, run_id: int, retry: int,
			full_prompt: str, identity_tags: str, verdict: str,
			right: str, wrong: str, rewritten: str, feel: str,
			image_path: str) -> None:
		conn = self.connect()
		try:
			conn.execute(
				"INSERT INTO advice_rounds (stage_db_id, run_id, retry_number,"
				" full_prompt, identity_tags_used, critique_verdict,"
				" critique_right, critique_wrong, critique_rewritten,"
				" critique_feel, image_path)"
				" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(stage_db_id, run_id, retry, full_prompt[:2000],
					identity_tags[:2000], verdict, right[:1000],
					wrong[:1000], rewritten[:2000], feel[:500], image_path))
			conn.commit()
		finally:
			conn.close()

	# --- queries ---

	def list_runs(self, character: Optional[str] = None, limit: int = 50) -> list:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			if character:
				return conn.execute(
					"SELECT * FROM advice_runs WHERE character = ? ORDER BY id DESC LIMIT ?",
					(character, limit)).fetchall()
			return conn.execute(
				"SELECT * FROM advice_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
		finally:
			conn.close()

	def get_run(self, run_id: int) -> Optional[dict]:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			row = conn.execute("SELECT * FROM advice_runs WHERE id = ?", (run_id,)).fetchone()
			return dict(row) if row else None
		finally:
			conn.close()

	def get_stages(self, run_id: int) -> list:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			return conn.execute(
				"SELECT * FROM advice_stages WHERE run_id = ? ORDER BY id",
				(run_id,)).fetchall()
		finally:
			conn.close()

	def get_rounds(self, run_id: int) -> list:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			return conn.execute(
				"SELECT * FROM advice_rounds WHERE run_id = ? ORDER BY id",
				(run_id,)).fetchall()
		finally:
			conn.close()

	def get_round(self, round_id: int) -> Optional[dict]:
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			row = conn.execute(
				"SELECT * FROM advice_rounds WHERE id = ?", (round_id,)).fetchone()
			return dict(row) if row else None
		finally:
			conn.close()

	def aggregate_failures(self, character: Optional[str] = None) -> list:
		"""Most common 'what went wrong' entries across unresolved rounds."""
		conn = self.connect()
		conn.row_factory = sqlite3.Row
		try:
			query = """
				SELECT ar.critique_wrong, COUNT(*) as count,
					GROUP_CONCAT(DISTINCT aru.character) as characters
				FROM advice_rounds ar
				JOIN advice_stages ast ON ar.stage_db_id = ast.id
				JOIN advice_runs aru ON ar.run_id = aru.id
				WHERE ast.status = 'unresolved'
					AND ar.critique_wrong IS NOT NULL
					AND ar.critique_wrong != ''
			"""
			params = []
			if character:
				query += " AND aru.character = ?"
				params.append(character)
			query += " GROUP BY ar.critique_wrong ORDER BY count DESC LIMIT 30"
			return conn.execute(query, params).fetchall()
		finally:
			conn.close()
