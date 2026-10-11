"""Heal advice_runs rows orphaned by the KeyError-at-the-tail bug (790d06c).

For every run with stages recorded but finished_at NULL:
- recompute converged_count (a stage counts if any round's verdict is "yes"
  — robust against the old initial-status-"skipped" bug too)
- identity_tags = last round's identity block (identity persists across stages)
- negative_tags from the matching NDJSON event log when one exists
- regenerate the final advice via the (now fixed) character_advice — the
  expensive stages already happened; the tail should exist for them
- stamp finished_at and update characters.canonical_tags/advice

Usage: python scripts/repair_advice_runs.py [--dry-run] [--skip-advice]
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB = ROOT / "data" / "images.db"
LOGS = ROOT / "data" / "advice_analysis"

DRY_RUN = "--dry-run" in sys.argv
SKIP_ADVICE = "--skip-advice" in sys.argv


def load_config() -> dict:
	import yaml
	with open(ROOT / "data" / "config.yaml", encoding="utf-8") as fh:
		return yaml.safe_load(fh) or {}


def negatives_from_log(run_id: int, character: str) -> str:
	"""The run's final negative list from its NDJSON event log, if any."""
	import glob
	best: str = ""
	for f in sorted(LOGS.glob(f"*_{character}.ndjson")):
		never_seen: set = set()
		for line in Path(f).read_text(encoding="utf-8").splitlines():
			try:
				e = json.loads(line)
			except json.JSONDecodeError:
				continue
			if e.get("event") == "critique" and e.get("never"):
				for t in e["never"].split(","):
					if t.strip():
						never_seen.add(t.strip())
		if never_seen:
			best = ", ".join(sorted(never_seen))
	return best


def main() -> None:
	conn = sqlite3.connect(str(DB))
	conn.row_factory = sqlite3.Row
	runs = conn.execute("""SELECT id, character FROM advice_runs
		WHERE finished_at IS NULL ORDER BY id""").fetchall()
	if not runs:
		print("nothing to repair")
		return

	advice_fn = None
	if not SKIP_ADVICE:
		from imagine_generator_proxy.review.character import character_advice
		config = load_config()
		or_key = config.get("openRouterToken")
		model = config.get("critique_model", "x-ai/grok-4.7")
		if or_key:

			async def _advice(name: str, description: str, stages_summary: str,
					tags: str, negs: str) -> str:
				return await character_advice(or_key, model, name, description,
					stages_summary, final_tags=tags, final_negative=negs) or ""

			advice_fn = _advice

	for run in runs:
		rid, name = run["id"], run["character"]
		stages = conn.execute("""SELECT id, stage_id, stage_label, status, notes FROM advice_stages
			WHERE run_id = ? ORDER BY id""", (rid,)).fetchall()
		rounds = conn.execute("""SELECT stage_db_id, retry_number, identity_tags_used,
			critique_verdict FROM advice_rounds WHERE run_id = ? ORDER BY id""",
			(rid,)).fetchall()
		if not rounds:
			print(f"run {rid} ({name}): no rounds — leaving as-is")
			continue

		# a stage converged if any of its rounds said yes
		yes_stages = {r["stage_db_id"] for r in rounds if r["critique_verdict"] == "yes"}
		converged = sum(1 for s in stages if s["id"] in yes_stages)
		final_tags = rounds[-1]["identity_tags_used"] or ""
		# NEVER USE is a post-feature (run 11+) addition — older runs never had it
		negs = negatives_from_log(rid, name) if rid >= 11 else ""
		summary = "; ".join(
			f"{s['stage_label']}: {'converged' if s['id'] in yes_stages else s['status']}"
			for s in stages)

		advice_text = ""
		if advice_fn:
			desc_row = conn.execute("SELECT description FROM characters WHERE name = ?",
				(name,)).fetchone()
			if desc_row and desc_row["description"]:
				try:
					advice_text = asyncio.run(advice_fn(name, desc_row["description"],
						summary, final_tags, negs))
				except Exception as ex:  # noqa: BLE001 — repair must not die mid-batch
					print(f"run {rid} ({name}): advice regeneration failed: {ex}")

		print(f"run {rid} ({name}): converged {converged}/{len(stages)},"
			f" tags={'yes' if final_tags else 'NO'}, negs={'yes' if negs else 'no'},"
			f" advice={'yes' if advice_text else 'no'}")
		if DRY_RUN:
			continue

		conn.execute("""UPDATE advice_runs SET identity_tags = ?, negative_tags = ?,
			advice = ?, converged_count = ?, total_stages = ?,
			finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
			(final_tags, negs, advice_text, converged, len(stages), rid))
		if final_tags:
			conn.execute("""UPDATE characters SET canonical_tags = ?, advice = ?,
				updated_at = CURRENT_TIMESTAMP WHERE name = ?""",
				(final_tags, advice_text, name))
		conn.commit()

	print("done" if not DRY_RUN else "dry run — nothing written")


if __name__ == "__main__":
	main()
