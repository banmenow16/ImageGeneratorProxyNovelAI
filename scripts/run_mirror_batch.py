"""Drive Mirror (/advice) runs from the command line for batch iteration.

Usage:
	python scripts/run_mirror_batch.py Kestrel Akari Sable Vaela Shiori

Reads each character's stored description from the DB, streams one full
advice run per character against the local server, prints live progress,
and writes the NDJSON event log to data/advice_analysis/<time>_<name>.ndjson.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "images.db"
OUT = ROOT / "data" / "advice_analysis"
URL = "http://localhost:8080/advice/run/stream"


def run_character(name: str) -> dict:
	conn = sqlite3.connect(str(DB))
	row = conn.execute("SELECT description FROM characters WHERE name = ?",
		(name,)).fetchone()
	conn.close()
	if not row or not (row[0] or "").strip():
		print(f"[{name}] no stored description — skipping")
		return {}
	body = json.dumps({"character": name, "description": row[0]}).encode("utf-8")
	req = urllib.request.Request(URL, data=body,
		headers={"Content-Type": "application/json"})
	summary: dict = {"stages": {}, "done": False}
	OUT.mkdir(parents=True, exist_ok=True)
	log = OUT / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.ndjson"
	with (open(log, "w", encoding="utf-8") as fh,
			urllib.request.urlopen(req, timeout=7200) as resp):
		for line_b in resp:
			line = line_b.decode("utf-8").strip()
			if not line:
				continue
			fh.write(line + "\n")
			fh.flush()
			e = json.loads(line)
			ev, ph = e.get("event"), e.get("phase")
			if ev == "stage" and ph == "start":
				print(f"[{name}] stage: {e['label']}", flush=True)
			elif ev == "critique":
				v = e.get("verdict") or "?"
				wrong = (e.get("wrong") or "")[:70]
				never = (e.get("never") or "")[:40]
				print(f"  {e['stage']:12s} r{e['retry']:>2} [{v:6s}]"
					+ f" wrong: {wrong}"
					+ (f" | never: {never}" if never else ""), flush=True)
			elif ev == "stage" and ph == "done":
				summary["stages"][e["stage"]] = (e["status"], e["attempts"])
				print(f"  -> {e['stage']}: {e['status']} ({e['attempts']} attempts)",
					flush=True)
			elif ev == "done":
				summary["done"] = True
				for k in ("run_id", "final_prompt", "final_negative",
						"converged_count", "total_stages", "advice"):
					summary[k] = e.get(k)
				print(f"[{name}] DONE run_id={e['run_id']} converged"
					f" {e['converged_count']}/{e['total_stages']}", flush=True)
				print(f"  final tags: {(e.get('final_prompt') or '')[:150]}", flush=True)
				print(f"  negatives : {(e.get('final_negative') or '')[:100]}", flush=True)
			elif ev == "error":
				print(f"[{name}] ERROR: {e.get('error')}", flush=True)
	summary["log"] = str(log)
	return summary


if __name__ == "__main__":
	names = sys.argv[1:] or ["Kestrel", "Akari", "Sable", "Vaela", "Shiori"]
	for n in names:
		print(f"=== {n} ===", flush=True)
		try:
			run_character(n)
		except Exception as ex:  # keep the batch alive
			print(f"[{n}] FAILED: {ex}", flush=True)
