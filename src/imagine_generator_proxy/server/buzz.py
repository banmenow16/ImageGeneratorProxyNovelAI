"""Local Civitai Buzz tracking: Civitai's API exposes no balance endpoint for
our key, so we accumulate each workflow's reported transactions and show
remaining against a user-set baseline (data/buzz_state.json)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from aiohttp import web

_lock = threading.Lock()


def _state_path(data_dir: Path) -> Path:
	return data_dir / "buzz_state.json"


def _load(data_dir: Path) -> dict:
	path = _state_path(data_dir)
	if path.exists():
		try:
			return json.loads(path.read_text(encoding="utf-8"))
		except Exception:
			pass
	return {"baseline": None, "spent": {"blue": 0, "yellow": 0}}


def _save(data_dir: Path, state: dict) -> None:
	_state_path(data_dir).write_text(json.dumps(state), encoding="utf-8")


def record_transactions(data_dir: Path, transactions) -> None:
	"""Accumulate debits/credits from a workflow's transactions block."""
	if not transactions:
		return
	with _lock:
		state = _load(data_dir)
		for entry in transactions.get("list", []):
			account = entry.get("accountType", "blue")
			if account not in state["spent"]:
				state["spent"][account] = 0
			amount = entry.get("amount", 0)
			state["spent"][account] += amount if entry.get("type") == "debit" else -amount
		_save(data_dir, state)


async def buzz_handler(request: web.Request) -> web.Response:
	from imagine_generator_proxy.server.app import DATA_DIR
	state = _load(DATA_DIR)
	spent = state["spent"]
	total_spent = sum(spent.values())
	baseline = state.get("baseline")
	payload = {
		"spent": spent,
		"total_spent": total_spent,
		"baseline": baseline,
		"remaining": (baseline - total_spent) if isinstance(baseline, (int, float)) else None,
	}
	return web.json_response(payload)


def add_routes(app: web.Application) -> None:
	app.router.add_get("/api/buzz", buzz_handler)
