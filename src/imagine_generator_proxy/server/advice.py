"""Advice page (/advice): the character mirror loop.

Paste how she describes herself (from the RP bot via the mirror prompt),
give a starting prompt, and the loop runs: generate a portrait -> she
critiques it as herself -> her tag fixes are applied -> regenerate, until
she says LOOKS LIKE ME: yes (or rounds run out). The final tag block is
saved to the characters table as canonical_tags.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import ImageGenerator
from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.prompts.massager import sanitize_fix_tags
from imagine_generator_proxy.review.character import character_advice, character_critique
from imagine_generator_proxy.server.storage import Database

LOGGER = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "advice.html"

CRITIQUE_MODEL_DEFAULT = "x-ai/grok-4.7"  # persona voice benefits from the strong model


async def advice_page_handler(request: web.Request) -> web.Response:
	html = PAGE_PATH.read_text(encoding="utf-8")
	return web.Response(text=html, content_type="text/html",
		headers={"Cache-Control": "no-cache"})


def _backends(app: web.Application) -> dict:
	return app.get("backends") or {}


async def advice_run_stream_handler(request: web.Request) -> web.StreamResponse:
	try:
		body = await request.json()
	except Exception:
		return web.json_response({"error": "request body must be JSON"}, status=400)

	name = (body.get("character") or "").strip()
	description = (body.get("description") or "").strip()
	prompt = (body.get("prompt") or "").strip()
	if not name or not description:
		return web.json_response({"error": "character name and description are required"}, status=400)

	backends = _backends(request.app)
	backend_name = body.get("engine") or request.app.get("generation_backend", "novelai")
	backend = backends.get(backend_name)
	if backend is None or getattr(backend.generator, "nsfw_blocked", False):
		return web.json_response({"error": f"engine {backend_name!r} unavailable",
			"available": [k for k, v in backends.items() if not getattr(v.generator, "nsfw_blocked", False)]}, status=400)
	generator: ImageGenerator = backend.generator
	massager: PromptMassager = backend.massager

	try:
		rounds = max(1, min(int(body.get("rounds", 3)), 5))
	except (TypeError, ValueError):
		rounds = 3

	or_key = (request.app.get("config") or {}).get("openRouterToken")
	if not or_key:
		return web.json_response({"error": "no openRouterToken — critique needs the vision model"}, status=400)
	critique_model = (request.app.get("config") or {}).get("critique_model", CRITIQUE_MODEL_DEFAULT)

	response = web.StreamResponse(headers={
		"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
	await response.prepare(request)

	async def emit(event: dict) -> None:
		await response.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))

	database = Database(request.app["db_path"])
	database.set_character_description(name, description)
	character_dir = Path(request.app["videos_dir"]).parent / "advice" / name
	character_dir.mkdir(parents=True, exist_ok=True)

	current_prompt = prompt
	current_negative = ""
	critiques = []

	for round_no in range(rounds):
		await emit({"event": "step", "phase": "generating", "round": round_no + 1,
			"prompt": current_prompt})
		massaged = massager.massage(current_prompt, name)
		if massaged.blocked_reason is not None:
			await emit({"event": "error", "error": f"prompt blocked: {massaged.blocked_reason}"})
			stopped_early = True
			break
		negative = ", ".join([current_negative, massaged.negative_additions and
			", ".join(massaged.negative_additions) or ""]).strip(", ")
		image = await generator.generate(massaged.effective, negative,
			prompt_is_complete=True)
		if image is None:
			await emit({"event": "error", "error": "generation failed"})
			stopped_early = True
			break
		img_path = character_dir / f"round{round_no + 1}.png"
		img_path.write_bytes(bytes(image.data))
		url = f"/advice-files/{name}/round{round_no + 1}.png"
		await emit({"event": "image", "round": round_no + 1, "url": url,
			"prompt": massaged.effective})

		await emit({"event": "step", "phase": "critiquing", "round": round_no + 1})
		critique = await character_critique(or_key, critique_model, name,
			description, bytes(image.data))
		if critique is None:
			await emit({"event": "error", "error": "critique failed"})
			stopped_early = True
			break
		critiques.append(critique)
		await emit({"event": "critique", "round": round_no + 1, **critique})

		if critique["verdict"] == "yes":
			stopped_early = True  # she's satisfied — stop the loop below after save
			await emit({"event": "step", "phase": "satisfied", "round": round_no + 1})
			break

		# apply her tag fixes to the prompt for the next round
		add, negate = [], []
		for fix in critique["fixes"]:
			if fix.lower().startswith("neg:"):
				negate.append(fix[4:].strip())
			else:
				add.append(fix)
		add, negate = sanitize_fix_tags(add, negate, current_prompt)
		add = [t for t in add if massager.check_safety(t) is None]
		negate = [t for t in negate if massager.check_safety(t) is None]
		if add:
			current_prompt = current_prompt + ", " + ", ".join(add)
		if negate:
			current_negative = (current_negative + ", " + ", ".join(negate)).strip(", ")

	# she writes her advice for the RP bot based on the rounds
	await emit({"event": "step", "phase": "advising"})
	rounds_summary = "; ".join(
		f"round {i + 1}: {c.get('wrong', '') or 'nothing wrong'}"
		for i, c in enumerate(critiques))
	advice_text = await character_advice(or_key, critique_model, name,
		description, rounds_summary)

	# save the evolved tag block + advice
	conn = database.connect()
	try:
		conn.execute("UPDATE characters SET canonical_tags = ?, advice = ?,"
			" updated_at = CURRENT_TIMESTAMP WHERE name = ?",
			(current_prompt, advice_text or "", name))
		conn.commit()
	finally:
		conn.close()

	satisfied = bool(critiques) and critiques[-1].get("verdict") == "yes"
	await emit({"event": "done", "final_prompt": current_prompt,
		"final_negative": current_negative, "advice": advice_text or "",
		"satisfied": satisfied})
	await response.write_eof()
	return response


def add_routes(app: web.Application) -> None:
	app.router.add_get("/advice", advice_page_handler)
	app.router.add_post("/advice/run/stream", advice_run_stream_handler)
