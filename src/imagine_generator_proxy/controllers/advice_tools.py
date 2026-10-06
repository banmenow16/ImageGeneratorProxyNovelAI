"""Advice iteration tools: quick-test, cross-check, failure aggregation.

These endpoints exist to make prompt iteration fast — single-stage tests,
neutral re-reviews of persona critiques, and aggregate failure analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.daos.advice import AdviceDAO
from imagine_generator_proxy.review.openrouter import OpenRouterReviewer, parse_verdict

LOGGER = logging.getLogger(__name__)


async def quick_test_handler(request: web.Request) -> web.Response:
	"""Single-stage, single-retry Mirror test for fast prompt iteration.

	POST body: {character, description, identity_tags?, engine?, stage?}
	Returns: the generated image URL + the critique, in ~45 seconds.
	"""
	try:
		body = await request.json()
	except Exception:
		return web.json_response({"error": "request body must be JSON"}, status=400)

	name = (body.get("character") or "").strip()
	description = (body.get("description") or "").strip()
	if not name or not description:
		return web.json_response({"error": "character and description required"}, status=400)

	# pick a single stage (default: portrait)
	from imagine_generator_proxy.server.advice import STAGES
	stage_id = body.get("stage", "portrait")
	stage = next((s for s in STAGES if s["id"] == stage_id), STAGES[0])

	# identity tags: provided or derived
	identity_tags = (body.get("identity_tags") or "").strip()
	if not identity_tags:
		return web.json_response(
			{"error": "identity_tags required (derive via /advice first, or provide manually)"},
			status=400)

	# engine
	backends = request.app.get("backends") or {}
	engine_name = body.get("engine") or request.app.get("generation_backend", "novelai")
	backend = backends.get(engine_name)
	if backend is None:
		return web.json_response({"error": f"engine {engine_name!r} unavailable"}, status=400)

	# critique model
	config = request.app.get("config") or {}
	or_key = config.get("openRouterToken")
	if not or_key:
		return web.json_response({"error": "no openRouterToken"}, status=400)
	critique_model = config.get("critique_model", "x-ai/grok-4.7")

	from imagine_generator_proxy.review.character import character_critique

	full_prompt = f"{identity_tags}, {stage['composition']}"
	massaged = backend.massager.massage(full_prompt, name)
	if massaged.blocked_reason:
		return web.json_response({"error": f"blocked: {massaged.blocked_reason}"}, status=400)

	image = await backend.generator.generate(massaged.effective, "", prompt_is_complete=True)
	if image is None:
		return web.json_response({"error": "generation failed"}, status=502)

	# save image to a quick-test directory
	qt_dir = Path(request.app["videos_dir"]).parent / "advice" / "quicktest"
	qt_dir.mkdir(parents=True, exist_ok=True)
	img_path = qt_dir / f"{name}_{stage['id']}.png"
	img_path.write_bytes(bytes(image.data))

	critique = await character_critique(or_key, critique_model, name,
		description, identity_tags, bytes(image.data),
		stage_label=stage["label"], stage_composition=stage["composition"])

	return web.json_response({
		"stage": stage["id"],
		"prompt_sent": massaged.effective,
		"image_url": f"/advice-files/quicktest/{name}_{stage['id']}.png",
		"critique": critique,
	})


async def crosscheck_handler(request: web.Request) -> web.Response:
	"""Re-review a Mirror round's image with the NEUTRAL QC prompt (no persona).

	Compares what the persona critique said vs. what the neutral reviewer
	sees — if they disagree, the persona may be hallucinating.
	"""
	try:
		round_id = int(request.match_info["round_id"])
	except ValueError:
		return web.json_response({"error": "invalid round id"}, status=400)

	dao = AdviceDAO(request.app["db_path"])
	round_data = dao.get_round(round_id)
	if round_data is None:
		return web.json_response({"error": "round not found"}, status=404)

	# load the image
	data_dir = Path(request.app["videos_dir"]).parent / "advice"
	image_path = data_dir / round_data["image_path"]
	if not image_path.exists():
		return web.json_response({"error": f"image not found: {round_data['image_path']}"}, status=404)

	# run the neutral QC reviewer
	config = request.app.get("config") or {}
	or_key = config.get("openRouterToken")
	if not or_key:
		return web.json_response({"error": "no openRouterToken"}, status=400)

	reviewer = OpenRouterReviewer.from_config(config)
	if reviewer is None:
		return web.json_response({"error": "reviewer unavailable"}, status=500)

	image_bytes = image_path.read_bytes()
	neutral_verdict = await reviewer.review(image_bytes, round_data["full_prompt"] or "")

	return web.json_response({
		"round_id": round_id,
		"character": round_data.get("character", ""),
		"persona_verdict": round_data["critique_verdict"],
		"persona_wrong": round_data["critique_wrong"],
		"persona_feel": round_data["critique_feel"],
		"neutral_approved": neutral_verdict.approved,
		"neutral_category": neutral_verdict.category,
		"neutral_notes": neutral_verdict.notes,
		"agree": (neutral_verdict.approved and round_data["critique_verdict"] == "yes")
			or (not neutral_verdict.approved and round_data["critique_verdict"] != "yes"),
	})


async def failures_handler(request: web.Request) -> web.Response:
	"""Aggregate failure analysis: what goes wrong most across unresolved stages."""
	dao = AdviceDAO(request.app["db_path"])
	character = request.query.get("character")
	failures = dao.aggregate_failures(character)

	return web.json_response({
		"failures": [dict(f) for f in failures],
		"count": len(failures),
	})


def add_routes(app: web.Application) -> None:
	app.router.add_post("/api/advice/quick-test", quick_test_handler)
	app.router.add_get("/api/advice/rounds/{round_id}/crosscheck", crosscheck_handler)
	app.router.add_get("/api/advice/failures", failures_handler)
