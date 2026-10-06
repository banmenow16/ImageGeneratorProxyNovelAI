"""Advice page (/advice): the character mirror — staged wardrobe edition.

Five composition probes (portrait, from behind, topless, nude, nude behind),
each with up to N retries. Identity tags evolve across stages; composition
tags are per-stage and never saved to the identity block. The final advice
summarizes per-view reliability and produces a structured instruction block
for the RP bot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import ImageGenerator
from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.review.character import character_advice, character_critique
from imagine_generator_proxy.server.storage import Database

LOGGER = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "advice.html"

CRITIQUE_MODEL_DEFAULT = "x-ai/grok-4.7"

STAGES = [
	{"id": "portrait", "label": "Portrait",
		"composition": "standing, front view, full body, looking at viewer, simple background"},
	{"id": "from_behind", "label": "From Behind",
		"composition": "from behind, back view, standing, looking away"},
	{"id": "topless", "label": "Topless",
		"composition": "topless, upper body, front view"},
	{"id": "nude", "label": "Nude (front)",
		"composition": "nude, full body, front view"},
	{"id": "nude_behind", "label": "Nude from Behind",
		"composition": "nude, from behind, back view, full body"},
]

MAX_RETRIES = 5  # per stage


def _dedupe_tags(tag_string: str) -> str:
	seen = set()
	out = []
	for tag in tag_string.split(","):
		tag = tag.strip()
		if not tag:
			continue
		key = tag.lower()
		if key not in seen:
			seen.add(key)
			out.append(tag)
	return ", ".join(out)


def _strip_composition(tags: str) -> str:
	"""Remove known composition tags from an identity string (the critique
	model sometimes includes them despite instructions)."""
	composition_tags = {
		"standing", "front view", "full body", "looking at viewer",
		"simple background", "from behind", "back view", "looking away",
		"upper body", "nude", "topless", "solo", "1girl",
	}
	return ", ".join(t.strip() for t in tags.split(",")
		if t.strip() and t.strip().lower() not in composition_tags)


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

	# --- derive identity tags if no starting prompt ---
	identity_tags: str = prompt
	if not identity_tags:
		await emit({"event": "step", "phase": "deriving-prompt"})
		identity_tags = await derive_identity_tags(or_key, critique_model, name, description)
		if not identity_tags:
			await emit({"event": "error", "error": "could not derive identity tags"})
			await response.write_eof()
			return response
		await emit({"event": "prompt", "prompt": identity_tags})

	identity_tags = _strip_composition(_dedupe_tags(identity_tags))
	current_negative: str = ""
	stage_results: list = []
	all_critiques: list = []
	image_counter = 0

	# --- staged probes ---
	for stage in STAGES:
		stage_id = stage["id"]
		stage_label = stage["label"]
		stage_composition = stage["composition"]
		stage_result = {"id": stage_id, "label": stage_label, "status": "skipped",
			"attempts": 0, "notes": ""}
		await emit({"event": "stage", "phase": "start", "stage": stage_id,
			"label": stage_label})

		for retry in range(MAX_RETRIES):
			stage_result["attempts"] = retry + 1
			# combine identity + this stage's composition
			full_prompt = f"{identity_tags}, {stage_composition}"
			image_counter += 1

			await emit({"event": "step", "phase": "generating", "stage": stage_id,
				"retry": retry + 1, "prompt": full_prompt})
			massaged = massager.massage(full_prompt, name)
			if massaged.blocked_reason is not None:
				stage_result["notes"] = f"safety block: {massaged.blocked_reason}"
				await emit({"event": "step", "phase": "skipped", "stage": stage_id})
				break

			negative = ", ".join(filter(None, [current_negative,
				", ".join(massaged.negative_additions) if massaged.negative_additions else ""]))
			image = await generator.generate(massaged.effective, negative, prompt_is_complete=True)
			if image is None:
				stage_result["notes"] = "generation failed"
				await emit({"event": "step", "phase": "skipped", "stage": stage_id})
				break

			img_path = character_dir / f"{stage_id}_r{retry + 1}.png"
			img_path.write_bytes(bytes(image.data))
			url = f"/advice-files/{name}/{stage_id}_r{retry + 1}.png"
			await emit({"event": "image", "stage": stage_id, "retry": retry + 1,
				"url": url, "prompt": massaged.effective})

			await emit({"event": "step", "phase": "critiquing", "stage": stage_id,
				"retry": retry + 1})
			critique = await character_critique(or_key, critique_model, name,
				description, identity_tags, bytes(image.data),
				stage_label=stage_label, stage_composition=stage_composition)
			if critique is None:
				stage_result["notes"] = "critique failed"
				await emit({"event": "step", "phase": "skipped", "stage": stage_id})
				break

			all_critiques.append(critique)
			await emit({"event": "critique", "stage": stage_id, "retry": retry + 1,
				**critique})

			if critique["verdict"] == "yes":
				stage_result["status"] = "converged"
				stage_result["notes"] = critique.get("right", "")
				await emit({"event": "step", "phase": "converged", "stage": stage_id})
				break

			# she rewrites the identity tags (composition stays separate)
			if critique.get("rewritten"):
				rewritten = _strip_composition(critique["rewritten"].strip())
				if massager.check_safety(rewritten) is None:
					identity_tags = _dedupe_tags(rewritten)

			if retry < MAX_RETRIES - 1:
				await emit({"event": "step", "phase": "retrying", "stage": stage_id,
					"retry": retry + 1})
		else:
			# retries exhausted without convergence
			if stage_result["status"] not in ("converged", "skipped"):
				stage_result["status"] = "unresolved"
				last = all_critiques[-1] if all_critiques else {}
				stage_result["notes"] = last.get("wrong", "did not converge")

		stage_results.append(stage_result)
		await emit({"event": "stage", "phase": "done", "stage": stage_id,
			"status": stage_result["status"], "attempts": stage_result["attempts"]})

	# --- final advice ---
	await emit({"event": "step", "phase": "advising"})
	rounds_summary = "; ".join(
		f"{s['label']}: {s['status']}"
		+ (f" ({s['notes'][:120]})" if s["notes"] else "")
		for s in stage_results)
	advice_text = await character_advice(or_key, critique_model, name,
		description, rounds_summary, final_tags=identity_tags,
		final_negative=current_negative)

	# save
	conn = database.connect()
	try:
		conn.execute("UPDATE characters SET canonical_tags = ?, advice = ?,"
			" updated_at = CURRENT_TIMESTAMP WHERE name = ?",
			(identity_tags, advice_text or "", name))
		conn.commit()
	finally:
		conn.close()

	converged_count = sum(1 for s in stage_results if s["status"] == "converged")
	await emit({"event": "done", "final_prompt": identity_tags,
		"final_negative": current_negative, "advice": advice_text or "",
		"stage_results": stage_results,
		"satisfied": converged_count == len(STAGES),
		"converged_count": converged_count, "total_stages": len(STAGES)})
	await response.write_eof()
	return response


async def derive_identity_tags(api_key: str, model: str, name: str,
		description: str) -> str:
	"""Ask the critique model to produce identity-only tags from her portrait."""
	from openai import AsyncOpenAI
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=60.0, max_retries=2)
	instruction = (
		f"Based on this self-description of {name}, write a comma-separated "
		f"IDENTITY tag list for generating anime images of her. IDENTITY ONLY — "
		f"her permanent physical features (hair color/length, eye color, skin, "
		f"build, breast size, body features like wings/horns/tail/scales, "
		f"distinguishing marks). NO pose tags, NO composition tags, NO setting "
		f"tags, NO clothing (unless it's permanently part of her look). "
		f"Keep under 20 tags. Output ONLY the tag list.\n\n"
		f"{description[:3000]}")
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=200,
			messages=[{"role": "user", "content": instruction}])
		result = (resp.choices[0].message.content or "").strip()
		return result.strip("`").strip()
	except Exception:
		LOGGER.exception("identity tag derivation failed for %s", name)
		return ""


def add_routes(app: web.Application) -> None:
	app.router.add_get("/advice", advice_page_handler)
	app.router.add_post("/advice/run/stream", advice_run_stream_handler)
