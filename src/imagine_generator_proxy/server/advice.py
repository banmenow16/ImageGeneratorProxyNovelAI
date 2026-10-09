"""Advice page (/advice): the character mirror — staged wardrobe edition.

Five composition probes (portrait, from behind, topless, nude, nude behind),
each with up to N retries. Identity tags evolve across stages; composition
tags are per-stage and never saved to the identity block. The final advice
summarizes per-view reliability and produces a structured instruction block
for the RP bot.
"""

from __future__ import annotations

import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import ImageGenerator
from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.review.character import (
	character_advice,
	character_critique,
	stage_compliance_check,
)
from imagine_generator_proxy.server.storage import Database

LOGGER = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "advice.html"

CRITIQUE_MODEL_DEFAULT = "x-ai/grok-4.7"
COMPLIANCE_MODEL_DEFAULT = "qwen/qwen3-vl-32b-instruct"

STAGES = [
	{"id": "portrait", "label": "Portrait", "dress": "keep", "chest": False,
		"composition": "standing, front view, full body, looking at viewer, simple background"},
	{"id": "from_behind", "label": "From Behind", "dress": "keep", "chest": False,
		"composition": "from behind, back view, standing, looking away"},
	{"id": "topless", "label": "Topless", "dress": "topless", "chest": True,
		"composition": "topless, bare breasts, nipples, upper body, front view"},
	{"id": "nude", "label": "Nude (front)", "dress": "nude", "chest": True,
		"composition": "completely nude, bare breasts, nipples, full body, front view"},
	{"id": "nude_behind", "label": "Nude from Behind", "dress": "nude", "chest": False,
		"composition": "completely nude, from behind, back view, full body"},
]

# Garment keywords for per-stage clothing suppression. Word-boundary regex:
# "top" matches "tube top"/"crop top" but not "topless"; "bra" does not match
# "bracelet" or "braids". Jewelry and accessories (tiara, choker, ribbon)
# are identity and never stripped.
_TOP_WORDS = ("top|bra|bikini|bandeau|corset|blouse|shirt|sweater|hoodie|"
	r"jacket|uniform|serafuku|seifuku|kimono|yukata|dress|leotard|bodysuit|"
	r"camisole")
_ALL_WORDS = (_TOP_WORDS + r"|skirt|panties|pants|shorts|stockings|thighhighs|"
	r"socks|shoes|boots|gloves|scarf|cloak|cape|coat|armor|lingerie|clothes|"
	r"clothing|outfit")
TOP_GARMENTS = re.compile(r"\b(?:" + _TOP_WORDS + r")\b")
ALL_GARMENTS = re.compile(r"\b(?:" + _ALL_WORDS + r")\b")

MAX_RETRIES = 10  # per stage — convergence is rejection sampling; more rolls, better odds


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


def _strip_clothing(tags: str, dress: str) -> str:
	"""Drop garment tags from an identity block for undressed stages.

	Signature-outfit tags live in the identity block (correctly), but they
	beat a late-position `nude` tag in NovelAI — Sable's tube top rendered
	over every "nude" round. The identity block itself is unchanged; this
	only affects what a given stage sends to the artist.
	"""
	pattern = {"keep": None, "topless": TOP_GARMENTS, "nude": ALL_GARMENTS}.get(dress)
	if pattern is None:
		return tags
	kept = []
	for tag in tags.split(","):
		bare = tag.strip().lstrip("{").rstrip("}").strip()
		if bare and not pattern.search(bare.lower()):
			kept.append(tag.strip())
	return ", ".join(kept)


def _gate_violations(facts: Optional[dict], stage: dict) -> str:
	"""Judge the gate's neutral observations against the stage contract.
	Code, not the vision model, decides compliance — a primed prompt makes
	the model see what the contract tells it to."""
	if not facts:
		return ""
	problems: list = []
	garments = (facts.get("garments") or "").strip().lower()
	if garments and garments not in ("none", "none.", "n/a", "-"):
		pattern = ALL_GARMENTS if stage.get("dress") == "nude" else TOP_GARMENTS
		bad = [g.strip() for g in garments.split(",")
			if g.strip() and pattern.search(g.strip())]
		if bad:
			problems.append("garments: " + ", ".join(bad))
	nipples = (facts.get("nipples") or "").strip().lower()
	if stage.get("chest") and nipples != "drawn":
		problems.append(f"nipples {nipples or 'unreported'}")
	return "; ".join(problems)


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
	compliance_model = (request.app.get("config") or {}).get("compliance_model",
		COMPLIANCE_MODEL_DEFAULT)

	response = web.StreamResponse(headers={
		"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
	await response.prepare(request)

	async def emit(event: dict) -> None:
		await response.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))

	database = Database(request.app["db_path"])
	database.set_character_description(name, description)

	# start a tracked run with a unique id for folder isolation
	import hashlib
	desc_hash = hashlib.md5(description.encode()).hexdigest()[:8]
	run_id = database.start_advice_run(name, backend_name, desc_hash)
	run_dir = Path(request.app["videos_dir"]).parent / "advice" / "runs" / str(run_id)
	run_dir.mkdir(parents=True, exist_ok=True)

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
		stage_result = {"id": stage_id, "label": stage_label, "status": "pending",
			"attempts": 0, "notes": ""}
		critique: Optional[dict] = None
		stage_db_id = database.start_advice_stage(run_id, stage_id, stage_label)
		await emit({"event": "stage", "phase": "start", "stage": stage_id,
			"label": stage_label, "run_id": run_id})

		for retry in range(MAX_RETRIES):
			stage_result["attempts"] = retry + 1
			# combine identity (clothing suppressed per stage) + composition
			stage_identity = _strip_clothing(identity_tags, stage.get("dress", "keep"))
			full_prompt = f"{stage_identity}, {stage_composition}"
			# which of her tags this view suppresses (so she knows why they're gone)
			kept_keys = {t.strip().lower() for t in stage_identity.split(",") if t.strip()}
			suppressed_tags = ", ".join(t.strip() for t in identity_tags.split(",")
				if t.strip() and t.strip().lower() not in kept_keys)

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

			img_path = run_dir / f"{stage_id}_r{retry + 1}.png"
			img_path.write_bytes(bytes(image.data))
			img_rel = f"runs/{run_id}/{stage_id}_r{retry + 1}.png"
			url = f"/advice-files/runs/{run_id}/{stage_id}_r{retry + 1}.png"
			await emit({"event": "image", "stage": stage_id, "retry": retry + 1,
				"url": url, "prompt": massaged.effective, "run_id": run_id})

			# neutral compliance gate — cheap model reports what's drawn,
			# code judges it against the stage contract, so the persona
			# never wastes her critique on a non-compliant image
			if stage.get("dress") in ("topless", "nude"):
				facts = await stage_compliance_check(or_key, compliance_model,
					bytes(image.data))
				violation = _gate_violations(facts, stage)
				if violation:
					stage_result["notes"] = f"compliance gate: {violation[:70]}"
					await emit({"event": "step", "phase": "gate-retry",
						"stage": stage_id, "retry": retry + 1,
						"violation": violation})
					continue

			await emit({"event": "step", "phase": "critiquing", "stage": stage_id,
				"retry": retry + 1})
			critique = await character_critique(or_key, critique_model, name,
				description, identity_tags, bytes(image.data),
				stage_label=stage_label, stage_composition=stage_composition,
				negative_tags=current_negative,
				suppressed_tags=suppressed_tags)
			if critique is None or (not critique.get("verdict")
					and not critique.get("wrong") and not critique.get("feel")):
				# blank/unparseable critique — treat as a retryable failure
				stage_result["notes"] = "critique failed"
				await emit({"event": "step", "phase": "retrying", "stage": stage_id,
					"retry": retry + 1})
				continue

			all_critiques.append(critique)

			# fold her NEVER-USE tags into the run's negative prompt
			if critique.get("never"):
				merged = ", ".join(filter(None, [current_negative, critique["never"]]))
				current_negative = _dedupe_tags(merged)
			await emit({"event": "critique", "stage": stage_id, "retry": retry + 1,
				**critique})

			# record this round in the database
			database.record_advice_round(stage_db_id, run_id, retry + 1,
				full_prompt, identity_tags,
				critique.get("verdict", ""),
				critique.get("right", ""), critique.get("wrong", ""),
				critique.get("rewritten", ""), critique.get("feel", ""),
				img_rel)

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
			# retries exhausted without convergence (safety/generation breaks
			# set "skipped" themselves and never reach here)
			if stage_result["status"] == "pending":
				stage_result["status"] = "unresolved"
				if not stage_result["notes"]:
					last = all_critiques[-1] if all_critiques else {}
					stage_result["notes"] = last.get("wrong", "did not converge")

		stage_results.append(stage_result)
		attempts_used: int = stage_result["attempts"]  # type: ignore[assignment]
		database.finish_advice_stage(stage_db_id, str(stage_result["status"]),
			attempts_used, str(stage_result["notes"]),
			str(critique.get("verdict", "")) if critique else "")
		await emit({"event": "stage", "phase": "done", "stage": stage_id,
			"status": stage_result["status"], "attempts": stage_result["attempts"]})

	# --- final advice ---
	await emit({"event": "step", "phase": "advising"})
	rounds_summary = "; ".join(
		f"{s['label']}: {s['status']}"
		+ (f" ({s['notes'][:120]})" if s["notes"] else "")
		for s in stage_results)
	converged_count = sum(1 for s in stage_results if s["status"] == "converged")
	advice_text = ""
	try:
		advice_text = await character_advice(or_key, critique_model, name,
			description, rounds_summary, final_tags=identity_tags,
			final_negative=current_negative) or ""
	except Exception:
		# the stages are the expensive part — never lose them to a tail failure
		LOGGER.exception("final advice failed for %s — finalizing run %d without it",
			name, run_id)

	# save to advice_runs + characters
	database.finish_advice_run(run_id, identity_tags, current_negative,
		advice_text, converged_count, len(STAGES))
	try:
		conn = database.connect()
		try:
			conn.execute("UPDATE characters SET canonical_tags = ?, advice = ?,"
				" updated_at = CURRENT_TIMESTAMP WHERE name = ?",
				(identity_tags, advice_text, name))
			conn.commit()
		finally:
			conn.close()
	except Exception:
		LOGGER.exception("characters-table update failed for %s", name)

	await emit({"event": "done", "run_id": run_id,
		"final_prompt": identity_tags,
		"final_negative": current_negative, "advice": advice_text,
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


async def advice_runs_handler(request: web.Request) -> web.Response:
	"""Query past Mirror runs for aggregate analysis."""
	conn = sqlite3.connect(str(request.app["db_path"]))
	conn.row_factory = sqlite3.Row
	try:
		character = request.query.get("character")
		if character:
			runs = conn.execute(
				"SELECT * FROM advice_runs WHERE character = ? ORDER BY id DESC",
				(character,)).fetchall()
		else:
			runs = conn.execute(
				"SELECT * FROM advice_runs ORDER BY id DESC LIMIT 50").fetchall()

		out = []
		for run in runs:
			stages = conn.execute(
				"SELECT stage_id, stage_label, status, attempts, notes, final_verdict"
				" FROM advice_stages WHERE run_id = ? ORDER BY id",
				(run["id"],)).fetchall()
			out.append({
				"run_id": run["id"], "character": run["character"],
				"engine": run["engine"],
				"identity_tags": run["identity_tags"],
				"advice": run["advice"],
				"converged_count": run["converged_count"],
				"total_stages": run["total_stages"],
				"started_at": run["started_at"],
				"finished_at": run["finished_at"],
				"stages": [dict(s) for s in stages],
			})
		return web.json_response({"runs": out})
	finally:
		conn.close()


async def advice_run_detail_handler(request: web.Request) -> web.Response:
	"""Full detail for one run: all stages and every round's prompt/critique."""
	run_id = int(request.match_info["run_id"])
	conn = sqlite3.connect(str(request.app["db_path"]))
	conn.row_factory = sqlite3.Row
	try:
		run = conn.execute("SELECT * FROM advice_runs WHERE id = ?", (run_id,)).fetchone()
		if run is None:
			return web.json_response({"error": "run not found"}, status=404)
		stages = conn.execute(
			"SELECT * FROM advice_stages WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
		rounds = conn.execute(
			"SELECT * FROM advice_rounds WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
		return web.json_response({
			"run": dict(run),
			"stages": [dict(s) for s in stages],
			"rounds": [dict(r) for r in rounds],
		})
	finally:
		conn.close()


def add_routes(app: web.Application) -> None:
	app.router.add_get("/advice", advice_page_handler)
	app.router.add_post("/advice/run/stream", advice_run_stream_handler)
	app.router.add_get("/api/advice/runs", advice_runs_handler)
	app.router.add_get("/api/advice/runs/{run_id}", advice_run_detail_handler)
