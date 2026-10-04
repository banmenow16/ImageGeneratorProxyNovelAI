"""Experiment workbench: run the full prompt pipeline with every step visible.

POST /experiment/run executes, for one prompt:
  1. a RAW baseline generation (legacy prompt assembly, legacy negatives,
     no review) — what the server would have produced pre-massager;
  2. a MASSAGED chain: prompt-massager output, then the vision review ->
     fix -> regenerate loop, one stored image per attempt, the accepted
     attempt flagged as final.

Every image is stored under cache/<character>/experiments/r<run_id>/ and is
served by the existing /cache static route; provenance lives in the
experiment_runs / experiment_images tables. The production /generate path is
untouched — experiments always generate fresh (no cache, no event dedup).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.base import ImageGenerator
from imagine_generator_proxy.prompts.massager import (
	MassagedPrompt,
	PromptMassager,
	build_negative_prompt,
	legacy_assemble,
	sanitize_fix_tags,
)
from imagine_generator_proxy.review.base import NoopReviewer, Reviewer, Verdict
from imagine_generator_proxy.review.character import CharacterReviewer
from imagine_generator_proxy.review.openrouter import OpenRouterReviewer
from imagine_generator_proxy.server.service import sanitize_character
from imagine_generator_proxy.server.storage import Database

LOGGER = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "experiment.html"


def _variant_label(variant: str, fix_applied: str = "") -> str:
	"""Human label for an experiment image card. Shared by the live run and
	the history serializer so the two never drift apart."""
	if variant == "raw":
		return "Raw prompt (legacy assembly, no massager, no review)"
	n = int(variant.split("-")[1]) + 1
	if n == 1:
		return "Attempt 1 — massaged prompt, vision-reviewed"
	suffix = f" (review fixes: {fix_applied})" if fix_applied else " (new seed)"
	return f"Attempt {n} — regenerated after review rejection{suffix}"


class ExperimentRunner:
	"""Runs one full experiment: raw baseline + massaged review chain.

	`on_event` (set per run) receives live progress events as dicts:
	  step   — {phase: generating|reviewing|failed, variant, label, fix_applied}
	  image  — a full variant payload, the moment its image is on disk
	  verdict— {variant, approved, category, notes, is_final}
	  blocked— {rule, reason}
	"""

	def __init__(self, massager: PromptMassager, generator: ImageGenerator,
			reviewer: Optional[Reviewer], config: dict, cache_dir: Path,
			database: Database):
		self._massager = massager
		self._generator = generator
		self._reviewer = reviewer or NoopReviewer()
		self._config = config
		self._cache_dir = cache_dir
		self._database = database
		self._on_event = None

	async def _emit(self, event: dict) -> None:
		if self._on_event is not None:
			try:
				await self._on_event(event)
			except Exception:
				LOGGER.exception("Experiment event emission failed")

	async def run(self, character: str, raw_prompt: str, q_negative: str = "",
			do_massage: bool = True, do_review: bool = True, max_attempts: int = 5,
			on_event=None) -> dict:
		"""Execute one experiment. Returns a JSON-serializable dict.

		The massaged chain iterates until the reviewer approves, the
		age-safety filter trips (fail-closed), generation fails, or the
		attempt cap is reached. `on_event` (async callable) receives live
		progress events; see the class docstring.
		"""
		character = sanitize_character(character)
		review_on = do_review and not isinstance(self._reviewer, NoopReviewer)
		self._on_event = on_event

		try:
			return await self._run(character, raw_prompt, q_negative,
				do_massage, do_review, review_on, max_attempts)
		finally:
			self._on_event = None

	async def _run(self, character: str, raw_prompt: str, q_negative: str,
			do_massage: bool, do_review: bool, review_on: bool, max_attempts: int) -> dict:
		# Safety first, on every path — same policy as /generate.
		blocked = self._massager.check_safety(raw_prompt)
		if blocked is not None:
			LOGGER.warning("Experiment blocked for character=%s rule=%s", character, blocked[0])
			await self._emit({"event": "blocked", "rule": blocked[0], "reason": blocked[1]})
			return {"blocked": {"rule": blocked[0], "reason": blocked[1]}}

		run_id = self._record_run(character, raw_prompt, q_negative)
		variants: list = []

		# --- variant 1: raw baseline (legacy assembly + legacy negatives) ---
		raw_effective = legacy_assemble(raw_prompt)
		raw_negative = build_negative_prompt(q_negative, self._config, legacy=True)
		raw_label = _variant_label("raw")
		await self._emit({"event": "step", "phase": "generating", "variant": "raw", "label": raw_label})
		image = await self._generator.generate(raw_effective, raw_negative, prompt_is_complete=False)
		raw_rel: Optional[str] = None
		if image is not None:
			raw_rel = self._save_image(character, run_id, "raw", bytes(image.data))
		else:
			LOGGER.warning("Experiment %s: raw baseline generation failed (busy)", run_id)
			await self._emit({"event": "step", "phase": "failed", "variant": "raw", "label": raw_label})
		self._record_image(run_id, "raw", raw_effective, raw_negative, raw_rel,
			None, None, "" if raw_rel else "generation failed (NovelAI busy)", "", False)
		raw_variant = {
			"variant": "raw", "label": raw_label,
			"image_url": f"/cache/{raw_rel}" if raw_rel else None,
			"prompt": raw_effective, "negative": raw_negative,
			"approved": None, "category": None, "notes": "", "fix_applied": "", "is_final": False,
		}
		variants.append(raw_variant)
		if raw_rel:
			await self._emit({"event": "image", **raw_variant})

		# --- variants 2..n: massaged chain (review -> fix -> regenerate) ---
		massager_info: Optional[dict] = None
		if do_massage:
			result = self._massager.massage(raw_prompt, character)
			chain_negative = build_negative_prompt(q_negative, self._config,
				extra=result.negative_additions, legacy=False)
			self._update_run_massager(run_id, result, chain_negative)
			massager_info = {
				"version": result.version,
				"effective": result.effective,
				"negative": chain_negative,
				"applied_rules": result.applied_rules,
				"removed": result.removed,
				"negative_additions": result.negative_additions,
			}
			variants += await self._massaged_chain(run_id, character, result,
				chain_negative, review_on, max_attempts)

		return {
			"run_id": run_id,
			"character": character,
			"massager": massager_info,
			"reviewer_used": review_on,
			"variants": variants,
		}

	async def _massaged_chain(self, run_id: int, character: str, result: MassagedPrompt,
			chain_negative: str, review_on: bool, max_attempts: int) -> list:
		variants: list = []
		attempt_prompt = result.effective
		previous_notes = ""
		last_fix_add: list = []
		last_fix_negate: list = []

		for attempt in range(max(1, max_attempts)):
			variant = f"attempt-{attempt}"
			if attempt == 0:
				fix_applied = ""
			else:
				fix_applied = ", ".join(
					[t for t in last_fix_add] + [f"neg:{t}" for t in last_fix_negate])
			label = _variant_label(variant, fix_applied)

			await self._emit({"event": "step", "phase": "generating", "variant": variant,
				"label": label, "fix_applied": fix_applied})
			image = await self._generator.generate(attempt_prompt, chain_negative, prompt_is_complete=True)
			if image is None:
				self._record_image(run_id, variant, attempt_prompt, chain_negative, None,
					None, None, "generation failed (NovelAI busy)", fix_applied, False)
				variants.append({"variant": variant, "label": label, "image_url": None,
					"prompt": attempt_prompt, "negative": chain_negative, "approved": None,
					"category": None, "notes": "generation failed (NovelAI busy)",
					"fix_applied": fix_applied, "is_final": False})
				await self._emit({"event": "step", "phase": "failed", "variant": variant, "label": label})
				break

			rel = self._save_image(character, run_id, variant, bytes(image.data))

			if not review_on:
				self._record_image(run_id, variant, attempt_prompt, chain_negative, rel,
					None, None, "review not enabled", fix_applied, True)
				variants.append({"variant": variant, "label": label, "image_url": f"/cache/{rel}",
					"prompt": attempt_prompt, "negative": chain_negative, "approved": None,
					"category": None, "notes": "reviewer unavailable (no OpenRouter key or review off)",
					"fix_applied": fix_applied, "is_final": True})
				await self._emit({"event": "image", "variant": variant, "label": label,
					"image_url": f"/cache/{rel}", "prompt": attempt_prompt, "negative": chain_negative,
					"approved": None, "category": None,
					"notes": "reviewer unavailable (no OpenRouter key or review off)",
					"fix_applied": fix_applied, "is_final": True})
				break

			await self._emit({"event": "image", "variant": variant, "label": label,
				"image_url": f"/cache/{rel}", "prompt": attempt_prompt, "negative": chain_negative,
				"approved": None, "category": None, "notes": "", "fix_applied": fix_applied,
				"is_final": False})
			await self._emit({"event": "step", "phase": "reviewing", "variant": variant, "label": label})

			verdict: Verdict = await self._reviewer.review(bytes(image.data), attempt_prompt,
				attempt=attempt, previous_notes=previous_notes, character=character)
			LOGGER.info("Experiment %s %s: approved=%s category=%s",
				run_id, variant, verdict.approved, verdict.category)

			at_cap = attempt >= max_attempts - 1
			approved = bool(verdict.approved)
			is_final = approved or at_cap
			if is_final and not approved:
				label += " — attempt cap reached, still age-flagged" if verdict.category == "minor" \
					else " — attempt cap reached, still rejected"
			self._record_image(run_id, variant, attempt_prompt, chain_negative, rel,
				approved, verdict.category, verdict.notes, fix_applied, is_final)
			variants.append({"variant": variant, "label": label, "image_url": f"/cache/{rel}",
				"prompt": attempt_prompt, "negative": chain_negative,
				"approved": approved, "category": verdict.category,
				"notes": verdict.notes or "", "fix_applied": fix_applied, "is_final": is_final})
			await self._emit({"event": "verdict", "variant": variant, "label": label,
				"approved": approved, "category": verdict.category,
				"notes": verdict.notes or "", "is_final": is_final,
				"fix_applied": fix_applied})

			if approved:
				break
			previous_notes = verdict.notes or ""
			if verdict.category == "minor":
				# remediate instead of aborting: apply the age-correction fixes
				# (adult-reinforcing tags) and regenerate. A minor-flagged image
				# is still never approved — if the cap arrives while flagged,
				# the chain ends rejected.
				LOGGER.warning("Experiment %s: minor-flag on attempt %d — regenerating with age fixes",
					run_id, attempt + 1)
			if at_cap:
				LOGGER.warning("Experiment %s: attempt cap (%d) reached without approval", run_id, max_attempts)
				break

			# apply reviewer fixes (composition- and safety-checked), same
			# policy as /generate
			last_fix_add, last_fix_negate = sanitize_fix_tags(verdict.fix_add, verdict.fix_negate, attempt_prompt)
			last_fix_add = [t for t in last_fix_add if self._massager.check_safety(t) is None]
			last_fix_negate = [t for t in last_fix_negate if self._massager.check_safety(t) is None]
			if last_fix_add:
				attempt_prompt = attempt_prompt + ", " + ", ".join(last_fix_add)
			if last_fix_negate:
				chain_negative = chain_negative + ", " + ", ".join(last_fix_negate)

		return variants

	# --- persistence helpers ---

	def _connect(self) -> sqlite3.Connection:
		return self._database.connect()

	def _record_run(self, character: str, raw_prompt: str, raw_negative: str) -> int:
		conn = self._connect()
		try:
			cur = conn.execute(
				"INSERT INTO experiment_runs (character, raw_prompt, raw_negative) VALUES (?, ?, ?)",
				(character, raw_prompt, raw_negative),
			)
			conn.commit()
			return int(cur.lastrowid or 0)
		finally:
			conn.close()

	def _update_run_massager(self, run_id: int, result: MassagedPrompt, negative: str) -> None:
		conn = self._connect()
		try:
			conn.execute(
				"UPDATE experiment_runs SET effective_prompt=?, effective_negative=?, massager_version=?, "
				"applied_rules=?, removed_tags=? WHERE id=?",
				(result.effective, negative, result.version,
					", ".join(result.applied_rules), ", ".join(result.removed), run_id),
			)
			conn.commit()
		finally:
			conn.close()

	def _record_image(self, run_id: int, variant: str, prompt_used: str, negative_used: str,
			image_rel_path: Optional[str], approved: Optional[bool], category: Optional[str],
			notes: str, fix_applied: str, is_final: bool) -> int:
		conn = self._connect()
		try:
			cur = conn.execute(
				"INSERT INTO experiment_images (run_id, variant, prompt_used, negative_used, image_path, "
				"approved, category, notes, fix_applied, is_final) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
				(run_id, variant, prompt_used, negative_used, image_rel_path,
					None if approved is None else (1 if approved else 0),
					category, (notes or "")[-1000:], fix_applied, 1 if is_final else 0),
			)
			conn.commit()
			return int(cur.lastrowid or 0)
		finally:
			conn.close()

	def _save_image(self, character: str, run_id: int, variant: str, data: bytes) -> str:
		run_dir = self._cache_dir / character / "experiments" / f"r{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)
		(run_dir / f"{variant}.png").write_bytes(data)
		return f"{character}/experiments/r{run_id}/{variant}.png"


# --- serializers ---

def _variants_payload(conn: sqlite3.Connection, run_id: int) -> list:
	rows = conn.execute(
		"SELECT variant, prompt_used, negative_used, image_path, approved, category, notes, fix_applied, is_final "
		"FROM experiment_images WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
	out: list = []
	for variant, prompt, negative, rel, approved, category, notes, fix_applied, is_final in rows:
		label = _variant_label(variant, fix_applied or "")
		# a terminal still-rejected card means the attempt cap hit
		if is_final and approved == 0 and variant != "raw":
			label += " — attempt cap reached, still age-flagged" if category == "minor" \
				else " — attempt cap reached, still rejected"
		out.append({
			"variant": variant, "label": label,
			"image_url": f"/cache/{rel}" if rel else None,
			"prompt": prompt or "", "negative": negative or "",
			"approved": None if approved is None else bool(approved),
			"category": category, "notes": notes or "",
			"fix_applied": fix_applied or "", "is_final": bool(is_final),
		})
	return out


def _run_payload(conn: sqlite3.Connection, run_id: int) -> Optional[dict]:
	row = conn.execute(
		"SELECT id, character, raw_prompt, raw_negative, effective_prompt, effective_negative, "
		"massager_version, applied_rules, removed_tags, created_at FROM experiment_runs WHERE id = ?",
		(run_id,)).fetchone()
	if row is None:
		return None
	(run_id, character, raw_prompt, raw_negative, effective, effective_neg, version,
		applied, removed, created_at) = row
	return {
		"run_id": run_id, "character": character,
		"raw_prompt": raw_prompt, "raw_negative": raw_negative or "",
		"massager": {
			"version": version, "effective": effective or "", "negative": effective_neg or "",
			"applied_rules": (applied or "").split(", ") if applied else [],
			"removed": (removed or "").split(", ") if removed else [],
		} if effective else None,
		"created_at": created_at,
		"variants": _variants_payload(conn, run_id),
	}


# --- HTTP handlers (thin adapters) ---

def _runner_from_app(app: web.Application, backend_name: Optional[str] = None,
		character_name: str = "default") -> ExperimentRunner:
	backends: dict = app.get("backends") or {}
	backend = backends.get(backend_name) if backend_name else None
	if backend is None:
		backend = backends.get(app.get("generation_backend", "novelai")) \
			or backends.get("novelai")
	assert backend is not None, "no generation backends registered"
	reviewer: Optional[Reviewer] = app.get("experiment_reviewer")
	character = Database(app["db_path"]).get_character(character_name)
	if reviewer is not None and character and character.get("description"):
		reviewer = CharacterReviewer(reviewer, character_name, character["description"])
	return ExperimentRunner(
		massager=backend.massager,
		generator=backend.generator,
		reviewer=reviewer,
		config=app.get("config", {}),
		cache_dir=app["cache_dir"],
		database=Database(app["db_path"]),
	)


async def experiment_page_handler(request: web.Request) -> web.Response:
	html = PAGE_PATH.read_text(encoding="utf-8")
	return web.Response(text=html, content_type="text/html")


async def _parse_run_body(request: web.Request):
	"""Common request parsing for the run endpoints.

	Returns (params dict, backend name, error response)."""
	try:
		body = await request.json()
	except Exception:
		return None, None, web.json_response({"error": "request body must be JSON"}, status=400)

	raw_prompt = (body.get("prompt") or "").strip()
	if not raw_prompt:
		return None, web.json_response({"error": "missing 'prompt'"}, status=400)

	def _parse_attempts(body: dict) -> int:
		"""Max generation attempts (1-10); accepts the legacy 'retries' key
		as retries+1 for older callers. Defaults to 5."""
		try:
			if body.get("attempts"):
				return max(1, min(int(body["attempts"]), 10))
			if body.get("retries") is not None:
				return max(1, min(int(body["retries"]) + 1, 10))
		except (TypeError, ValueError):
			pass
		return 5

	params = {
		"character": body.get("character") or "default",
		"raw_prompt": raw_prompt,
		"q_negative": body.get("negative") or "",
		"do_massage": bool(body.get("massage", True)),
		"do_review": bool(body.get("review", True)),
		"max_attempts": _parse_attempts(body),
	}
	return params, body.get("backend"), None


async def experiment_run_handler(request: web.Request) -> web.Response:
	params, backend, error = await _parse_run_body(request)
	if error is not None:
		return error

	result = await _runner_from_app(request.app, backend,
		str(params.get("character") or "default")).run(**params)
	if "blocked" in result:
		return web.json_response({"error": "prompt blocked by safety filter", **result["blocked"]}, status=400)
	result["reviewer_available"] = bool(request.app.get("experiment_reviewer"))
	return web.json_response(result)


async def experiment_run_stream_handler(request: web.Request) -> web.StreamResponse:
	"""Live version of /experiment/run: newline-delimited JSON events as the
	pipeline progresses (step / image / verdict / blocked), then one final
	'done' event carrying the full run payload."""
	params, backend, error = await _parse_run_body(request)
	if error is not None:
		return error

	response = web.StreamResponse(headers={
		"Content-Type": "application/x-ndjson",
		"Cache-Control": "no-cache",
		"X-Accel-Buffering": "no",
	})
	await response.prepare(request)

	async def on_event(event: dict) -> None:
		await response.write((json.dumps(event) + "\n").encode("utf-8"))

	result = await _runner_from_app(request.app, backend,
		str(params.get("character") or "default")).run(**params, on_event=on_event)
	result["reviewer_available"] = bool(request.app.get("experiment_reviewer"))
	await response.write((json.dumps({"event": "done", "run": result}) + "\n").encode("utf-8"))
	await response.write_eof()
	return response


async def experiment_runs_handler(request: web.Request) -> web.Response:
	try:
		limit = max(1, min(int(request.query.get("limit", 20)), 100))
	except ValueError:
		limit = 20
	conn = sqlite3.connect(str(request.app["db_path"]))
	try:
		runs = []
		for (run_id,) in conn.execute(
			"SELECT id FROM experiment_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall():
			payload = _run_payload(conn, run_id)
			if payload:
				runs.append(payload)
	finally:
		conn.close()
	return web.json_response({
		"runs": runs,
		"reviewer_available": bool(request.app.get("experiment_reviewer")),
	})


async def experiment_run_detail_handler(request: web.Request) -> web.Response:
	try:
		run_id = int(request.match_info["run_id"])
	except ValueError:
		return web.json_response({"error": "invalid run id"}, status=400)
	conn = sqlite3.connect(str(request.app["db_path"]))
	try:
		payload = _run_payload(conn, run_id)
	finally:
		conn.close()
	if payload is None:
		return web.json_response({"error": "run not found"}, status=404)
	return web.json_response(payload)


async def character_description_handler(request: web.Request) -> web.Response:
	name = request.match_info["name"]
	database = Database(request.app["db_path"])
	if request.method == "GET":
		character = database.get_character(name)
		return web.json_response({"name": name,
			"description": (character or {}).get("description") or ""})
	body = await request.json()
	description = str(body.get("description") or "").strip()
	if not description:
		return web.json_response({"error": "description must be non-empty"}, status=400)
	database.set_character_description(name, description)
	return web.json_response({"name": name, "description": description, "saved": True})


def add_routes(app: web.Application) -> None:
	app.router.add_get("/api/characters/{name}/description", character_description_handler)
	app.router.add_post("/api/characters/{name}/description", character_description_handler)
	app.router.add_get("/experiment", experiment_page_handler)
	app.router.add_post("/experiment/run", experiment_run_handler)
	app.router.add_post("/experiment/run/stream", experiment_run_stream_handler)
	app.router.add_get("/experiment/runs", experiment_runs_handler)
	app.router.add_get("/experiment/run/{run_id}", experiment_run_detail_handler)


def build_experiment_reviewer(config: dict, db_path) -> Optional[OpenRouterReviewer]:
	"""Reviewer for experiments: runs whenever an OpenRouter key exists,
	independently of the production `vision_review` flag (experiments are
	explicit, user-triggered actions)."""
	return OpenRouterReviewer.from_config(config, db_path=db_path)
