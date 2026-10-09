"""Advice iteration tools: quick-test, cross-check, failure aggregation.

These endpoints make Mirror (/advice) iteration fast — single-stage tests,
neutral re-reviews of persona critiques, and aggregate failure analysis.
Same routes as before, now a proper controller class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.daos.advice import AdviceDAO
from imagine_generator_proxy.generation.base import Backend
from imagine_generator_proxy.review.openrouter import OpenRouterReviewer, parse_verdict
from imagine_generator_proxy.controllers.base import bad_request, json_body, not_found
from imagine_generator_proxy.server.advice import STAGES

LOGGER = logging.getLogger(__name__)

CRITIQUE_MODEL_DEFAULT = "x-ai/grok-4.7"


class AdviceToolsController:
	"""Iteration endpoints over Mirror session data."""

	def __init__(self, config: dict, backends: dict[str, Backend],
			default_engine: str, db_path: Path, data_dir: Path) -> None:
		self._config = config
		self._backends = backends
		self._default_engine = default_engine
		self._dao = AdviceDAO(db_path)
		self._data_dir = data_dir

	def add_routes(self, router: web.UrlDispatcher) -> None:
		router.add_post("/api/advice/quick-test", self.quick_test)
		router.add_get("/api/advice/rounds/{round_id}/crosscheck", self.crosscheck)
		router.add_get("/api/advice/failures", self.failures)

	# --- quick-test: one stage, one round ---

	async def quick_test(self, request: web.Request) -> web.Response:
		body = await json_body(request)
		if body is None:
			return bad_request("request body must be a JSON object")
		name = str(body.get("character") or "").strip()
		description = str(body.get("description") or "").strip()
		if not name or not description:
			return bad_request("character and description required")
		identity_tags = str(body.get("identity_tags") or "").strip()
		if not identity_tags:
			return bad_request("identity_tags required"
				" (derive via /advice first, or provide manually)")

		stage_id = str(body.get("stage", "portrait"))
		stage = next((s for s in STAGES if s["id"] == stage_id), STAGES[0])

		engine = str(body.get("engine") or self._default_engine)
		backend: Optional[Backend] = self._backends.get(engine)
		if backend is None:
			return bad_request(f"engine {engine!r} unavailable",
				available=list(self._backends))

		or_key = self._config.get("openRouterToken")
		if not or_key:
			return bad_request("server has no openRouterToken configured")
		critique_model = self._config.get("critique_model", CRITIQUE_MODEL_DEFAULT)

		from imagine_generator_proxy.review.character import character_critique
		from imagine_generator_proxy.server.advice import _strip_clothing

		full_prompt = (f"{_strip_clothing(identity_tags, stage.get('dress', 'keep'))},"
			f" {stage['composition']}")
		massaged = backend.massager.massage(full_prompt, name)
		if massaged.blocked_reason:
			return bad_request(f"blocked: {massaged.blocked_reason}")

		image = await backend.generator.generate(massaged.effective, "",
			prompt_is_complete=True)
		if image is None:
			return web.json_response({"error": "generation failed"}, status=502)

		quick_dir = self._data_dir / "advice" / "quicktest"
		quick_dir.mkdir(parents=True, exist_ok=True)
		img_path = quick_dir / f"{name}_{stage['id']}.png"
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

	# --- crosscheck: neutral re-review of one round ---

	async def crosscheck(self, request: web.Request) -> web.Response:
		try:
			round_id = int(request.match_info["round_id"])
		except ValueError:
			return bad_request("invalid round id")
		round_data = self._dao.get_round(round_id)
		if round_data is None:
			return not_found("round not found")

		image_path = self._data_dir / "advice" / (round_data["image_path"] or "")
		if not round_data["image_path"] or not image_path.exists():
			return not_found(f"image not found: {round_data['image_path']}")

		reviewer = OpenRouterReviewer.from_config(self._config)
		if reviewer is None:
			return web.json_response({"error": "reviewer unavailable"}, status=500)

		neutral_verdict = await reviewer.review(image_path.read_bytes(),
			round_data["full_prompt"] or "")
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

	# --- failures: aggregate over unresolved stages ---

	async def failures(self, request: web.Request) -> web.Response:
		character = request.query.get("character")
		rows = self._dao.aggregate_failures(character)
		return web.json_response({
			"failures": [dict(r) for r in rows],
			"count": len(rows),
		})
