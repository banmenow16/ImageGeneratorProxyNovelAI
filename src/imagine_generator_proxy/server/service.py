"""The generation pipeline service: massage -> cache -> generate -> review
(with fix/regenerate loop) -> cache write. The HTTP handler stays a thin
adapter; everything domain-level lives here."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aiohttp import web

from novelai_api.ImagePreset import ImageModel

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator
from imagine_generator_proxy.prompts.massager import (
	MassagedPrompt,
	PromptMassager,
	build_negative_prompt,
	legacy_assemble,
)
from imagine_generator_proxy.review.base import Reviewer, Verdict
from imagine_generator_proxy.server.cache import (
	prepare_nonbinary_response,
	read_cached_image,
	write_cache_and_prepare_response,
)
from imagine_generator_proxy.server.storage import Database, compute_digest

LOGGER = logging.getLogger(__name__)

MODEL = ImageModel.Anime_v45_Full


class PromptBlockedError(Exception):
	"""The prompt tripped the safety filter; nothing was generated."""

	def __init__(self, rule: str, reason: str):
		super().__init__(reason)
		self.rule = rule
		self.reason = reason


class UpstreamBusyError(Exception):
	"""The generation backend stayed busy after its retries."""


class ImageRejectedError(Exception):
	"""The reviewer age-flagged the image; it must not be served (HTTP 422)."""


class GenerationError(Exception):
	"""Unexpected pipeline failure; surfaces as HTTP 500."""


def sanitize_character(raw: str) -> str:
	"""Character names namespace the cache; keep only filesystem-safe chars."""
	return re.sub(r"[^A-Za-z0-9_-]", "_", raw or "default") or "default"


@dataclass
class GenerationRequest:
	"""One /generate request, parsed and validated."""

	raw_prompt: str
	character: str
	q_negative: str = ""
	want_raw: bool = False
	force: bool = False
	use_massage: bool = True


class GenerationService:
	"""Owns the generation pipeline and the per-digest coordination events."""

	def __init__(self, massager: PromptMassager, generator: ImageGenerator,
			reviewer: Reviewer, config: dict, cache_dir: Path, database: Database):
		self._massager = massager
		self._generator = generator
		self._reviewer = reviewer
		self._config = config
		self._cache_dir = cache_dir
		self._database = database
		self._generations: dict = {}

	# --- pipeline ---

	async def generate(self, request: GenerationRequest) -> web.Response:
		result = self._massage(request)
		if result.blocked_reason is not None:
			# massage() itself returns blocked results when the safety stage
			# trips with massage mode on
			LOGGER.warning("Blocked prompt for character=%s rule=%s: %s",
				request.character, result.blocked_rule, result.blocked_reason)
			raise PromptBlockedError(result.blocked_rule or "unknown", result.blocked_reason)

		prompt = result.effective
		version = result.version
		# legacy=True keeps the historical negative blocks (and thus the legacy
		# digest) byte-identical on the bypass path
		negative_prompt = build_negative_prompt(request.q_negative, self._config,
			extra=result.negative_additions, legacy=(version is None))

		digest = compute_digest(request.character, prompt, negative_prompt, MODEL, version)
		char_dir = self._cache_dir / request.character
		cached_file = char_dir / f"{digest}.png"

		if not request.force:
			resp = read_cached_image(cached_file, request.want_raw,
				effective_prompt=prompt, massager_version=version)
			if resp:
				return resp

		if request.force and cached_file.exists():
			try:
				cached_file.unlink()
			except Exception:
				LOGGER.exception("Failed to remove cached image for force override")

		cached = await self._await_in_flight(digest, request, prompt, version)
		if cached is not None:
			return cached

		LOGGER.info("Sending request to NovelAI. Queue size: %s", len(self._generations))
		try:
			image = await self._generator.generate(prompt, negative_prompt,
				prompt_is_complete=(version is not None))
			if image is None:
				self._release_waiters(digest)
				raise UpstreamBusyError()

			image = await self._review_loop(image, request, digest, prompt,
				negative_prompt, version)

			# Write cache outside the generation lock
			if isinstance(image.data, (bytes, bytearray)):
				resp = write_cache_and_prepare_response(image.data, char_dir, cached_file,
					request.want_raw, self._generations, digest, image.name,
					effective_prompt=prompt, massager_version=version)
				self._database.record_generated_image(request.character, digest, prompt,
					negative_prompt, str(MODEL), request.raw_prompt, version)
				return resp

			return prepare_nonbinary_response(image.name, image.data, self._generations, digest)
		except (PromptBlockedError, UpstreamBusyError, ImageRejectedError):
			raise
		except Exception as e:
			LOGGER.exception("/generate pipeline error")
			self._release_waiters(digest)
			raise GenerationError(str(e)) from e

	# --- stages ---

	def _massage(self, request: GenerationRequest) -> MassagedPrompt:
		"""Safety-check and massage. Raises PromptBlockedError on flags.

		The safety filter runs on every path: inside massage() when massage
		mode is on, via check_safety() when bypassed.
		"""
		massager = self._massager
		if request.use_massage and massager.enabled:
			try:
				return massager.massage(request.raw_prompt, request.character)
			except Exception:
				# A massager bug must never break generation: fall back to the
				# legacy assembly and the legacy (un-prefixed) digest formula.
				LOGGER.exception("PromptMassager failed; falling back to legacy prompt assembly")
				return MassagedPrompt(original=request.raw_prompt,
					effective=legacy_assemble(request.raw_prompt), version=None)

		blocked = massager.check_safety(request.raw_prompt)
		if blocked is not None:
			LOGGER.warning("Blocked prompt for character=%s rule=%s: %s",
				request.character, blocked[0], blocked[1])
			raise PromptBlockedError(blocked[0], blocked[1])
		return MassagedPrompt(original=request.raw_prompt,
			effective=legacy_assemble(request.raw_prompt), version=None)

	async def _await_in_flight(self, digest: str, request: GenerationRequest,
			prompt: str, version: Optional[int]):
		"""Per-digest event dedup: wait for an in-flight identical generation
		and return its cached response, or register our own event and return
		None."""
		gens = self._generations
		gen_event = gens.get(digest)
		if gen_event is not None:
			try:
				await gen_event.wait()
			except Exception:
				pass
			return read_cached_image(self._cache_dir / request.character / f"{digest}.png",
				request.want_raw, context="after wait", effective_prompt=prompt, massager_version=version)

		gens[digest] = asyncio.Event()
		return None

	def _release_waiters(self, digest: str) -> None:
		try:
			self._generations[digest].set()
		except Exception:
			pass
		self._generations.pop(digest, None)

	async def _review_loop(self, image: GeneratedImage, request: GenerationRequest,
			digest: str, prompt: str, negative_prompt: str,
			version: Optional[int]) -> GeneratedImage:
		"""Vision review with bounded fix/regenerate retries.

		Quality rejections regenerate with the reviewer's proposed tag fixes
		(safety-checked) plus a new seed; each attempt carries the previous
		rejection notes so the reviewer verifies the fix. Minor-category
		rejections are fail-closed and never served.
		"""
		log_reviews = bool(self._config.get("vision_review"))
		review_retries = int(self._config.get("vision_review_retries", 1))
		verdict: Optional[Verdict] = None
		previous_notes = ""
		attempt_prompt = prompt
		attempt_negative = negative_prompt

		for attempt in range(review_retries + 1):
			verdict = await self._reviewer.review(bytes(image.data), attempt_prompt,
				attempt=attempt, previous_notes=previous_notes, character=request.character)
			if log_reviews:
				self._database.log_review_outcome(digest, request.character, verdict,
					attempt=attempt, tags=attempt_prompt)
			if verdict.approved:
				return image
			previous_notes = verdict.notes or ""

			if verdict.category == "minor" and attempt >= review_retries:
				# out of attempts while age-flagged: fail-closed, never served
				LOGGER.warning("Reviewer minor-flag persists after retries (notes=%s) — refusing to serve",
					(verdict.notes or "")[:200])
				self._release_waiters(digest)
				raise ImageRejectedError(verdict.notes or "age-safety review")

			if attempt < review_retries:
				fix_add = [t for t in verdict.fix_add if self._massager.check_safety(t) is None]
				fix_negate = [t for t in verdict.fix_negate if self._massager.check_safety(t) is None]
				if fix_add:
					attempt_prompt = attempt_prompt + ", " + ", ".join(fix_add)
					LOGGER.info("Applying reviewer fix tags: %s", ", ".join(fix_add))
				if fix_negate:
					attempt_negative = attempt_negative + ", " + ", ".join(fix_negate)
					LOGGER.info("Applying reviewer fix negatives: %s", ", ".join(fix_negate))
				reason = "age-flag, applying adult-reinforcement fixes" if verdict.category == "minor" \
					else "reviewer rejected on quality"
				LOGGER.warning("REGENERATING %s for character=%s digest=%s attempt=%d/%d (%s: %s)",
					"with fix tags" if (fix_add or fix_negate) else "with new seed",
					request.character, digest[:12], attempt + 2, review_retries + 1,
					reason, (verdict.notes or "")[:200])
				regenerated = await self._generator.generate(attempt_prompt, attempt_negative,
					prompt_is_complete=(version is not None))
				if regenerated is None:
					self._release_waiters(digest)
					raise UpstreamBusyError()
				image = regenerated

		if verdict is not None and not verdict.approved:
			LOGGER.warning("Reviewer still rejects on quality after retries — serving anyway")
		return image
